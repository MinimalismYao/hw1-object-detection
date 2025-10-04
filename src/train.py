# src/train.py
import os
import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import PigsDataset, collate_fn
from transforms import get_transforms
from model import get_fasterrcnn_r50_fpn


# ========= 可自行修改的設定 =========
EPOCHS      = 1
BATCH_SIZE  = 4
MAX_SIDE    = 800
LR          = 0.005
WEIGHT_DECAY= 1e-4
STEP_SIZE   = 5
GAMMA       = 0.1
CKPT_DIR    = "experiments/logs"
CKPT_NAME   = "fasterrcnn_r50fpn_final.pth"   # 只保存最後一個權重
# ==================================


class SmoothedValue:
    def __init__(self, alpha=0.9):
        self.alpha = alpha
        self.avg = None
    def update(self, v: float):
        self.avg = v if self.avg is None else self.alpha * self.avg + (1 - self.alpha) * v
    def value(self):
        return float("nan") if self.avg is None else float(self.avg)


def train_one_epoch(model, loader, optimizer, device, epoch, grad_clip=10.0):
    model.train()
    total_loss, num_batches = 0.0, 0

    s_total = SmoothedValue(0.9)
    s_rpn_obj = SmoothedValue(0.9)
    s_rpn_reg = SmoothedValue(0.9)
    s_cls     = SmoothedValue(0.9)
    s_box     = SmoothedValue(0.9)

    epoch_t0 = time.perf_counter()
    pbar = tqdm(loader, total=len(loader), ncols=120, desc=f"Epoch {epoch+1}/{EPOCHS}")

    for images, targets in pbar:
        bt0 = time.perf_counter()

        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        if any([not torch.isfinite(v) for v in loss_dict.values()]):
            print(f"[Warn] Non-finite loss, skip batch.")
            continue

        loss = sum(loss_dict.values())
        if not torch.isfinite(loss):
            print(f"[Warn] total loss is NaN/Inf, skip batch.")
            continue

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=grad_clip
            )
        optimizer.step()

        l_total = float(loss.item())
        total_loss += l_total
        num_batches += 1

        s_total.update(l_total)
        s_rpn_obj.update(float(loss_dict["loss_objectness"].item()))
        s_rpn_reg.update(float(loss_dict["loss_rpn_box_reg"].item()))
        s_cls.update(float(loss_dict["loss_classifier"].item()))
        s_box.update(float(loss_dict["loss_box_reg"].item()))

        bt = time.perf_counter() - bt0
        lr = optimizer.param_groups[0]["lr"]
        pbar.set_postfix({
            "loss": f"{s_total.value():.3f}",
            "rpn_obj": f"{s_rpn_obj.value():.3f}",
            "rpn_reg": f"{s_rpn_reg.value():.3f}",
            "cls": f"{s_cls.value():.3f}",
            "box": f"{s_box.value():.3f}",
            "bt": f"{bt*1000:.0f}ms",
            "lr": f"{lr:.3e}",
        })

    epoch_time = time.perf_counter() - epoch_t0
    avg_loss = total_loss / max(1, num_batches)
    print(f"[Epoch {epoch+1}/{EPOCHS}] avg_loss={avg_loss:.4f} | time={epoch_time:.1f}s")
    return avg_loss


def quick_sanity_check(model, device):
    ds = PigsDataset("data/train/img", "data/train/gt.txt",
                     transforms=get_transforms(train=True, max_side=640))
    loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=2, collate_fn=collate_fn)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.SGD(params, lr=0.01, momentum=0.9)
    images, targets = next(iter(loader))
    images = [images[0].to(device)]
    targets = [{k: v.to(device) for k, v in targets[0].items()}]
    loss = sum(model(images, targets).values())
    optim.zero_grad()
    loss.backward()
    optim.step()
    print(f"[Sanity] one-step OK, loss={loss.item():.4f}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    model = get_fasterrcnn_r50_fpn(num_classes=2, freeze_backbone=True).to(device)

    quick_sanity_check(model, device)

    train_ds = PigsDataset("data/train/img", "data/train/gt.txt",
                           transforms=get_transforms(train=True, max_side=MAX_SIDE))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, collate_fn=collate_fn)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LR, momentum=0.9, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    os.makedirs(CKPT_DIR, exist_ok=True)

    for epoch in range(EPOCHS):
        avg_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        scheduler.step()

    # 只存最後一個
    ckpt_path = os.path.join(CKPT_DIR, CKPT_NAME)
    torch.save(model.state_dict(), ckpt_path)
    print(f"[Save Final] {ckpt_path}")


if __name__ == "__main__":
    main()
