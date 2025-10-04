# src/train.py
import os
import time
import csv
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import PigsDataset, collate_fn
from transforms import get_transforms
from model import get_fasterrcnn_r50_fpn


class SmoothedValue:
    """簡單的指數平滑，讓進度條上的 loss 比較穩定。"""
    def __init__(self, alpha=0.9):
        self.alpha = alpha
        self.avg = None

    def update(self, v: float):
        if self.avg is None:
            self.avg = v
        else:
            self.avg = self.alpha * self.avg + (1 - self.alpha) * v

    def value(self):
        return float(self.avg) if self.avg is not None else float("nan")


def train_one_epoch(model, loader, optimizer, device, epoch, log_writer=None, grad_clip=10.0):
    model.train()
    total_loss, num_batches = 0.0, 0
    smooth_total = SmoothedValue(alpha=0.9)
    smooth_rpn_obj = SmoothedValue(alpha=0.9)
    smooth_rpn_reg = SmoothedValue(alpha=0.9)
    smooth_cls     = SmoothedValue(alpha=0.9)
    smooth_box_reg = SmoothedValue(alpha=0.9)

    start_epoch = time.perf_counter()
    pbar = tqdm(loader, total=len(loader), ncols=120, desc=f"Epoch {epoch:02d}")

    for images, targets in pbar:
        batch_start = time.perf_counter()

        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)  # dict: loss_classifier, loss_box_reg, loss_objectness, loss_rpn_box_reg

        # 檢查每個 loss 是否為有限值
        bad = False
        for k, v in loss_dict.items():
            if not torch.isfinite(v):
                ids = [int(t.get("image_id", torch.tensor(-1)).item()) for t in targets]
                print(f"[Warn] {k} is {v.item()} on images {ids}. Skip this batch.")
                bad = True
                break
        if bad:
            continue

        loss = sum(loss_dict.values())
        if not torch.isfinite(loss):
            ids = [int(t.get("image_id", torch.tensor(-1)).item()) for t in targets]
            print(f"[Warn] total loss is NaN/Inf on images {ids}. Skip.")
            continue

        optimizer.zero_grad()
        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=grad_clip
            )
        optimizer.step()

        # 統計
        loss_item = float(loss.item())
        total_loss += loss_item
        num_batches += 1

        # 更新平滑顯示
        smooth_total.update(loss_item)
        smooth_rpn_obj.update(float(loss_dict["loss_objectness"].item()))
        smooth_rpn_reg.update(float(loss_dict["loss_rpn_box_reg"].item()))
        smooth_cls.update(float(loss_dict["loss_classifier"].item()))
        smooth_box_reg.update(float(loss_dict["loss_box_reg"].item()))

        # 速度 & ETA
        batch_time = time.perf_counter() - batch_start
        lr = optimizer.param_groups[0]["lr"]
        pbar.set_postfix({
            "loss": f"{smooth_total.value():.3f}",
            "rpn_obj": f"{smooth_rpn_obj.value():.3f}",
            "rpn_reg": f"{smooth_rpn_reg.value():.3f}",
            "cls": f"{smooth_cls.value():.3f}",
            "box": f"{smooth_box_reg.value():.3f}",
            "bt": f"{batch_time*1000:.0f}ms",
            "lr": f"{lr:.3e}",
        })

        # 逐批寫入 CSV（可選）
        if log_writer is not None:
            log_writer.writerow({
                "epoch": epoch,
                "iter": num_batches,
                "lr": lr,
                "loss_total": loss_item,
                "loss_rpn_objectness": float(loss_dict["loss_objectness"].item()),
                "loss_rpn_box_reg": float(loss_dict["loss_rpn_box_reg"].item()),
                "loss_classifier": float(loss_dict["loss_classifier"].item()),
                "loss_box_reg": float(loss_dict["loss_box_reg"].item()),
                "batch_time_sec": batch_time,
            })

    epoch_time = time.perf_counter() - start_epoch
    avg_loss = total_loss / max(1, num_batches)
    imgs_seen = num_batches * getattr(loader.dataset, "_batch_size_hint", 1)
    speed = (len(loader.dataset) / epoch_time) if epoch_time > 0 else 0.0

    print(f"[Epoch {epoch:02d}] loss={avg_loss:.4f} | time={epoch_time:.1f}s | speed={speed:.1f} img/s")

    return avg_loss, epoch_time


def quick_sanity_check(model, device):
    # 用 batch_size=1 快速測試 forward/backward 是否正常
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

    # === 模型 ===
    model = get_fasterrcnn_r50_fpn(num_classes=2, freeze_backbone=True).to(device)

    # === Sanity Check（小 batch）===
    quick_sanity_check(model, device)

    # === 資料 ===
    train_ds = PigsDataset("data/train/img", "data/train/gt.txt",
                           transforms=get_transforms(train=True, max_side=800))
    # hint 給速度估算用（非必要）
    setattr(train_ds, "_batch_size_hint", 4)

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True,
                              num_workers=4, collate_fn=collate_fn, pin_memory=True)

    # === 優化器與學習率排程 ===
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    os.makedirs("experiments/logs", exist_ok=True)

    # 準備 CSV 紀錄
    csv_path = "experiments/logs/train_log.csv"
    write_header = not os.path.isfile(csv_path)
    csv_file = open(csv_path, "a", newline="")
    log_writer = csv.DictWriter(csv_file, fieldnames=[
        "epoch", "iter", "lr",
        "loss_total",
        "loss_rpn_objectness", "loss_rpn_box_reg",
        "loss_classifier", "loss_box_reg",
        "batch_time_sec"
    ])
    if write_header:
        log_writer.writeheader()

    # === 正式訓練 ===
    best_loss = float("inf")
    EPOCHS = 3
    for epoch in range(EPOCHS):
        avg_loss, epoch_time = train_one_epoch(model, train_loader, optimizer, device, epoch, log_writer)
        scheduler.step()

        # 儲存每個 epoch 的權重
        ckpt_path = f"experiments/logs/fasterrcnn_r50fpn_e{epoch}.pth"
        torch.save(model.state_dict(), ckpt_path)
        print(f"[Save] {ckpt_path}")

        # 保留最佳
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "experiments/logs/best_model.pth")
            print(f"[Best] Updated best_model.pth (loss={best_loss:.4f})")

    csv_file.close()


if __name__ == "__main__":
    main()
