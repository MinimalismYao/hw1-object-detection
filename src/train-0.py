# src/train.py
import os, torch
from torch.utils.data import DataLoader
from dataset import PigsDataset, collate_fn
from transforms import get_transforms
from model import get_fasterrcnn_r50_fpn


def train_one_epoch(model, loader, optimizer, device, print_every: int = 50):
    model.train()
    total_loss, nb = 0.0, 0

    for step, (images, targets) in enumerate(loader):
        # 移到 GPU
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # === 前向與 loss ===
        loss_dict = model(images, targets)  # dict: 4 個子 loss

        # 檢查每個 loss 是否為有限值
        bad = False
        for k, v in loss_dict.items():
            if not torch.isfinite(v):
                ids = [int(t["image_id"].item()) for t in targets if "image_id" in t]
                print(f"[Warn] {k} is {v.item()} on images {ids}. Skip this batch.")
                bad = True
                break
        if bad:
            continue

        # 顯示每批次的子 loss（方便監控）
        losses = {k: v.item() for k, v in loss_dict.items()}
        total = sum(losses.values())
        if (step % print_every) == 0:
            print(
                f"rpn_obj={losses['loss_objectness']:.3f}  "
                f"rpn_reg={losses['loss_rpn_box_reg']:.3f}  "
                f"cls={losses['loss_classifier']:.3f}  "
                f"box_reg={losses['loss_box_reg']:.3f}  "
                f"total={total:.3f}"
            )

        loss = sum(loss_dict.values())
        if not torch.isfinite(loss):
            ids = [int(t["image_id"].item()) for t in targets if "image_id" in t]
            print(f"[Warn] total loss is NaN/Inf on images {ids}. Skip this batch.")
            continue

        # === 反向與更新 ===
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=10.0
        )
        optimizer.step()

        total_loss += loss.item()
        nb += 1

    return total_loss / max(1, nb)


def quick_sanity_check(model, device):
    """用 batch_size=1 快速測試 forward/backward 是否正常。"""
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

    # === 模型 ===
    model = get_fasterrcnn_r50_fpn(num_classes=2, freeze_backbone=True).to(device)

    # === Sanity Check（小 batch）===
    quick_sanity_check(model, device)

    # === 資料 ===
    train_ds = PigsDataset("data/train/img", "data/train/gt.txt",
                           transforms=get_transforms(train=True, max_side=800))
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True,
                              num_workers=4, collate_fn=collate_fn)

    # === 優化器與學習率排程 ===
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    os.makedirs("experiments/logs", exist_ok=True)

    # === 正式訓練 ===
    for epoch in range(6):
        loss = train_one_epoch(model, train_loader, optimizer, device, print_every=50)
        print(f"[Epoch {epoch:02d}] loss={loss:.4f}")
        scheduler.step()
        torch.save(model.state_dict(), f"experiments/logs/fasterrcnn_r50fpn_e{epoch}.pth")


if __name__ == "__main__":
    main()
