#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train.py
Faster R-CNN + ResNet50-FPN（backbone 可選凍結）訓練腳本。
- 內建可調參數（檔頭區域）
- tqdm 進度列 + per-batch 訊息（loss 分項、batch time、LR）
- NaN/Inf 防護 + 斜率裁剪（gradient clipping）
- 只保存最後一個權重檔
- 開機前做一次 quick sanity check（batch_size=1）

資料位置假設：
  data/train/img    (影像)
  data/train/gt.txt (標註：frame,x,y,w,h)


"""

import os
import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import PigsDataset, collate_fn
from transforms import get_transforms
from model import get_fasterrcnn_r50_fpn


# ========= 可自行修改的設定 =========
EPOCHS       = 30
BATCH_SIZE   = 4
MAX_SIDE     = 1024
LR           = 0.003
WEIGHT_DECAY = 1e-4
STEP_SIZE    = 5
GAMMA        = 0.1
FREEZE_BB    = False   # 是否凍結 ResNet50 backbone
NUM_WORKERS  = 8       # 使用幾顆 CPU 讀資料
GRAD_CLIP    = 10.0    # 0 或 None 代表不裁剪

CKPT_DIR     = "experiments/logs"
CKPT_NAME    = "fasterrcnn_r50fpn_final_v2.pth"   # 只保存最後一個
# ====================================


class SmoothedValue:
    """簡單的指數平滑，用於 tqdm 顯示"""
    def __init__(self, alpha=0.9):
        self.alpha = alpha
        self.avg = None
    def update(self, v: float):
        self.avg = v if self.avg is None else self.alpha * self.avg + (1 - self.alpha) * v
    def value(self):
        return float("nan") if self.avg is None else float(self.avg)


def train_one_epoch(model, loader, optimizer, device, epoch_idx, total_epochs, grad_clip=10.0):
    model.train()
    total_loss, num_batches = 0.0, 0

    s_total  = SmoothedValue(0.9)
    s_rpnobj = SmoothedValue(0.9)
    s_rpnreg = SmoothedValue(0.9)
    s_cls    = SmoothedValue(0.9)
    s_box    = SmoothedValue(0.9)

    epoch_t0 = time.perf_counter()
    pbar = tqdm(loader, total=len(loader), ncols=120, desc=f"Epoch {epoch_idx+1}/{total_epochs}")

    for step, (images, targets) in enumerate(pbar, start=1):
        bt0 = time.perf_counter()

        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)  # dict: loss_classifier, loss_box_reg, loss_objectness, loss_rpn_box_reg

        # 檢查每個 loss 是否為有限值
        if any(not torch.isfinite(val) for val in loss_dict.values()):
            print("[Warn] Non-finite loss dict, skip this batch.")
            continue

        loss = sum(loss_dict.values())
        if not torch.isfinite(loss):
            print("[Warn] total loss is NaN/Inf, skip this batch.")
            continue

        optimizer.zero_grad()
        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=grad_clip
            )
        optimizer.step()

        l_total = float(loss.item())
        total_loss += l_total
        num_batches += 1

        # 個別分項（有些版本 key 可能不同，用 get 並設 0.0 預設）
        l_obj = float(loss_dict.get("loss_objectness",   torch.tensor(0.0)).item())
        l_rpn = float(loss_dict.get("loss_rpn_box_reg",  torch.tensor(0.0)).item())
        l_cls = float(loss_dict.get("loss_classifier",   torch.tensor(0.0)).item())
        l_box = float(loss_dict.get("loss_box_reg",      torch.tensor(0.0)).item())

        s_total.update(l_total)
        s_rpnobj.update(l_obj)
        s_rpnreg.update(l_rpn)
        s_cls.update(l_cls)
        s_box.update(l_box)

        bt = time.perf_counter() - bt0
        lr = optimizer.param_groups[0]["lr"]
        pbar.set_postfix({
            "loss":    f"{s_total.value():.3f}",
            "rpn_obj": f"{s_rpnobj.value():.3f}",
            "rpn_reg": f"{s_rpnreg.value():.3f}",
            "cls":     f"{s_cls.value():.3f}",
            "box":     f"{s_box.value():.3f}",
            "bt":      f"{bt*1000:.0f}ms",
            "lr":      f"{lr:.3e}",
        })

    epoch_time = time.perf_counter() - epoch_t0
    avg_loss = total_loss / max(1, num_batches)
    print(f"[Epoch {epoch_idx+1}/{total_epochs}] avg_loss={avg_loss:.4f} | time={epoch_time:.1f}s")
    return avg_loss, epoch_time


def quick_sanity_check(model, device):
    """用 batch_size=1 快測 forward/backward 是否能跑通"""
    ds = PigsDataset("data/train/img", "data/train/gt.txt",
                     transforms=get_transforms(train=True, max_side=640))
    loader = DataLoader(ds, batch_size=1, shuffle=True, 
                        num_workers=6, 
                        pin_memory=True, 
                        persistent_workers=True, 
                        prefetch_factor=4,
                        collate_fn=collate_fn)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.SGD(params, lr=0.01, momentum=0.9)
    images, targets = next(iter(loader))
    images  = [images[0].to(device)]
    targets = [{k: v.to(device) for k, v in targets[0].items()}]
    loss = sum(model(images, targets).values())
    optim.zero_grad()
    loss.backward()
    optim.step()
    print(f"[Sanity] one-step OK, loss={loss.item():.4f}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    # === 建模 ===
    model = get_fasterrcnn_r50_fpn(num_classes=2, freeze_backbone=FREEZE_BB).to(device)

    # === Sanity Check（小 batch）===
    quick_sanity_check(model, device)

    # === 資料 ===
    train_ds = PigsDataset("data/train/img", "data/train/gt.txt",
                           transforms=get_transforms(train=True, max_side=MAX_SIDE))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, 
                              persistent_workers=True, prefetch_factor=4, 
                              collate_fn=collate_fn)

    # === 優化器與學習率排程 ===
    params    = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LR, momentum=0.9, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    os.makedirs(CKPT_DIR, exist_ok=True)

    # === 正式訓練 ===
    total_t0 = time.perf_counter()
    for epoch in range(EPOCHS):
        avg_loss, epoch_time = train_one_epoch(
            model, train_loader, optimizer, device,
            epoch_idx=epoch, total_epochs=EPOCHS, grad_clip=GRAD_CLIP
        )
        scheduler.step()
    total_time = time.perf_counter() - total_t0

    # === 只存最後一個權重 ===
    ckpt_path = os.path.join(CKPT_DIR, CKPT_NAME)
    torch.save(model.state_dict(), ckpt_path)
    print(f"[Save Final] {ckpt_path}")
    print(f"[Done] Total training time: {total_time/60:.1f} min")

if __name__ == "__main__":
    main()
