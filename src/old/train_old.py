#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/train.py
Faster R-CNN + ResNet50-FPN 訓練腳本（檔頭可調參數版）。
- 在檔頭選擇 YAML、列出覆寫鍵值（不需要命令列參數）
- 保留：tqdm、Sanity Check、StepLR、Gradient Clipping、只存最後權重
- 預設讀取 experiments/configs/default.yaml
"""

# ========= 可在這裡快速調整的區域（只改這塊就好） =========
# 指定要讀的 YAML 設定檔（相對於專案根目錄）
CFG_PATH = "experiments/configs/default.yaml"

# 需要臨時覆寫的設定（採用 dot-notation），例：
#   "train.epochs=40", "optimizer.lr=0.001", "model.freeze_backbone=true"
OVERRIDES = [
    # "train.epochs=2",
    # "optimizer.lr=0.003",
    # "augment.max_side=1024",
    # "checkpoint.name=fasterrcnn_r50fpn_final_v3.pth",
]

# 若你想強制跳過或啟用 sanity check，可在此覆寫（None = 依 YAML）
SANITY_CHECK_FORCE = None  # 可選：True / False / None
# =======================================================


import os
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import PigsDataset, collate_fn
from transforms import get_transforms
from src.modelv4 import get_fasterrcnn_r50_fpn
from config import load_cfg  # 位於 src/config.py


# ---- 小工具：平滑顯示 ----
class SmoothedValue:
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

        # 防呆：略過非有限值
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

        # 分項（容錯 key）
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


def quick_sanity_check(model, device, cfg):
    """ batch_size=1 快測 forward/backward 是否能跑通 """
    ds = PigsDataset(cfg["data"]["train_img_dir"], cfg["data"]["train_gt"],
                     transforms=get_transforms(train=True, max_side=cfg["sanity_check"]["max_side"]))
    loader = DataLoader(
        ds, batch_size=cfg["sanity_check"]["batch_size"], shuffle=True,
        num_workers=cfg["data"]["num_workers"], pin_memory=True,
        persistent_workers=True, prefetch_factor=4, collate_fn=collate_fn
    )
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
    # 以 train.py 的位置推回到專案根目錄，再拼 config 路徑
    project_root = Path(__file__).resolve().parents[1]
    cfg_file = project_root / CFG_PATH

    # 讀設定（支援檔頭 OVERRIDES）
    cfg = load_cfg(str(cfg_file), overrides=OVERRIDES)

    # （可選）強制覆寫 sanity_check 開關
    if SANITY_CHECK_FORCE is not None:
        cfg["sanity_check"]["enabled"] = bool(SANITY_CHECK_FORCE)

    device = torch.device("cuda" if torch.cuda.is_available() and cfg["device"]["cuda"] else "cpu")
    torch.backends.cudnn.benchmark = cfg["train"]["cudnn_benchmark"]

    # === 建模 ===
    model = get_fasterrcnn_r50_fpn(
        num_classes=cfg["model"]["num_classes"],
        freeze_backbone=cfg["model"]["freeze_backbone"]
    ).to(device)

    # === Sanity Check ===
    if cfg["sanity_check"]["enabled"]:
        quick_sanity_check(model, device, cfg)

    # === 資料 ===
    train_ds = PigsDataset(
        cfg["data"]["train_img_dir"],
        cfg["data"]["train_gt"],
        transforms=get_transforms(train=True, max_side=cfg["augment"]["max_side"])
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=cfg["dataloader"]["pin_memory"],
        persistent_workers=cfg["dataloader"]["persistent_workers"],
        prefetch_factor=cfg["dataloader"]["prefetch_factor"],
        collate_fn=collate_fn
    )

    # === 優化器與學習率排程 ===
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=cfg["optimizer"]["lr"],
        momentum=cfg["optimizer"]["momentum"],
        weight_decay=cfg["optimizer"]["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=cfg["scheduler"]["step_size"],
        gamma=cfg["scheduler"]["gamma"]
    )

    # === 訓練 ===
    os.makedirs(cfg["checkpoint"]["dir"], exist_ok=True)
    total_t0 = time.perf_counter()
    for epoch in range(cfg["train"]["epochs"]):
        avg_loss, epoch_time = train_one_epoch(
            model, train_loader, optimizer, device,
            epoch_idx=epoch, total_epochs=cfg["train"]["epochs"],
            grad_clip=cfg["train"]["grad_clip"]
        )
        scheduler.step()
    total_time = time.perf_counter() - total_t0

    # === 保存（只存最後一個） ===
    ckpt_cfg  = cfg["checkpoint"]
    ckpt_path = ckpt_cfg.get("save_full_path") or os.path.join(ckpt_cfg["dir"], ckpt_cfg["name"])
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    state = model.state_dict()
    if ckpt_cfg.get("save_fp16", False):
        # 可選：半精度存檔，檔案大約減半
        state = {k: v.half() for k, v in state.items()}

    torch.save(state, ckpt_path)
    print(f"[Save Final] {ckpt_path}")



if __name__ == "__main__":
    main()
