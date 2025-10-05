#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =========================
# 直接在這裡指定要用的 YAML
# =========================
CONFIG_PATH = "experiments/configs/v4.yaml"   # ← 改這行即可切換設定

import os
import math
import time
import random
from pathlib import Path
from typing import Dict, Any

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 專案內部（依你現有檔案）
from dataset import PigsDataset, collate_fn
from transforms import get_transforms

# 使用 torchvision 內建的 Faster R-CNN（不改 anchors，避免維度不符）
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


# -------------------------
# 工具
# -------------------------
def load_cfg(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EarlyStopping:
    def __init__(self, mode: str = "min", patience: int = 5):
        assert mode in ("min", "max")
        self.mode = mode
        self.patience = patience
        self.best = None
        self.wait = 0
        self.should_stop = False

    def step(self, value: float) -> bool:
        if self.best is None:
            self.best = value
            self.wait = 0
            return True
        improved = (value < self.best) if self.mode == "min" else (value > self.best)
        if improved:
            self.best = value
            self.wait = 0
            return True
        self.wait += 1
        if self.wait >= self.patience:
            self.should_stop = True
        return False


# -------------------------
# 建立模型（用預設 anchors，避免維度錯誤）
# -------------------------
def build_model(cfg: Dict[str, Any]) -> nn.Module:
    num_classes = int(cfg.get("model", {}).get("num_classes", 1)) + 1  # 含背景
    backbone = str(cfg.get("model", {}).get("backbone", "resnet50")).lower()

    if backbone == "resnet50":
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights=None,              # 作業常規不允許外部預訓練
            weights_backbone=None      # 不載入 backbone 預訓練
        )
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    # 替換分類頭（確保類別數）
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # （不覆寫 model.rpn.anchor_generator、不改 rpn 參數 → 避免 decode 堆疊維度不符）
    return model


# -------------------------
# 訓練 / 驗證（簡潔版）
# -------------------------
def train_one_epoch(model, loader, optimizer, device, epoch_idx, total_epochs) -> float:
    model.train()
    total_loss, n = 0.0, 0
    optimizer.zero_grad(set_to_none=True)

    for images, targets in loader:
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item()
        n += 1

    avg = total_loss / max(1, n)
    print(f"[Train] epoch {epoch_idx+1}/{total_epochs} loss={avg:.4f}")
    return avg


@torch.no_grad()
def validate_one_epoch(model, loader, device) -> float:
    """
    torchvision 的 Faster R-CNN 在 eval() 下 forward(images) 回預測、沒有 loss。
    這裡用 train() + no_grad() 的技巧計 loss（不會更新參數/統計）。
    """
    model.train()
    total, n = 0.0, 0
    for images, targets in loader:
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        total += sum(loss_dict.values()).item()
        n += 1
    return total / max(1, n)


# -------------------------
# Main
# -------------------------
def main():
    cfg = load_cfg(CONFIG_PATH)

    # 基本設定
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # I/O
    ckpt_dir = Path(cfg.get("checkpoint", {}).get("dir", "experiments/logs"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    last_name = cfg.get("checkpoint", {}).get("name", "last.pth")
    best_name = cfg.get("early_stop", {}).get("best_name", "best.pth")
    last_path = ckpt_dir / last_name
    best_path = ckpt_dir / best_name

    # Data（沿用你的 dataset/transforms）
    aug_max_side = int(cfg.get("augment", {}).get("max_side", 1024))
    train_ds = PigsDataset(
        cfg["data"]["train_img_dir"],
        cfg["data"]["train_gt"],
        transforms=get_transforms(train=True, max_side=aug_max_side),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.get("train", {}).get("batch_size", 8)),
        shuffle=True,
        num_workers=int(cfg.get("data", {}).get("num_workers", 4)),
        pin_memory=bool(cfg.get("dataloader", {}).get("pin_memory", True)),
        collate_fn=collate_fn,
    )

    val_loader = None
    val_img_dir = cfg.get("data", {}).get("val_img_dir", "")
    val_gt = cfg.get("data", {}).get("val_gt", "")
    if val_img_dir and val_gt and os.path.isdir(val_img_dir) and os.path.exists(val_gt):
        val_ds = PigsDataset(
            val_img_dir,
            val_gt,
            transforms=get_transforms(train=False, max_side=aug_max_side),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=int(cfg.get("train", {}).get("batch_size", 8)),
            shuffle=False,
            num_workers=int(cfg.get("data", {}).get("num_workers", 4)),
            pin_memory=bool(cfg.get("dataloader", {}).get("pin_memory", True)),
            collate_fn=collate_fn,
        )
        print(f"[Data] val set enabled: {len(val_ds)} samples")
    else:
        print("[Data] no validation set configured; will monitor train loss for early stopping")

    # Model / Optimizer（極簡）
    model = build_model(cfg).to(device)

    opt_name = str(cfg.get("optimizer", {}).get("type", "sgd")).lower()
    lr = float(cfg.get("optimizer", {}).get("lr", 0.0025))
    weight_decay = float(cfg.get("optimizer", {}).get("weight_decay", 1e-4))
    if opt_name == "adamw":
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)
    else:
        momentum = float(cfg.get("optimizer", {}).get("momentum", 0.9))
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=lr, momentum=momentum, weight_decay=weight_decay)

    epochs = int(cfg.get("train", {}).get("epochs", 30))
    print(f"[Train] epochs={epochs}, batch_size={cfg.get('train',{}).get('batch_size',8)}")

    # Early Stopping（預設開啟，patience=5）
    es_cfg = cfg.get("early_stop", {})
    use_es = bool(es_cfg.get("enabled", True))
    patience = int(es_cfg.get("patience", 5))
    es = EarlyStopping(mode=str(es_cfg.get("mode", "min")), patience=patience) if use_es else None

    # Loop
    t0 = time.perf_counter()
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch, epochs)

        if val_loader is not None:
            val_loss = validate_one_epoch(model, val_loader, device)
            print(f"[Val]  epoch {epoch+1}/{epochs} val_loss={val_loss:.4f}")
            monitored = val_loss
        else:
            monitored = train_loss

        # 早停 + 存最佳
        if use_es:
            improved = es.step(monitored)
            if improved:
                torch.save(model.state_dict(), best_path.as_posix())
                print(f"[EarlyStop] New best ({monitored:.4f}). Saved -> {best_path}")
            if es.should_stop:
                print(f"[EarlyStop] No improvement in {patience} epochs. Stop at epoch {epoch+1}.")
                break

    print(f"[Done] total_time={(time.perf_counter()-t0)/60:.2f} min")

    # 存最後
    torch.save(model.state_dict(), last_path.as_posix())
    print(f"[Save Last] {last_path}")
    if use_es and best_path.exists():
        print(f"[Best Model] {best_path}")


if __name__ == "__main__":
    main()
