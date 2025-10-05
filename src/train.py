#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =========================
# 直接在這裡指定要用的 YAML
# =========================
CONFIG_PATH = "experiments/configs/v4.yaml"   # ← 改這行就能切換，如 "configs/v4.yaml"
OVERRIDES = [
    # 例： "train.epochs=72", "optimizer.lr=0.0025", "early_stop.enabled=true"
]

import os
import time
import math
import random
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from contextlib import nullcontext

# 專案內部（維持你的檔名與介面）
from config import load_cfg                 # 使用你專案內的 loader
from dataset import PigsDataset, collate_fn # dataset 介面沿用
from transforms import get_transforms       # transforms 介面沿用

# 模型（以 torchvision Faster R-CNN 為主，並支援自訂 anchors）
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -------------------------
# Early Stopping
# -------------------------
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
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.should_stop = True
            return False


# -------------------------
# 建立模型（Faster R-CNN + 自訂 Anchors/Head）
# -------------------------
def build_model(cfg: Dict[str, Any]) -> nn.Module:
    num_classes = int(cfg["model"].get("num_classes", 1)) + 1  # 含背景
    backbone_name = cfg["model"].get("backbone", "resnet50")
    pretrained_backbone = bool(cfg["model"].get("pretrained_backbone", False))

    # Anchors
    anchor_sizes = cfg["model"].get("anchor_sizes", [32, 64, 96, 128, 192])
    anchor_aspect_ratios = cfg["model"].get("anchor_aspect_ratios", [1.0, 1.5, 2.0, 2.5])
    sizes_per_level = tuple([(s,) for s in anchor_sizes])
    aspect_per_level = tuple([tuple(anchor_aspect_ratios)] * len(anchor_sizes))
    anchor_gen = AnchorGenerator(sizes_per_level, aspect_per_level)

    if backbone_name.lower() == "resnet50":
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights=None,  # 作業規範常不允許外部預訓練
            weights_backbone="IMAGENET1K_V1" if pretrained_backbone else None,
            trainable_backbone_layers=5,
        )
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")

    # 替換 RPN Anchors
    model.rpn.anchor_generator = anchor_gen

    # 替換分類 head（確保 num_classes 正確）
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # RPN proposals / NMS
    rpn_pre_nms_train = int(cfg["rpn"].get("pre_nms_topk_train", 2000))
    rpn_post_nms_train = int(cfg["rpn"].get("post_nms_topk_train", 1000))
    rpn_pre_nms_test  = int(cfg["rpn"].get("pre_nms_topk_test", 4000))
    rpn_post_nms_test = int(cfg["rpn"].get("post_nms_topk_test", 2000))
    rpn_nms_thr       = float(cfg["rpn"].get("nms_thr", 0.7))
    model.rpn.pre_nms_top_n = dict(training=rpn_pre_nms_train, testing=rpn_pre_nms_test)
    model.rpn.post_nms_top_n = dict(training=rpn_post_nms_train, testing=rpn_post_nms_test)
    model.rpn.nms_thresh = rpn_nms_thr

    return model


# -------------------------
# Cosine + Warmup LR
# -------------------------
class WarmupCosineLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_iters, total_iters, eta_min=0.0, last_epoch=-1):
        self.warmup_iters = max(1, warmup_iters)
        self.total_iters = max(self.warmup_iters + 1, total_iters)
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1
        lrs = []
        for base_lr in self.base_lrs:
            if step <= self.warmup_iters:
                lr = base_lr * step / self.warmup_iters
            else:
                t = (step - self.warmup_iters) / (self.total_iters - self.warmup_iters)
                lr = self.eta_min + 0.5 * (base_lr - self.eta_min) * (1 + math.cos(math.pi * t))
            lrs.append(lr)
        return lrs


# -------------------------
# 訓練 / 驗證
# -------------------------
def train_one_epoch(model, loader, optimizer, device, scaler, grad_clip=None, accumulate=1, epoch_idx=0, total_epochs=0) -> float:
    model.train()
    total_loss, n = 0.0, 0
    optimizer.zero_grad(set_to_none=True)

    for it, (images, targets) in enumerate(loader):
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        autocast_ctx = torch.cuda.amp.autocast if scaler is not None else nullcontext
        with autocast_ctx():
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

        if scaler is not None:
            scaler.scale(loss).backward()
            if (it + 1) % accumulate == 0:
                if grad_clip is not None and grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        else:
            loss.backward()
            if (it + 1) % accumulate == 0:
                if grad_clip is not None and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
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
    Faster R-CNN 在 eval() 下回傳的是預測，不含 loss。
    用 train() + no_grad() 取 loss（不更新參數/統計），是實務常見作法。
    """
    model.train()
    total, n = 0.0, 0
    for images, targets in loader:
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        loss = sum(loss_dict.values()).item()
        total += loss
        n += 1
    return total / max(1, n)


# -------------------------
# Main
# -------------------------
def main():
    # 讀設定：用檔頭的 CONFIG_PATH 與可選 OVERRIDES（沿用 config.py 的語法）
    cfg = load_cfg(CONFIG_PATH, overrides=OVERRIDES)

    # 輸出與隨機種子
    ckpt_dir = Path(cfg["checkpoint"]["dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    set_seed(int(cfg.get("seed", 42)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # Data
    aug_max_side = int(cfg["augment"].get("max_side", 1024))
    train_ds = PigsDataset(
        cfg["data"]["train_img_dir"],
        cfg["data"]["train_gt"],
        transforms=get_transforms(train=True, max_side=aug_max_side),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["data"]["num_workers"]),
        pin_memory=bool(cfg["dataloader"].get("pin_memory", True)),
        persistent_workers=bool(cfg["dataloader"].get("persistent_workers", False)),
        prefetch_factor=int(cfg["dataloader"].get("prefetch_factor", 2)),
        collate_fn=collate_fn,
        drop_last=False,
    )

    # Optional val
    val_loader = None
    val_img_dir = cfg["data"].get("val_img_dir", "")
    val_gt = cfg["data"].get("val_gt", "")
    if val_img_dir and val_gt and os.path.isdir(val_img_dir) and os.path.exists(val_gt):
        val_ds = PigsDataset(
            val_img_dir,
            val_gt,
            transforms=get_transforms(train=False, max_side=aug_max_side),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=int(cfg["train"]["batch_size"]),
            shuffle=False,
            num_workers=int(cfg["data"]["num_workers"]),
            pin_memory=bool(cfg["dataloader"].get("pin_memory", True)),
            persistent_workers=bool(cfg["dataloader"].get("persistent_workers", False)),
            prefetch_factor=int(cfg["dataloader"].get("prefetch_factor", 2)),
            collate_fn=collate_fn,
            drop_last=False,
        )
        print(f"[Data] val set enabled: {len(val_ds)} samples")
    else:
        print("[Data] no validation set configured; will monitor train loss for early stopping")

    # Model
    model = build_model(cfg).to(device)

    # Optimizer / AMP / Scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    base_lr = float(cfg["optimizer"]["lr"])
    opt_name = cfg["optimizer"].get("type", "sgd").lower()
    if opt_name == "sgd":
        optimizer = torch.optim.SGD(
            params,
            lr=base_lr,
            momentum=float(cfg["optimizer"].get("momentum", 0.9)),
            weight_decay=float(cfg["optimizer"].get("weight_decay", 1e-4)),
        )
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(
            params,
            lr=base_lr,
            weight_decay=float(cfg["optimizer"].get("weight_decay", 1e-4)),
        )
    else:
        raise ValueError(f"Unsupported optimizer: {opt_name}")

    epochs = int(cfg["train"]["epochs"])
    accumulate = int(cfg["train"].get("accumulate", 1))
    grad_clip = float(cfg["train"].get("grad_clip", 0.0)) or None

    use_amp = bool(cfg["train"].get("amp", False))
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    iters_per_epoch = max(1, len(train_loader))
    total_iters = epochs * iters_per_epoch
    warmup_iters = int(cfg["scheduler"].get("warmup_iters", min(1000, iters_per_epoch * 2)))
    eta_min = float(cfg["scheduler"].get("eta_min", base_lr * 0.01))
    scheduler = WarmupCosineLR(optimizer, warmup_iters=warmup_iters, total_iters=total_iters, eta_min=eta_min)

    # Early Stopping
    es_cfg = cfg.get("early_stop", {})
    use_es = bool(es_cfg.get("enabled", False))
    es_mode = str(es_cfg.get("mode", "min"))
    patience = int(es_cfg.get("patience", 5))
    es = EarlyStopping(mode=es_mode, patience=patience) if use_es else None

    best_name = es_cfg.get("best_name", "best.pth")
    best_path = ckpt_dir / best_name

    ckpt_name = cfg["checkpoint"].get("name", "last.pth")
    ckpt_path = ckpt_dir / ckpt_name

    # Loop
    print(f"[Train] epochs={epochs}, batch_size={cfg['train']['batch_size']}, accumulate={accumulate}, amp={use_amp}")
    t0 = time.perf_counter()

    for epoch in range(epochs):
        avg_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            grad_clip=grad_clip,
            accumulate=accumulate,
            epoch_idx=epoch,
            total_epochs=epochs,
        )

        # 以 iteration 為單位 step LR（這裡補齊一個 epoch 的步數）
        for _ in range(iters_per_epoch):
            scheduler.step()

        if val_loader is not None:
            val_loss = validate_one_epoch(model, val_loader, device)
            print(f"[Val]  epoch {epoch+1}/{epochs} val_loss={val_loss:.4f}")
            monitored = val_loss
        else:
            monitored = avg_loss

        if use_es:
            improved = es.step(monitored)
            if improved and bool(es_cfg.get("save_best", True)):
                torch.save(model.state_dict(), best_path.as_posix())
                print(f"[EarlyStop] New best ({monitored:.4f}). Saved -> {best_path}")
            if es.should_stop:
                print(f"[EarlyStop] No improvement in {patience} epochs. Stop at epoch {epoch+1}.")
                break

    total_time = time.perf_counter() - t0
    print(f"[Done] total_time={total_time/60.0:.2f} min")

    # 保存最後一個
    state = model.state_dict()
    if bool(cfg["checkpoint"].get("save_fp16", False)):
        state = {k: v.half() for k, v in state.items()}
    torch.save(state, ckpt_path.as_posix())
    print(f"[Save Last] {ckpt_path}")
    if use_es and best_path.exists():
        print(f"[Best Model] {best_path}")


if __name__ == "__main__":
    main()
