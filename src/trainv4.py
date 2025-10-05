#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =========================
# 指定要用的 YAML 設定檔
# =========================
CONFIG_PATH = "experiments/configs/v4.yaml"

import os
import time
import math
import random
from pathlib import Path
from typing import Dict, Any

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from contextlib import nullcontext

# 專案內部
from dataset import PigsDataset, collate_fn
from transforms import get_transforms
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# -------------------------------------------------
# 基本工具
# -------------------------------------------------
def load_cfg(path: str) -> Dict[str, Any]:
    """讀取 YAML 並嘗試展開 ${...} 變數（使用 config._expand_vars，若無則忽略）"""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 嘗試展開 ${...}
    try:
        from config import _expand_vars  # 你的專案已有此工具
        cfg = _expand_vars(cfg)
    except Exception:
        pass

    return cfg


def set_seed(seed: int = 42, deterministic: bool = False, benchmark: bool = True):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark


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


# -------------------------------------------------
# 模型
# -------------------------------------------------
def build_model(cfg: Dict[str, Any]) -> nn.Module:
    mcfg = cfg.get("model", {})
    num_classes = int(mcfg.get("num_classes", 1)) + 1  # + 背景
    backbone = str(mcfg.get("backbone", "resnet50")).lower()

    if backbone != "resnet50":
        raise ValueError(f"Unsupported backbone: {backbone} (use resnet50)")

    # 使用 torchvision 預設 anchors/FPN，避免維度不匹配
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=None
    )

    # 替換 ROI head 以對齊類別數
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # 可選：RPN 參數（若 YAML 無 rpn 區塊，安靜跳過）
    rpn_cfg = cfg.get("rpn", None)
    if isinstance(rpn_cfg, dict):
        pre_nms_train = int(rpn_cfg.get("pre_nms_topk_train", model.rpn.pre_nms_top_n["training"]))
        post_nms_train = int(rpn_cfg.get("post_nms_topk_train", model.rpn.post_nms_top_n["training"]))
        pre_nms_test  = int(rpn_cfg.get("pre_nms_topk_test",  model.rpn.pre_nms_top_n["testing"]))
        post_nms_test = int(rpn_cfg.get("post_nms_topk_test", model.rpn.post_nms_top_n["testing"]))
        nms_thr       = float(rpn_cfg.get("nms_thr", model.rpn.nms_thresh))
        model.rpn.pre_nms_top_n  = {"training": pre_nms_train, "testing": pre_nms_test}
        model.rpn.post_nms_top_n = {"training": post_nms_train, "testing": post_nms_test}
        model.rpn.nms_thresh = nms_thr

    return model


# -------------------------------------------------
# Scheduler: Warmup + Cosine（iteration 級）
# -------------------------------------------------
class WarmupCosineLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_iters, total_iters, eta_min=0.0, last_epoch=-1):
        self.warmup_iters = max(1, int(warmup_iters))
        self.total_iters = max(self.warmup_iters + 1, int(total_iters))
        self.eta_min = float(eta_min)
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


# -------------------------------------------------
# 訓練 / 驗證
# -------------------------------------------------
def move_to_device(batch_imgs, batch_tgts, device):
    imgs = [img.to(device, non_blocking=True) for img in batch_imgs]
    tgts = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in batch_tgts]
    return imgs, tgts


def train_one_epoch(model, loader, optimizer, device, epoch_idx, total_epochs,
                    amp=False, scaler=None, grad_clip=None, accumulate=1) -> float:
    model.train()
    total_loss, n = 0.0, 0
    optimizer.zero_grad(set_to_none=True)
    autocast_ctx = torch.amp.autocast("cuda") if amp else nullcontext()

    for it, (images, targets) in enumerate(loader):
        images, targets = move_to_device(images, targets, device)

        with autocast_ctx:
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

        if amp and scaler is not None:
            scaler.scale(loss).backward()
            if (it + 1) % accumulate == 0:
                if grad_clip and grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        else:
            loss.backward()
            if (it + 1) % accumulate == 0:
                if grad_clip and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        total_loss += float(loss.detach().item())
        n += 1

        # 釋放參考，避免 python 容器留住大量 tensor
        del loss_dict, loss, images, targets
        if (it + 1) % 100 == 0 and device.type == "cuda":
            torch.cuda.empty_cache()

    avg = total_loss / max(1, n)
    print(f"[Train] epoch {epoch_idx+1}/{total_epochs} loss={avg:.4f}")
    return avg


@torch.no_grad()
def validate_one_epoch(model, loader, device, amp=False) -> float:
    # Faster R-CNN 在 eval() 下不回傳 loss；用 train() + no_grad() + autocast 來計算
    model.train()
    total, n = 0.0, 0
    autocast_ctx = torch.amp.autocast("cuda") if amp else nullcontext()

    for it, (images, targets) in enumerate(loader):
        images, targets = move_to_device(images, targets, device)
        with autocast_ctx:
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values()).float()
        total += float(loss.item())
        n += 1
        del loss_dict, loss, images, targets
        if (it + 1) % 100 == 0 and device.type == "cuda":
            torch.cuda.empty_cache()

    return total / max(1, n)


# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    cfg = load_cfg(CONFIG_PATH)

    # seed：頂層優先，否則退回 project.seed
    seed = int(cfg.get("seed", cfg.get("project", {}).get("seed", 42)))
    set_seed(seed, deterministic=False, benchmark=bool(cfg.get("train", {}).get("cudnn_benchmark", True)))

    # 裝置
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("device", {}).get("cuda", True) else "cpu")
    print(f"[Device] {device}")

    # I/O
    out_dir = Path(cfg.get("checkpoint", {}).get("dir", "experiments/logs"))
    out_dir.mkdir(parents=True, exist_ok=True)
    last_name = cfg.get("checkpoint", {}).get("name", "last.pth")
    best_name = cfg.get("early_stop", {}).get("best_name", "best.pth")
    last_path = out_dir / last_name
    best_path = out_dir / best_name

    # Data
    aug_max_side = int(cfg.get("augment", {}).get("max_side", 1024))
    train_ds = PigsDataset(
        cfg["data"]["train_img_dir"],
        cfg["data"]["train_gt"],
        transforms=get_transforms(train=True, max_side=aug_max_side),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.get("train", {}).get("batch_size", 8)),
        shuffle=bool(cfg.get("dataloader", {}).get("shuffle", True)),
        num_workers=int(cfg.get("data", {}).get("num_workers", 4)),
        pin_memory=bool(cfg.get("dataloader", {}).get("pin_memory", True)),
        persistent_workers=bool(cfg.get("dataloader", {}).get("persistent_workers", False)),
        prefetch_factor=int(cfg.get("dataloader", {}).get("prefetch_factor", 2)),
        collate_fn=collate_fn,
        drop_last=False,
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
            persistent_workers=bool(cfg.get("dataloader", {}).get("persistent_workers", False)),
            prefetch_factor=int(cfg.get("dataloader", {}).get("prefetch_factor", 2)),
            collate_fn=collate_fn,
            drop_last=False,
        )
        print(f"[Data] val set enabled: {len(val_ds)} samples")
    else:
        print("[Data] no validation set configured; will monitor train loss for early stopping")

    # Model / Optimizer
    model = build_model(cfg).to(device)
    opt_cfg = cfg.get("optimizer", {})
    opt_type = str(opt_cfg.get("type", "sgd")).lower()
    lr = float(opt_cfg.get("lr", 0.003))
    wd = float(opt_cfg.get("weight_decay", 1e-4))
    if opt_type == "adamw":
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=wd)
    else:
        momentum = float(opt_cfg.get("momentum", 0.9))
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=lr, momentum=momentum, weight_decay=wd)

    # AMP
    use_amp = bool(cfg.get("train", {}).get("amp", True))
    scaler = torch.amp.GradScaler("cuda") if (use_amp and device.type == "cuda") else None

    # Scheduler（iteration 級）
    epochs = int(cfg.get("train", {}).get("epochs", 30))
    iters_per_epoch = max(1, len(train_loader))
    total_iters = iters_per_epoch * epochs
    warmup_iters = min(1000, iters_per_epoch * 2)
    eta_min = lr * 0.01
    scheduler = WarmupCosineLR(optimizer, warmup_iters=warmup_iters, total_iters=total_iters, eta_min=eta_min)

    # Early Stopping
    es_cfg = cfg.get("early_stop", {})
    use_es = bool(es_cfg.get("enabled", True))
    patience = int(es_cfg.get("patience", 5))
    es = EarlyStopping(mode=str(es_cfg.get("mode", "min")), patience=patience) if use_es else None

    # 其他訓練參數
    grad_clip = float(cfg.get("train", {}).get("grad_clip", 0.0)) or None
    accumulate = int(cfg.get("train", {}).get("accumulate", 1))

    # Loop
    print(f"[Train] epochs={epochs}, batch_size={cfg.get('train',{}).get('batch_size',8)}, amp={use_amp}")
    t0 = time.perf_counter()
    seen = 0

    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch_idx=epoch,
            total_epochs=epochs,
            amp=use_amp,
            scaler=scaler,
            grad_clip=grad_clip,
            accumulate=accumulate
        )

        # scheduler 以 iteration 級推進
        for _ in range(iters_per_epoch):
            scheduler.step()
            seen += 1

        if val_loader is not None:
            val_loss = validate_one_epoch(model, val_loader, device, amp=use_amp)
            print(f"[Val]  epoch {epoch+1}/{epochs} val_loss={val_loss:.4f}")
            monitored = val_loss
        else:
            monitored = train_loss

        if use_es:
            improved = es.step(monitored)
            if improved and bool(es_cfg.get("save_best", True)):
                torch.save(model.state_dict(), best_path.as_posix())
                print(f"[EarlyStop] New best ({monitored:.4f}). Saved -> {best_path}")
            if es.should_stop:
                print(f"[EarlyStop] No improvement in {patience} epochs. Stop at epoch {epoch+1}.")
                break

        # 每 epoch 清一次 cache，降低碎片
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"[Done] total_time={(time.perf_counter()-t0)/60:.2f} min")

    # Save last
    torch.save(model.state_dict(), last_path.as_posix())
    print(f"[Save Last] {last_path}")
    if use_es and best_path.exists():
        print(f"[Best Model] {best_path}")


if __name__ == "__main__":
    # 建議（可選）：export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    main()
