#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/trainv4.py
Faster R-CNN + ResNet50-FPN 訓練腳本（v4）
- 參考 src/train.py 的穩定架構：tqdm、sanity check、StepLR、grad clip、專案 model 介面
- 加上：AMP、Early Stopping（監控 val_loss 或 train_loss）、可選 Cosine+Warmup（iteration 級）
- 不覆寫 anchors / RPN 內部欄位（避免版本相依問題）
"""

# ========= 可在這裡快速調整的區域 =========
CFG_PATH = "experiments/configs/v4.yaml"  # 指定要讀的 YAML
OVERRIDES = [
    # 例："train.epochs=40", "optimizer.lr=0.0025"
]
SANITY_CHECK_FORCE = None  # 可填 True / False / None（None = 依 YAML）
# ======================================

import os
import time
import math
from pathlib import Path
from contextlib import nullcontext

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import PigsDataset, collate_fn
from transforms import get_transforms
from model import get_fasterrcnn_r50_fpn
from config import load_cfg  # 你專案內的 loader（支援 ${...} 與 overrides）


# ---------------- 小工具 ----------------
class SmoothedValue:
    def __init__(self, alpha=0.9):
        self.alpha = alpha
        self.avg = None
    def update(self, v: float):
        self.avg = v if self.avg is None else self.alpha * self.avg + (1 - self.alpha) * v
    def value(self):
        return float("nan") if self.avg is None else float(self.avg)


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


class WarmupCosineLR(torch.optim.lr_scheduler._LRScheduler):
    """以 iteration 為單位的 Warmup + Cosine"""
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


# ---------------- Sanity Check ----------------
def quick_sanity_check(model, device, cfg):
    """ batch_size=1 快測 forward/backward 是否能跑通（避免正式訓練才發現卡住） """
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
    images  = [images[0].to(device, non_blocking=True)]
    targets = [{k: v.to(device, non_blocking=True) for k, v in targets[0].items()}]
    loss = sum(model(images, targets).values())
    optim.zero_grad(set_to_none=True)
    loss.backward()
    optim.step()
    print(f"[Sanity] one-step OK, loss={loss.item():.4f}")


# ---------------- 訓練 / 驗證（支援 AMP） ----------------
def train_one_epoch(model, loader, optimizer, device, epoch_idx, total_epochs,
                    grad_clip=10.0, amp=False, scaler=None,
                    scheduler_iter=None) -> float:
    model.train()
    total_loss, num_batches = 0.0, 0

    s_total  = SmoothedValue(0.9)
    s_rpnobj = SmoothedValue(0.9)
    s_rpnreg = SmoothedValue(0.9)
    s_cls    = SmoothedValue(0.9)
    s_box    = SmoothedValue(0.9)

    autocast_ctx = torch.amp.autocast("cuda") if amp else nullcontext()
    pbar = tqdm(loader, total=len(loader), ncols=120, desc=f"Epoch {epoch_idx+1}/{total_epochs}")

    for step, (images, targets) in enumerate(pbar, start=1):
        images  = [img.to(device, non_blocking=True) for img in images]
        targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]

        with autocast_ctx:
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

        optimizer.zero_grad(set_to_none=True)

        if amp and scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], grad_clip)
            optimizer.step()

        # Step per-iteration scheduler（若有）
        if scheduler_iter is not None:
            scheduler_iter.step()

        l_total = float(loss.item())
        total_loss += l_total
        num_batches += 1

        # 分項容錯
        l_obj = float(loss_dict.get("loss_objectness",   torch.tensor(0.0)).item())
        l_rpn = float(loss_dict.get("loss_rpn_box_reg",  torch.tensor(0.0)).item())
        l_cls = float(loss_dict.get("loss_classifier",   torch.tensor(0.0)).item())
        l_box = float(loss_dict.get("loss_box_reg",      torch.tensor(0.0)).item())

        s_total.update(l_total); s_rpnobj.update(l_obj); s_rpnreg.update(l_rpn); s_cls.update(l_cls); s_box.update(l_box)
        pbar.set_postfix({
            "loss":    f"{s_total.value():.3f}",
            "rpn_obj": f"{s_rpnobj.value():.3f}",
            "rpn_reg": f"{s_rpnreg.value():.3f}",
            "cls":     f"{s_cls.value():.3f}",
            "box":     f"{s_box.value():.3f}",
            "lr":      f"{optimizer.param_groups[0]['lr']:.3e}",
        })

        # 釋放暫存參考，避免 Python 容器 hold 住 tensor
        del loss_dict, loss, images, targets

        # 避免長時間第一輪「卡住」的錯覺（其實在做大 batch 的第一個 forward/backward）
        # 有 pbar + 平滑 loss 可以觀察是否在緩慢前進

    avg_loss = total_loss / max(1, num_batches)
    return avg_loss


@torch.no_grad()
def validate_one_epoch(model, loader, device, amp=False) -> float:
    model.train()  # eval() 下不回 loss；採 train()+no_grad() 技巧
    total, n = 0.0, 0
    autocast_ctx = torch.amp.autocast("cuda") if amp else nullcontext()
    for images, targets in loader:
        images  = [img.to(device, non_blocking=True) for img in images]
        targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
        with autocast_ctx:
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values()).float()
        total += float(loss.item()); n += 1
        del loss_dict, loss, images, targets
    return total / max(1, n)


# ---------------- Main ----------------
def main():
    # 用專案的 load_cfg（支援 ${...} 與 overrides）
    project_root = Path(__file__).resolve().parents[1]
    cfg_file = project_root / CFG_PATH
    cfg = load_cfg(str(cfg_file), overrides=OVERRIDES)

    # 可強制開/關 sanity check
    if SANITY_CHECK_FORCE is not None:
        cfg["sanity_check"]["enabled"] = bool(SANITY_CHECK_FORCE)

    device = torch.device("cuda" if torch.cuda.is_available() and cfg["device"]["cuda"] else "cpu")
    torch.backends.cudnn.benchmark = cfg["train"]["cudnn_benchmark"]
    print(f"[Device] {device}")

    # === 建模（用專案的 model.get_fasterrcnn_r50_fpn，與 train.py 一致） ===
    model = get_fasterrcnn_r50_fpn(
        num_classes=cfg["model"]["num_classes"],
        freeze_backbone=cfg["model"]["freeze_backbone"]
    ).to(device)

    # === Sanity Check ===
    if cfg["sanity_check"]["enabled"]:
        quick_sanity_check(model, device, cfg)

    # === Data ===
    train_ds = PigsDataset(
        cfg["data"]["train_img_dir"],
        cfg["data"]["train_gt"],
        transforms=get_transforms(train=True, max_side=cfg["augment"]["max_side"])
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=cfg["dataloader"]["shuffle"],
        num_workers=cfg["data"]["num_workers"],
        pin_memory=cfg["dataloader"]["pin_memory"],
        persistent_workers=cfg["dataloader"]["persistent_workers"],
        prefetch_factor=cfg["dataloader"]["prefetch_factor"],
        collate_fn=collate_fn,
        drop_last=False,
    )

    val_loader = None
    if os.path.isdir(cfg["data"]["val_img_dir"]) and os.path.exists(cfg["data"]["val_gt"]):
        val_ds = PigsDataset(
            cfg["data"]["val_img_dir"],
            cfg["data"]["val_gt"],
            transforms=get_transforms(train=False, max_side=cfg["augment"]["max_side"])
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg["train"]["batch_size"],
            shuffle=False,
            num_workers=cfg["data"]["num_workers"],
            pin_memory=cfg["dataloader"]["pin_memory"],
            persistent_workers=cfg["dataloader"]["persistent_workers"],
            prefetch_factor=cfg["dataloader"]["prefetch_factor"],
            collate_fn=collate_fn,
            drop_last=False,
        )
        print(f"[Data] val set enabled: {len(val_ds)} samples")
    else:
        print("[Data] no validation set configured; will monitor train loss for early stopping")

    # === Optimizer ===
    params = [p for p in model.parameters() if p.requires_grad]
    if cfg["optimizer"]["type"].lower() == "adamw":
        optimizer = torch.optim.AdamW(params, lr=cfg["optimizer"]["lr"], weight_decay=cfg["optimizer"]["weight_decay"])
    else:
        optimizer = torch.optim.SGD(
            params,
            lr=cfg["optimizer"]["lr"],
            momentum=cfg["optimizer"]["momentum"],
            weight_decay=cfg["optimizer"]["weight_decay"]
        )

    # === Scheduler：支援 StepLR 或 Cosine+Warmup ===
    scheduler_epoch = None
    scheduler_iter = None
    if cfg["scheduler"]["type"].lower() == "cosine":
        iters_per_epoch = max(1, len(train_loader))
        total_iters = iters_per_epoch * cfg["train"]["epochs"]
        warmup_iters = min(1000, 2 * iters_per_epoch)
        eta_min = cfg["optimizer"]["lr"] * 0.01
        scheduler_iter = WarmupCosineLR(optimizer, warmup_iters, total_iters, eta_min=eta_min)
    else:
        scheduler_epoch = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg["scheduler"]["step_size"],
            gamma=cfg["scheduler"]["gamma"]
        )

    # === AMP / EarlyStop ===
    use_amp = bool(cfg["train"]["amp"])
    scaler = torch.amp.GradScaler("cuda") if (use_amp and device.type == "cuda") else None

    es_cfg = cfg["early_stop"]
    use_es = bool(es_cfg["enabled"])
    es = EarlyStopping(mode=str(es_cfg["mode"]), patience=int(es_cfg["patience"])) if use_es else None

    # === I/O ===
    os.makedirs(cfg["checkpoint"]["dir"], exist_ok=True)
    best_path = os.path.join(cfg["checkpoint"]["dir"], es_cfg.get("best_name", "best.pth"))
    last_path = cfg["checkpoint"].get("save_full_path") or os.path.join(cfg["checkpoint"]["dir"], cfg["checkpoint"]["name"])

    # === Loop ===
    print(f"[Train] epochs={cfg['train']['epochs']}, batch_size={cfg['train']['batch_size']}, amp={use_amp}")
    total_t0 = time.perf_counter()

    for epoch in range(cfg["train"]["epochs"]):
        avg_loss = train_one_epoch(
            model, train_loader, optimizer, device,
            epoch_idx=epoch, total_epochs=cfg["train"]["epochs"],
            grad_clip=cfg["train"]["grad_clip"], amp=use_amp, scaler=scaler,
            scheduler_iter=scheduler_iter
        )

        # epoch 級 scheduler
        if scheduler_epoch is not None:
            scheduler_epoch.step()

        # 驗證 / 早停
        if val_loader is not None:
            val_loss = validate_one_epoch(model, val_loader, device, amp=use_amp)
            print(f"[Val ] epoch {epoch+1}/{cfg['train']['epochs']} val_loss={val_loss:.4f}")
            monitored = val_loss
        else:
            monitored = avg_loss

        if use_es:
            improved = es.step(monitored)
            if improved and bool(es_cfg.get("save_best", True)):
                torch.save(model.state_dict(), best_path)
                print(f"[EarlyStop] New best ({monitored:.4f}). Saved -> {best_path}")
            if es.should_stop:
                print(f"[EarlyStop] No improvement for {es_cfg['patience']} epochs. Stop at epoch {epoch+1}.")
                break

    total_time = time.perf_counter() - total_t0
    print(f"[Done] total_time={total_time/60:.2f} min")

    # 保存最後
    state = model.state_dict()
    if cfg["checkpoint"].get("save_fp16", False):
        state = {k: v.half() for k, v in state.items()}
    torch.save(state, last_path)
    print(f"[Save Last] {last_path}")
    if use_es and os.path.exists(best_path):
        print(f"[Best Model] {best_path}")


if __name__ == "__main__":
    main()
