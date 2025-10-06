#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
trainv5.py — Faster R-CNN ResNet50-FPN (from scratch)
版本 v5：對應 modelv5 + YAML 參數化。
"""

import os, time, math, traceback, sys
from pathlib import Path
from contextlib import nullcontext

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import PigsDataset, collate_fn
from transforms import get_transforms
from modelv5 import get_fasterrcnn_r50_fpn
from config import load_cfg


# ---------------- 小工具 ----------------
class SmoothedValue:
    def __init__(self, alpha=0.9):
        self.alpha = alpha
        self.avg = None
    def update(self, v: float):
        self.avg = v if self.avg is None else self.alpha * self.avg + (1 - self.alpha) * v
    def value(self): return float("nan") if self.avg is None else float(self.avg)


class EarlyStopping:
    def __init__(self, mode="min", patience=5):
        assert mode in ("min", "max")
        self.mode, self.patience = mode, patience
        self.best, self.wait, self.should_stop = None, 0, False
    def step(self, value: float):
        if self.best is None:
            self.best = value; return True
        improved = (value < self.best) if self.mode == "min" else (value > self.best)
        if improved:
            self.best, self.wait = value, 0
            return True
        self.wait += 1
        if self.wait >= self.patience:
            self.should_stop = True
        return False


class WarmupCosineLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_iters, total_iters, eta_min=0.0, last_epoch=-1):
        self.warmup_iters = max(1, int(warmup_iters))
        self.total_iters  = max(self.warmup_iters + 1, int(total_iters))
        self.eta_min = float(eta_min)
        super().__init__(optimizer, last_epoch)
    def get_lr(self):
        step = self.last_epoch + 1
        out = []
        for base_lr in self.base_lrs:
            if step <= self.warmup_iters:
                lr = base_lr * step / self.warmup_iters
            else:
                t = (step - self.warmup_iters) / (self.total_iters - self.warmup_iters)
                lr = self.eta_min + 0.5 * (base_lr - self.eta_min) * (1 + math.cos(math.pi * t))
            out.append(lr)
        return out


# ---------------- Sanity Check ----------------
def quick_sanity_check(model, device, cfg):
    ds = PigsDataset(cfg["data"]["train_img_dir"], cfg["data"]["train_gt"],
                     transforms=get_transforms(train=True, max_side=cfg["sanity_check"]["max_side"]))
    loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0, collate_fn=collate_fn)
    model.train()
    images, targets = next(iter(loader))
    images  = [images[0].to(device)]
    targets = [{k: v.to(device) for k, v in targets[0].items()}]
    loss = sum(model(images, targets).values())
    print(f"[Sanity] one-step OK, loss={loss.item():.4f}")


# ---------------- Train / Val ----------------
def train_one_epoch(model, loader, optimizer, device, epoch_idx, total_epochs,
                    grad_clip=10.0, amp=False, scaler=None, scheduler_iter=None):
    model.train()
    total_loss, n = 0.0, 0
    autocast_ctx = torch.amp.autocast("cuda") if amp else nullcontext()

    pbar = tqdm(loader, total=len(loader), ncols=120, desc=f"Epoch {epoch_idx+1}/{total_epochs}")
    for images, targets in pbar:
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        with autocast_ctx:
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
        optimizer.zero_grad(set_to_none=True)
        if amp and scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer); scaler.update()
        else:
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        if scheduler_iter is not None:
            scheduler_iter.step()
        total_loss += float(loss.item()); n += 1
        pbar.set_postfix({"loss": f"{total_loss/n:.3f}", "lr": f"{optimizer.param_groups[0]['lr']:.2e}"})
        del loss_dict, loss, images, targets
    return total_loss / max(1, n)


@torch.no_grad()
def validate_one_epoch(model, loader, device, amp=False):
    model.train()
    total, n = 0.0, 0
    autocast_ctx = torch.amp.autocast("cuda") if amp else nullcontext()
    for images, targets in loader:
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        with autocast_ctx:
            loss = sum(model(images, targets).values())
        total += float(loss.item()); n += 1
    return total / max(1, n)


# ---------------- Main ----------------
def main():
    print("[DBG] Enter main()")
    project_root = Path(__file__).resolve().parents[1]
    cfg_path = project_root / "experiments/configs/v5.yaml"
    print(f"[DBG] CFG_PATH={cfg_path} -> exists={cfg_path.exists()}")
    cfg = load_cfg(str(cfg_path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    model = get_fasterrcnn_r50_fpn(
        num_classes=cfg["model"]["num_classes"],
        freeze_backbone=cfg["model"]["freeze_backbone"]
    ).to(device)
    print("[DBG] model built OK")

    if cfg["sanity_check"]["enabled"]:
        quick_sanity_check(model, device, cfg)

    ds = PigsDataset(cfg["data"]["train_img_dir"], cfg["data"]["train_gt"],
                     transforms=get_transforms(train=True, max_side=cfg["augment"]["max_side"]))
    loader = DataLoader(ds, batch_size=cfg["train"]["batch_size"],
                        shuffle=True, num_workers=cfg["data"]["num_workers"], collate_fn=collate_fn)
    print(f"[Data] train samples={len(ds)}")

    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                                lr=cfg["optimizer"]["lr"], momentum=cfg["optimizer"]["momentum"])
    print(f"[Train] epochs={cfg['train']['epochs']}, batch={cfg['train']['batch_size']}")

    for epoch in range(cfg["train"]["epochs"]):
        avg_loss = train_one_epoch(model, loader, optimizer, device, epoch, cfg["train"]["epochs"])
        print(f"[Epoch {epoch+1}] loss={avg_loss:.4f}")

    out = Path(cfg["checkpoint"]["save_full_path"])
    torch.save(model.state_dict(), out)
    print(f"[Save] {out}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n[DBG] Uncaught exception!\n", file=sys.stderr)
        traceback.print_exc()
        raise
