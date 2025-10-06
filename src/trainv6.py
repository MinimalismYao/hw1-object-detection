#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
trainv6.py — YAML-driven detector training (no pretrained weights)
對應：modelv6.py、transforms.py、experiments/configs/v6.yaml

重點更新：
1) 改為由 YAML 的 model.detector 建模（透過 modelv6.build_detector_from_cfg）
   - 可選：fasterrcnn_r50_fpn / fasterrcnn_mbv3_fpn / fasterrcnn_r101_fpn / retinanet_r50_fpn / ssdlite_mbv3
2) 其餘流程（資料管線、AMP、early-stop、accumulate、StepLR）保持不變
"""

import os, time, math, traceback, sys, random
from pathlib import Path
from contextlib import nullcontext
from typing import Dict, Any

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from dataset import PigsDataset, collate_fn
from transforms import get_transforms
# from modelv6 import get_fasterrcnn_r50_fpn
from modelv6 import build_detector_from_cfg  # ← 改用模型工廠
from omegaconf import OmegaConf
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
        self.mode, self.patience = mode, int(patience)
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


# ---------------- Sanity Check ----------------
@torch.no_grad()
def quick_sanity_check(model, device, cfg: Dict[str, Any]):
    tfm = get_transforms(train=True, max_side=cfg["sanity_check"]["max_side"])
    ds = PigsDataset(cfg["data"]["train_img_dir"], cfg["data"]["train_gt"], transforms=tfm)
    loader = DataLoader(ds, batch_size=cfg["sanity_check"]["batch_size"], shuffle=True, num_workers=0, collate_fn=collate_fn)
    model.train()
    images, targets = next(iter(loader))
    images  = [im.to(device) for im in images]
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    loss = sum(model(images, targets).values())
    print(f"[Sanity] one-step OK, loss={loss.item():.4f}")


# ---------------- Train / Val ----------------
def train_one_epoch(model, loader, optimizer, device, epoch_idx, total_epochs,
                    grad_clip=10.0, amp=False, scaler=None, scheduler_iter=None, accumulate=1, print_interval=10):
    model.train()
    total_loss, n = 0.0, 0
    loss_smooth = SmoothedValue()
    autocast_ctx = torch.amp.autocast("cuda") if (amp and device.type == "cuda") else nullcontext()

    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(loader, total=len(loader), ncols=120, desc=f"Epoch {epoch_idx+1}/{total_epochs}")
    for step, (images, targets) in enumerate(pbar, 1):
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        with autocast_ctx:
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

        loss_scaled = loss / accumulate
        if amp and scaler is not None:
            scaler.scale(loss_scaled).backward()
        else:
            loss_scaled.backward()

        if step % max(1, accumulate) == 0:
            if grad_clip:
                if amp and scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            if amp and scaler is not None:
                scaler.step(optimizer); scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler_iter is not None:
                scheduler_iter.step()

        total_loss += float(loss.item()); n += 1
        loss_smooth.update(float(loss.item()))
        if step % max(1, print_interval) == 0:
            pbar.set_postfix({
                "loss": f"{loss_smooth.value():.3f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}"
            })
        del loss_dict, loss, images, targets
    return total_loss / max(1, n)


@torch.no_grad()
def validate_one_epoch(model, loader, device, amp=False):
    model.train()  # torchvision detection head 在 train 模式回傳 loss
    total, n = 0.0, 0
    autocast_ctx = torch.amp.autocast("cuda") if (amp and device.type == "cuda") else nullcontext()
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
    cfg_path = project_root / "experiments/configs/v6.yaml"
    print(f"[DBG] CFG_PATH={cfg_path} -> exists={cfg_path.exists()}")

    # 1) 載入 + resolve YAML
    cfg_raw = load_cfg(str(cfg_path))  # 可能回傳 OmegaConf 或普通 dict
    try:
        if OmegaConf.is_config(cfg_raw):
            cfg = OmegaConf.to_container(cfg_raw, resolve=True)
        else:
            # 將普通 dict 包成 OmegaConf，再做變數展開（支援 ${...}）
            cfg = OmegaConf.to_container(OmegaConf.create(cfg_raw), resolve=True)
    except Exception:
        # 保底：就用原始 dict（不展開 ${...}），但不建議
        print("[WARN] OmegaConf resolve 失敗，改用原始 cfg（${...} 不會被展開）")
        cfg = cfg_raw

    # 2) 固定隨機種子 & CUDNN
    seed = int(cfg.get("project", {}).get("seed", 42))
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = bool(cfg["train"].get("cudnn_benchmark", True))

    device = torch.device("cuda" if (cfg["device"]["cuda"] and torch.cuda.is_available()) else "cpu")
    print(f"[Device] {device}")

    # 3) 模型（從 YAML 的 model.detector 建模）
    model = build_detector_from_cfg(cfg).to(device)
    det_name = str(cfg.get("model", {}).get("detector", "fasterrcnn_r50_fpn"))
    print(f"[DBG] model built OK — detector={det_name}")

    # 4) Sanity check
    if cfg["sanity_check"]["enabled"]:
        quick_sanity_check(model, device, cfg)

    # 5) 資料集 & 轉換
    train_tfms = get_transforms(
        train=True,
        max_side=cfg["augment"]["max_side"],
        flip_p=cfg["augment"]["flip_p"],
        hsv=cfg["augment"]["hsv"],                       # ← 改為可傳 [h,s,v]
        resize=cfg["augment"]["resize"],
        mosaic=cfg["augment"]["mosaic"],
        min_box_size=float(cfg["data"]["min_box_wh"][0]),# ← 接 data.min_box_wh
        color_jitter_prob=cfg["augment"]["color_jitter_prob"],
        color_jitter=cfg["augment"]["color_jitter"],     # ← [b,c,s,h]
    )
    val_tfms = get_transforms(
        train=False,
        max_side=cfg["augment"]["max_side"],
        min_box_size=float(cfg["data"]["min_box_wh"][0]),# ← 驗證集同樣門檻
    )

    ds_train = PigsDataset(cfg["data"]["train_img_dir"], cfg["data"]["train_gt"], transforms=train_tfms)
    ds_val   = PigsDataset(cfg["data"]["val_img_dir"],   cfg["data"]["val_gt"],   transforms=val_tfms)

    loader_train = DataLoader(
        ds_train,
        batch_size=cfg["train"]["batch_size"],
        shuffle=cfg["dataloader"]["shuffle"],
        num_workers=cfg["data"]["num_workers"],
        pin_memory=cfg["dataloader"]["pin_memory"],
        persistent_workers=cfg["dataloader"]["persistent_workers"],
        prefetch_factor=cfg["dataloader"]["prefetch_factor"],
        collate_fn=collate_fn,
    )
    loader_val = DataLoader(
        ds_val,
        batch_size=max(1, cfg["sanity_check"]["batch_size"]),
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=cfg["dataloader"]["pin_memory"],
        persistent_workers=cfg["dataloader"]["persistent_workers"],
        prefetch_factor=cfg["dataloader"]["prefetch_factor"],
        collate_fn=collate_fn,
    )
    print(f"[Data] train={len(ds_train)} | val={len(ds_val)}")

    # 6) Optimizer / Scheduler / AMP / Accumulate / EarlyStop
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=cfg["optimizer"]["lr"],
        momentum=cfg["optimizer"]["momentum"],
        weight_decay=cfg["optimizer"]["weight_decay"],
    )

    scheduler = None
    if str(cfg["scheduler"]["type"]).lower() == "steplr":
        scheduler = StepLR(
            optimizer,
            step_size=int(cfg["scheduler"]["step_size"]),
            gamma=float(cfg["scheduler"]["gamma"])
        )

    use_amp = bool(cfg["train"]["amp"])
    scaler = torch.amp.GradScaler(enabled=use_amp)
    accumulate = int(cfg["train"].get("accumulate", 1))
    grad_clip = float(cfg["train"].get("grad_clip", 0.0)) or None
    print_interval = int(cfg["logging"].get("print_interval", 10))

    es_cfg = cfg.get("early_stop", {})
    es_enabled = bool(es_cfg.get("enabled", False))
    es_monitor = str(es_cfg.get("monitor", "val_loss"))
    es_mode    = str(es_cfg.get("mode", "min"))
    es_patience= int(es_cfg.get("patience", 6))
    es_savebest= bool(es_cfg.get("save_best", True))
    early_stopper = EarlyStopping(mode=es_mode, patience=es_patience) if es_enabled else None

    # 7) 訓練回圈
    epochs = int(cfg["train"]["epochs"])
    best_val = float("inf") if es_mode == "min" else -float("inf")
    ckpt_dir = Path(cfg["checkpoint"]["dir"]); ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_last = Path(cfg["checkpoint"]["save_full_path"])
    ckpt_best = ckpt_dir / cfg["checkpoint"]["name"].replace(".pth", "_best.pth")

    print(f"[Train] epochs={epochs}, batch={cfg['train']['batch_size']}, amp={use_amp}, accumulate={accumulate}")
    for epoch in range(epochs):
        avg_train = train_one_epoch(
            model, loader_train, optimizer, device, epoch, epochs,
            grad_clip=grad_clip, amp=use_amp, scaler=scaler,
            scheduler_iter=None, accumulate=accumulate, print_interval=print_interval
        )
        avg_val = validate_one_epoch(model, loader_val, device, amp=use_amp)

        # per-epoch scheduler
        if scheduler is not None:
            scheduler.step()

        print(f"[Epoch {epoch+1}/{epochs}] train_loss={avg_train:.4f} | val_loss={avg_val:.4f}")

        # early stop
        monitor_val = avg_val if es_monitor == "val_loss" else avg_train
        is_best = False
        if es_enabled:
            if early_stopper.step(monitor_val):
                is_best = True
                best_val = monitor_val
            if early_stopper.should_stop:
                print(f"[EarlyStop] stopped at epoch {epoch+1} (best={best_val:.4f})")
                break

        # save best
        if es_savebest and is_best:
            torch.save(model.state_dict(), ckpt_best)
            print(f"[Save][best] {ckpt_best}")

    # 8) save last / or per-epoch
    save_last_only = bool(cfg["checkpoint"].get("save_last_only", True))
    if save_last_only:
        torch.save(model.state_dict(), ckpt_last)
        print(f"[Save][last] {ckpt_last}")
    else:
        save_name = f"{cfg['project']['run_name']}_epoch{epoch+1}.pth"
        save_path = ckpt_dir / save_name
        torch.save(model.state_dict(), save_path)
        print(f"[Save][epoch] {save_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n[DBG] Uncaught exception!\n", file=sys.stderr)
        traceback.print_exc()
        raise
