#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
trainv6.py — YAML-driven detector training (no pretrained weights)
對應：modelv6.py、transforms.py、experiments/configs/v6.yaml

更新重點：
1) 由 YAML 的 model.detector 建模（透過 modelv6.build_detector_from_cfg）
2) 訓練完成後，自動整合 eval.py 執行驗證（可用 eval.run_after_train 開啟/關閉）
   - 使用與訓練同一份 YAML
   - 直接指定剛剛輸出的 checkpoint（best 優先，否則 last）
   - 讓 eval 也走「模型工廠」，不同 detector 自動對齊
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

        loss_scaled = loss / max(1, accumulate)
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
            cfg = OmegaConf.to_container(OmegaConf.create(cfg_raw), resolve=True)
    except Exception:
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
        hsv=cfg["augment"]["hsv"],
        resize=cfg["augment"]["resize"],
        mosaic=cfg["augment"]["mosaic"],
        min_box_size=float(cfg["data"]["min_box_wh"][0]),
        color_jitter_prob=cfg["augment"]["color_jitter_prob"],
        color_jitter=cfg["augment"]["color_jitter"],
    )
    val_tfms = get_transforms(
        train=False,
        max_side=cfg["augment"]["max_side"],
        min_box_size=float(cfg["data"]["min_box_wh"][0]),
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

    # 9) （新增）訓練後自動評估：整合 eval.py
    run_eval_after = bool(cfg.get("eval", {}).get("run_after_train", True))
    if run_eval_after:
        try:
            # 目標 checkpoint：若有 best 且設定存 best，就用 best；否則用 last
            ckpt_use = ckpt_best if (es_savebest and ckpt_best.exists()) else ckpt_last
            print(f"[Post-Eval] Using checkpoint: {ckpt_use}")

            # 匯入 eval 作為模組，避免與內建 eval() 衝突
            import importlib
            eval_mod = importlib.import_module("eval")

            # 讓 eval 也走「模型工廠」而不是寫死 R50-FPN：
            # eval.py 內部會呼叫 get_fasterrcnn_r50_fpn(..., cfg=cfg)
            # 我們把它 monkey-patch 成用 build_detector_from_cfg(cfg)
            from modelv6 import build_detector_from_cfg as _factory_for_eval
            def _patched_builder(**kwargs):
                # kwargs 會帶 cfg
                return _factory_for_eval(kwargs.get("cfg"))
            setattr(eval_mod, "get_fasterrcnn_r50_fpn", _patched_builder)

            # 指定 eval 讀同一份 YAML，並覆寫 checkpoint 路徑
            # eval.py 會：cfg = load_cfg(project_root/CFG_PATH, overrides=OVERRIDES)
            rel_cfg_path = str(cfg_path.relative_to(project_root))
            setattr(eval_mod, "CFG_PATH", rel_cfg_path)
            setattr(eval_mod, "OVERRIDES", [f"checkpoint.save_full_path={ckpt_use}"])

            print("[Post-Eval] Running eval.main() ...")
            eval_mod.main()  # 會印出 COCO 摘要，並寫 details/txt 到 cfg.eval.save_dir
            print("[Post-Eval] Done.")
        except Exception as e:
            print(f"[Post-Eval][WARN] 自動評估失敗：{e}")
            traceback.print_exc()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n[DBG] Uncaught exception!\n", file=sys.stderr)
        traceback.print_exc()
        raise
