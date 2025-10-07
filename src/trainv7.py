#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
trainv7.py — YAML-driven detector training (clean & safe)
對應：modelv7.py、transforms.py、experiments/configs/v7.yaml

設計原則：
- 不用 CLI；所有參數來自 YAML，必要覆寫集中在檔頭 OVERRIDES。
- 與推論同概念：簡潔、可重現、安全。
- 功能：單類別標籤位移（Retina=0-based / FRCNN=1-based）、AMP、Cosine+Warmup 或 StepLR、
        EMA（可選）、Early Stop（可選），儲存 best/last state_dict（weights-only）。
- 可選「訓練後自動評估」：簡化成直接呼叫 eval.main() 並只覆寫 checkpoint 路徑。
"""

from pathlib import Path
import os, sys, math, random, traceback
from typing import Dict, Any, List

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, LambdaLR
from tqdm import tqdm
from PIL import Image  # 只是確保 pillow 安裝，未直接使用

from config import load_cfg
from modelv7 import build_detector_from_cfg
from dataset import PigsDataset, collate_fn
from transforms import get_transforms

# ========= 可在這裡覆寫 YAML 參數（不需 CLI）=========
CFG_PATH   = "experiments/configs/v7.yaml"
OVERRIDES  = [
    # 例： "train.epochs=36", "optimizer.lr=0.0025"
]
# ===============================================


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
        self.mode = mode
        self.patience = int(patience)
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


class ModelEMA:
    """簡單 EMA（不含 BN/Dropout 狀態）。"""
    def __init__(self, model: torch.nn.Module, decay: float = 0.9995, device=None):
        import copy
        self.decay = decay
        self.ema = copy.deepcopy(model).to(device or next(model.parameters()).device)
        self.ema.eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        d = self.decay
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if k in msd and v.dtype.is_floating_point:
                v.copy_(v * d + msd[k] * (1.0 - d))


def _targets_to_device(targets: List[dict], device):
    out = []
    for t in targets:
        tt = {}
        for k, v in t.items():
            if isinstance(v, torch.Tensor):
                if k == "boxes" and v.dtype != torch.float32:
                    v = v.float()
                tt[k] = v.to(device, non_blocking=True)
            else:
                tt[k] = v
        out.append(tt)
    return out


def _remap_labels_for_detector(detector_name: str, targets: List[dict]):
    """
    RetinaNet / SSD：前景 0-based（num_classes 不含背景）→ labels=0
    Faster R-CNN  ：前景 1-based（num_classes 含背景）→ labels=1
    """
    dn = (detector_name or "").lower()
    use_zero = ("retina" in dn) or ("ssd" in dn)
    out = []
    for t in targets:
        tt = dict(t)
        if "labels" in tt and isinstance(tt["labels"], torch.Tensor) and tt["labels"].numel() > 0:
            tt["labels"] = (torch.zeros_like(tt["labels"]) if use_zero else torch.ones_like(tt["labels"]))
        out.append(tt)
    return out


# ---------------- Scheduler（Cosine + Warmup，iteration 級） ----------------
def build_iter_scheduler(optimizer, scfg: Dict[str, Any], total_iters: int):
    name = str(scfg.get("name", scfg.get("type", "cosine"))).lower()
    if name not in ("cosine", "cosineanneal", "cosineannealing", "cosineannealinglr"):
        return None

    wcfg = scfg.get("warmup", {}) or {}
    warmup_steps = int(wcfg.get("steps", 0))
    start_factor = float(wcfg.get("start_factor", 0.1))

    def lr_lambda(cur_iter):
        # 線性 warmup
        if warmup_steps > 0 and cur_iter < warmup_steps:
            return start_factor + (1.0 - start_factor) * (cur_iter / float(max(1, warmup_steps)))
        # cosine 到 0
        progress = (cur_iter - warmup_steps) / float(max(1, total_iters - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


# ---------------- Quick Sanity Check ----------------
@torch.no_grad()
def quick_sanity_check(model, device, cfg: Dict[str, Any], detector_name: str):
    tfm = get_transforms(train=True, max_side=int(cfg.get("sanity_check", {}).get("max_side", 640)))
    ds = PigsDataset(cfg["data"]["train_img_dir"], cfg["data"]["train_gt"], transforms=tfm)
    loader = DataLoader(ds, batch_size=int(cfg.get("sanity_check", {}).get("batch_size", 2)),
                        shuffle=True, num_workers=0, collate_fn=collate_fn)
    model.train()
    images, targets = next(iter(loader))
    images  = [im.to(device, non_blocking=True) for im in images]
    targets = _targets_to_device([{k: v for k, v in t.items()} for t in targets], device)
    targets = _remap_labels_for_detector(detector_name, targets)
    loss = sum(model(images, targets).values())
    print(f"[Sanity] one-step OK, loss={loss.item():.4f}")


# ---------------- Train / Val ----------------
def train_one_epoch(model, loader, optimizer, device, epoch_idx, total_epochs,
                    grad_clip=None, amp=False, scaler=None, scheduler_iter=None, accumulate=1,
                    print_interval=10, ema: ModelEMA = None, detector_name: str = ""):
    model.train()
    loss_smooth = SmoothedValue()
    pbar = tqdm(loader, total=len(loader), ncols=120, desc=f"Epoch {epoch_idx+1}/{total_epochs}")

    optimizer.zero_grad(set_to_none=True)
    autocast_ctx = torch.amp.autocast(device_type="cuda") if (amp and device.type == "cuda") else torch.no_grad if False else None
    # 用簡潔語法包裝 autocast
    class _Ctx:
        def __enter__(self): return torch.amp.autocast(device_type="cuda").__enter__() if (amp and device.type=="cuda") else None
        def __exit__(self, exc_type, exc, tb): 
            if (amp and device.type=="cuda"): torch.amp.autocast(device_type="cuda").__exit__(exc_type, exc, tb)
    ctx = _Ctx()

    total_loss, steps = 0.0, 0
    for step, (images, targets) in enumerate(pbar, 1):
        images  = [img.to(device, non_blocking=True) for img in images]
        targets = _targets_to_device([{k: v for k, v in t.items()} for t in targets], device)
        targets = _remap_labels_for_detector(detector_name, targets)

        with (torch.amp.autocast(device_type="cuda") if (amp and device.type=="cuda") else torch.autocast("cpu", enabled=False)):
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            if amp and scaler is not None:
                scaler.step(optimizer); scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler_iter is not None:
                scheduler_iter.step()
            if ema is not None:
                ema.update(model)

        loss_val = float(loss.item())
        total_loss += loss_val
        steps += 1
        loss_smooth.update(loss_val)
        if step % max(1, print_interval) == 0:
            pbar.set_postfix({"loss": f"{loss_smooth.value():.3f}", "lr": f"{optimizer.param_groups[0]['lr']:.2e}"})

        # 釋放張量引用
        del loss_dict, loss, images, targets

    return total_loss / max(1, steps)


@torch.no_grad()
def validate_one_epoch(model, loader, device, amp=False, detector_name: str = ""):
    # torchvision detection 需保持 train() 以回傳 loss
    model.train()
    total, n = 0.0, 0
    for images, targets in loader:
        images  = [img.to(device, non_blocking=True) for img in images]
        targets = _targets_to_device([{k: v for k, v in t.items()} for t in targets], device)
        targets = _remap_labels_for_detector(detector_name, targets)
        with (torch.amp.autocast(device_type="cuda") if (amp and device.type=="cuda") else torch.autocast("cpu", enabled=False)):
            loss = sum(model(images, targets).values())
        total += float(loss.item())
        n += 1
    return total / max(1, n)


# ---------------- Main ----------------
def main():
    print("[DBG] Enter main()")
    project_root = Path(__file__).resolve().parents[1]
    cfg = load_cfg(str(project_root / CFG_PATH), overrides=OVERRIDES)
    assert isinstance(cfg, dict), "[Err] load_cfg 應回傳 dict"

    # 固定隨機性
    seed = int(cfg.get("project", {}).get("seed", 42))
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = bool(cfg.get("train", {}).get("cudnn_benchmark", True))

    # 裝置
    device = torch.device("cuda" if (cfg.get("device", {}).get("cuda", True) and torch.cuda.is_available()) else "cpu")
    print(f"[Device] {device}")

    # 模型（YAML 建模）
    det_name = str(cfg.get("model", {}).get("detector", "fasterrcnn_r101_fpn_v7"))
    model = build_detector_from_cfg(cfg).to(device)
    print(f"[DBG] model built OK — detector={det_name}")

    # Sanity check（可選）
    if bool(cfg.get("sanity_check", {}).get("enabled", True)):
        quick_sanity_check(model, device, cfg, det_name)

    # 轉換（與 YAML 對齊；若缺值使用保守預設）
    A = cfg.get("augment", {}) or {}
    max_side = int(A.get("max_side", cfg.get("model", {}).get("max_size", 1280)))
    train_tfms = get_transforms(
        train=True,
        max_side=max_side,
        flip_p=float(A.get("flip_p", 0.5)),
        hsv=A.get("hsv", [0.015, 0.70, 0.40]),
        resize=A.get("resize", [896, 1024, 1200, 1400]),
        mosaic=bool(A.get("mosaic", False)),
        min_box_size=float(cfg.get("data", {}).get("min_box_wh", [1, 1])[0]),
        color_jitter_prob=float(A.get("color_jitter_prob", 0.0)),
        color_jitter=A.get("color_jitter", [0.0, 0.0, 0.0, 0.0]),
    )
    val_tfms = get_transforms(
        train=False,
        max_side=max_side,
        min_box_size=float(cfg.get("data", {}).get("min_box_wh", [1, 1])[0]),
    )

    # Dataset / DataLoader
    ds_train = PigsDataset(cfg["data"]["train_img_dir"], cfg["data"]["train_gt"], transforms=train_tfms)
    ds_val   = PigsDataset(cfg["data"]["val_img_dir"],   cfg["data"]["val_gt"],   transforms=val_tfms)
    print(f"[Data] train={len(ds_train)} | val={len(ds_val)}")

    D = cfg.get("dataloader", {}) or {}
    loader_train = DataLoader(
        ds_train,
        batch_size=int(cfg.get("train", {}).get("batch_size", 2)),
        shuffle=bool(D.get("shuffle", True)),
        num_workers=int(cfg.get("data", {}).get("num_workers", 4)),
        pin_memory=bool(D.get("pin_memory", True)),
        persistent_workers=bool(D.get("persistent_workers", False)),
        prefetch_factor=int(D.get("prefetch_factor", 2)),
        collate_fn=collate_fn,
    )
    loader_val = DataLoader(
        ds_val,
        batch_size=max(1, int(cfg.get("sanity_check", {}).get("batch_size", 2))),
        shuffle=False,
        num_workers=int(cfg.get("data", {}).get("num_workers", 4)),
        pin_memory=bool(D.get("pin_memory", True)),
        persistent_workers=bool(D.get("persistent_workers", False)),
        prefetch_factor=int(D.get("prefetch_factor", 2)),
        collate_fn=collate_fn,
    )

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    O = cfg.get("optimizer", {}) or {}
    opt_name = str(O.get("name", O.get("type", "sgd"))).lower()
    if opt_name == "adamw":
        optimizer = torch.optim.AdamW(params, lr=float(O.get("lr", 0.0005)), weight_decay=float(O.get("weight_decay", 0.0005)))
    else:
        optimizer = torch.optim.SGD(params, lr=float(O.get("lr", 0.005)),
                                    momentum=float(O.get("momentum", 0.9)),
                                    weight_decay=float(O.get("weight_decay", 0.0005)))

    # Scheduler（iteration 級 cosine 或 epoch 級 StepLR）
    S = cfg.get("scheduler", {}) or {}
    epochs = int(cfg.get("train", {}).get("epochs", 30))
    total_iters = len(loader_train) * epochs
    scheduler_iter = None
    sname = str(S.get("name", S.get("type", "cosine"))).lower()
    if sname in ("cosine", "cosineanneal", "cosineannealing", "cosineannealinglr"):
        scheduler_iter = build_iter_scheduler(optimizer, S, total_iters)
        scheduler_epoch = None
    else:
        scheduler_iter = None
        scheduler_epoch = StepLR(optimizer, step_size=int(S.get("step_size", 8)), gamma=float(S.get("gamma", 0.1)))

    # AMP / Accumulate / EMA / EarlyStop
    T = cfg.get("train", {}) or {}
    use_amp     = bool(T.get("amp", True))
    scaler      = torch.amp.GradScaler(enabled=use_amp)
    accumulate  = int(T.get("accumulate", 1))
    grad_clip   = float(T.get("grad_clip", T.get("grad_clip_norm", 0.0))) or None
    print_itvl  = int(cfg.get("logging", {}).get("print_interval", 10))

    ema_cfg     = T.get("model_ema", {}) or {}
    ema_enabled = bool(ema_cfg.get("enabled", False))
    ema_decay   = float(ema_cfg.get("decay", 0.9995))
    ema         = ModelEMA(model, decay=ema_decay, device=device) if ema_enabled else None

    es_cfg      = T.get("early_stop", {}) or {}
    es_enabled  = bool(es_cfg.get("enabled", False))
    es_monitor  = str(es_cfg.get("monitor", "val_loss"))
    es_mode     = str(es_cfg.get("mode", "min"))
    es_patience = int(es_cfg.get("patience", 8))
    es_savebest = bool(es_cfg.get("save_best", True))
    stopper     = EarlyStopping(mode=es_mode, patience=es_patience) if es_enabled else None
    print(f"[EarlyStop] {'enabled' if es_enabled else 'disabled'} | monitor={es_monitor} mode={es_mode} patience={es_patience} save_best={es_savebest}")

    # Checkpoint 路徑
    C = cfg.get("checkpoint", {}) or {}
    ckpt_dir  = Path(C.get("dir", "experiments/logs"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_name = C.get("name", "last.pth")
    ckpt_last = ckpt_dir / ckpt_name
    ckpt_best = ckpt_dir / ckpt_name.replace(".pth", "_best.pth")

    # 訓練回圈
    best_val = float("inf") if es_mode == "min" else -float("inf")
    print(f"[Train] epochs={epochs}, batch={T.get('batch_size', 2)}, amp={use_amp}, accumulate={accumulate}")
    for epoch in range(epochs):
        avg_train = train_one_epoch(
            model, loader_train, optimizer, device, epoch, epochs,
            grad_clip=grad_clip, amp=use_amp, scaler=scaler,
            scheduler_iter=scheduler_iter, accumulate=accumulate,
            print_interval=print_itvl, ema=ema, detector_name=det_name
        )
        model_for_val = ema.ema if ema_enabled else model
        avg_val = validate_one_epoch(model_for_val, loader_val, device, amp=use_amp, detector_name=det_name)

        if 'scheduler_epoch' in locals() and scheduler_epoch is not None:
            scheduler_epoch.step()

        print(f"[Epoch {epoch+1}/{epochs}] train_loss={avg_train:.4f} | val_loss={avg_val:.4f}")

        # EarlyStop
        monitor_val = avg_val if es_monitor == "val_loss" else avg_train
        if es_enabled:
            improved = stopper.step(monitor_val)
            if improved and es_savebest:
                torch.save((ema.ema if ema_enabled else model).state_dict(), ckpt_best)
                print(f"[Save][best] {ckpt_best}")
            if stopper.should_stop:
                print(f"[EarlyStop] stop at epoch {epoch+1}, best={stopper.best:.4f}")
                break
        else:
            # 無 early stop 時也更新 best（方便拿 best.pth）
            if es_savebest:
                if (es_mode == "min" and monitor_val <= best_val) or (es_mode == "max" and monitor_val >= best_val):
                    best_val = monitor_val
                    torch.save((ema.ema if ema_enabled else model).state_dict(), ckpt_best)
                    print(f"[Save][best] {ckpt_best}")

    # Save last
    torch.save((ema.ema if ema_enabled else model).state_dict(), ckpt_last)
    print(f"[Save][last] {ckpt_last}")

    # 訓練後自動評估（簡化版，可選）
    if bool(cfg.get("eval", {}).get("run_after_train", False)):
        try:
            ckpt_use = ckpt_best if (ckpt_best.exists() and es_savebest) else ckpt_last
            print(f"[Post-Eval] Using checkpoint: {ckpt_use}")
            import importlib
            eval_mod = importlib.import_module("eval")
            # 讓 eval.py 使用相同 YAML + 覆寫 checkpoint
            if hasattr(eval_mod, "CFG_PATH"):
                setattr(eval_mod, "CFG_PATH", CFG_PATH)
            if hasattr(eval_mod, "OVERRIDES"):
                eval_mod.OVERRIDES = [f"checkpoint.save_full_path={ckpt_use}"]
            print("[Post-Eval] Running eval.main() ...")
            eval_mod.main()
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
