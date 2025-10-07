#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
trainv7.py — YAML-driven detector training (clean & safe)
對應：modelv7.py、transforms.py、experiments/configs/v7.yaml

設計原則：
- 不用 CLI；所有參數來自 YAML，必要覆寫集中在檔頭 OVERRIDES。
- 功能：AMP（PyTorch 2.x）、Cosine+Warmup 或 StepLR、EMA、Early Stop、
        凍結/解凍 backbone、NaN/Inf 防呆、可選 DBG 首批次檢查、異常批次旁路（skip-OOD）。
- 與 torchvision detection 相容：val 階段維持 model.train() 以回傳 loss。
"""

from pathlib import Path
import sys, math, random, traceback
from typing import Dict, Any, List

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, LambdaLR
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from config import load_cfg
from modelv7 import build_detector_from_cfg
from dataset import PigsDataset, collate_fn
from transforms import get_transforms

# ========= 可在這裡覆寫 YAML 參數（不需 CLI）=========
CFG_PATH   = "experiments/configs/v7.yaml"
OVERRIDES  = [
    # 例："train.epochs=36", "optimizer.lr=0.0025"
]
# ===============================================

# === 開關：為了安全，先關閉任何自動 label 轉換 ===
ENABLE_LABEL_REMAP = False


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
    """預設不改 label；必要時再開 ENABLE_LABEL_REMAP。"""
    if not ENABLE_LABEL_REMAP:
        return targets
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
    min_lr_ratio = float(scfg.get("min_lr_ratio", 0.0))

    def lr_lambda(cur_iter):
        # 線性 warmup
        if warmup_steps > 0 and cur_iter < warmup_steps:
            return start_factor + (1.0 - start_factor) * (cur_iter / float(max(1, warmup_steps)))
        # cosine 到 min_lr_ratio
        progress = (cur_iter - warmup_steps) / float(max(1, total_iters - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        cos = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cos

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
                    grad_clip=None, amp_enabled=False, scaler: GradScaler=None, scheduler_iter=None, accumulate=1,
                    print_interval=10, ema: ModelEMA = None, detector_name: str = "",
                    skip_cfg: Dict[str, Any] = None):
    model.train()
    loss_smooth = SmoothedValue()
    pbar = tqdm(loader, total=len(loader), ncols=120, desc=f"Epoch {epoch_idx+1}/{total_epochs}")

    optimizer.zero_grad(set_to_none=True)
    total_loss, steps = 0.0, 0

    # 旁路設定
    skip_enabled = bool((skip_cfg or {}).get("enabled", True))
    thr_rpn_reg = float((skip_cfg or {}).get("rpn_box_reg_thresh", 200.0))
    thr_roi_reg = float((skip_cfg or {}).get("box_reg_thresh", 100.0))
    skip_log_every = int((skip_cfg or {}).get("log_every", 50))

    for step, (images, targets) in enumerate(pbar, 1):
        images  = [img.to(device, non_blocking=True) for img in images]
        targets = _targets_to_device([{k: v for k, v in t.items()} for t in targets], device)
        targets = _remap_labels_for_detector(detector_name, targets)

        with autocast("cuda", enabled=amp_enabled and device.type == "cuda"):
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

        # NaN/Inf guard
        loss_val = float(loss.detach().item())
        if (math.isnan(loss_val)) or (math.isinf(loss_val)):
            comp = {k: float(v.detach().item()) for k, v in loss_dict.items()}
            print(f"\n[NaN/Inf DETECTED] epoch={epoch_idx+1} step={step} "
                  f"loss={loss_val} parts={comp} lr={optimizer.param_groups[0]['lr']:.3e}")
            raise RuntimeError("Loss became NaN/Inf. Check LR/AMP/labels/boxes.")

        # 異常批次旁路（skip-OOD）
        if skip_enabled:
            comp = {k: float(v.detach().item()) for k, v in loss_dict.items()}
            too_large = (comp.get("loss_rpn_box_reg", 0.0) > thr_rpn_reg) or (comp.get("loss_box_reg", 0.0) > thr_roi_reg)
            if too_large:
                if (step % max(1, skip_log_every)) == 0:
                    print(f"[SKIP-OOD] epoch={epoch_idx+1} step={step} parts={comp}")
                optimizer.zero_grad(set_to_none=True)
                del loss_dict, loss, images, targets
                continue

        loss_scaled = loss / max(1, accumulate)
        if amp_enabled and scaler is not None and device.type == "cuda":
            scaler.scale(loss_scaled).backward()
        else:
            loss_scaled.backward()

        if step % max(1, accumulate) == 0:
            if grad_clip:
                if amp_enabled and scaler is not None and device.type == "cuda":
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            if amp_enabled and scaler is not None and device.type == "cuda":
                scaler.step(optimizer); scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler_iter is not None:
                # 正確順序：optimizer.step() 之後再 scheduler.step()
                scheduler_iter.step()
            if ema is not None:
                ema.update(model)

        total_loss += loss_val
        steps += 1
        loss_smooth.update(loss_val)
        if step % max(1, print_interval) == 0:
            pbar.set_postfix({"loss": f"{loss_smooth.value():.3f}",
                              "lr": f"{optimizer.param_groups[0]['lr']:.2e}"})

        # 釋放張量引用
        del loss_dict, loss, images, targets

    return total_loss / max(1, steps)


@torch.no_grad()
def debug_one_batch(model, loader, device, detector_name: str = ""):
    model.train()
    images, targets = next(iter(loader))
    images  = [img.to(device, non_blocking=True) for img in images]
    targets = _targets_to_device([{k: v for k, v in t.items()} for t in targets], device)
    targets = _remap_labels_for_detector(detector_name, targets)

    # 1) 標籤與框統計
    all_labels = torch.cat([t["labels"].cpu() for t in targets if "labels" in t and len(t["labels"])>0], dim=0)
    all_boxes  = torch.cat([t["boxes"].cpu()  for t in targets if "boxes"  in t and len(t["boxes"]) >0], dim=0)
    x1y1 = all_boxes[:, :2]; x2y2 = all_boxes[:, 2:]
    wh = (x2y2 - x1y1).clamp(min=0)
    print(f"[DBG-BATCH] labels uniq={torch.unique(all_labels).tolist()} "
          f"boxes N={len(all_boxes)}, w/h mean={wh.mean(0).tolist()}, min={wh.min(0).values.tolist()}")

    # 2) 損失細項
    loss_dict = model(images, targets)
    comp = {k: float(v.detach().item()) for k, v in loss_dict.items()}
    print(f"[DBG-BATCH] loss parts = {comp} | sum={sum(comp.values()):.4f}")


def validate_one_epoch(model, loader, device, amp_enabled=False, detector_name: str = ""):
    """注意：torchvision detection 要保持 model.train() 才會回傳 loss。"""
    model.train()
    total = 0.0
    n = 0
    for images, targets in loader:
        images  = [img.to(device, non_blocking=True) for img in images]
        targets = _targets_to_device([{k: v for k, v in t.items()} for t in targets], device)
        targets = _remap_labels_for_detector(detector_name, targets)

        with autocast("cuda", enabled=amp_enabled and device.type == "cuda"):
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

        total += float(loss.item())
        n += 1

        # 每 50 個 batch 印一次詳細 loss 組成
        if (n % 50) == 0:
            comp = {k: float(v.detach().item()) for k, v in loss_dict.items()}
            print(f"[VAL DBG] step={n} loss={float(loss.item()):.4f} parts={comp}")

        del loss_dict, loss, images, targets

    return total / max(1, n)


def _set_backbone_requires_grad(model: torch.nn.Module, enable: bool):
    """嘗試凍結/解凍 backbone；依常見命名覆蓋到 FPN 前的主幹。"""
    names = ["backbone.body", "backbone", "transformer.backbone"]
    found = False
    for n, p in model.named_parameters():
        if any(n.startswith(k) for k in names):
            p.requires_grad_(enable)
            found = True
    return found


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

    # 轉換（與 YAML 對齊）
    A = cfg.get("augment", {}) or {}
    max_side = int(A.get("max_side", cfg.get("model", {}).get("max_size", 1280)))
    train_tfms = get_transforms(
        train=True,
        max_side=max_side,
        flip_p=float(A.get("flip_p", 0.5)),
        hsv=A.get("hsv", [0.015, 0.70, 0.40]),
        resize=A.get("resize", [896, 1024]),
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

    # 可選 DBG 首批次
    if bool(cfg.get("train", {}).get("debug_first_batch", False)):
        debug_one_batch(model, loader_train, device, det_name)

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

    # === 先凍結再建 Optimizer（關鍵順序） ===
    T = cfg.get("train", {}) or {}
    freeze_epochs = int(T.get("freeze_backbone_epochs", 0))
    if freeze_epochs > 0:
        if _set_backbone_requires_grad(model, False):
            print(f"[Backbone] frozen for first {freeze_epochs} epochs.")
        else:
            print("[Backbone][WARN] 未找到可凍結之 backbone 參數（名稱不符？）")

    # Optimizer（只挑 requires_grad=True 的參數）
    params = [p for p in model.parameters() if p.requires_grad]
    O = cfg.get("optimizer", {}) or {}
    opt_name = str(O.get("name", O.get("type", "sgd"))).lower()
    if opt_name == "adamw":
        optimizer = torch.optim.AdamW(
            params,
            lr=float(O.get("lr", 0.0005)),
            weight_decay=float(O.get("weight_decay", 0.0005)),
            betas=tuple(O.get("betas", [0.9, 0.999])),
        )
    else:
        optimizer = torch.optim.SGD(
            params,
            lr=float(O.get("lr", 0.005)),
            momentum=float(O.get("momentum", 0.9)),
            weight_decay=float(O.get("weight_decay", 0.0005)),
        )

    # Scheduler（iteration 級 cosine 或 epoch 級 StepLR）
    S = cfg.get("scheduler", {}) or {}
    epochs = int(T.get("epochs", 30))
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
    amp_enabled = bool(T.get("amp", True))
    scaler      = GradScaler(enabled=amp_enabled and device.type == "cuda")
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

    # Skip-OOD 門檻（可由 YAML train.skip_ood.* 覆蓋）
    skip_cfg = (T.get("skip_ood", {}) or {})

    # 訓練回圈
    best_val = float("inf") if es_mode == "min" else -float("inf")
    print(f"[Train] epochs={epochs}, batch={T.get('batch_size', 2)}, amp={amp_enabled}, accumulate={accumulate}")
    for epoch in range(epochs):
        # 達到解凍時間點時，解凍 backbone（並把新解凍參數加入 optimizer）
        if freeze_epochs > 0 and epoch == freeze_epochs:
            if _set_backbone_requires_grad(model, True):
                print(f"[Backbone] unfrozen at epoch {epoch+1}.")
                # 重新把 requires_grad=True 的參數補進 optimizer（保留已存在的 param group）
                new_params = [p for p in model.parameters() if p.requires_grad and (not any(p is q for g in optimizer.param_groups for q in g['params']))]
                if len(new_params):
                    optimizer.add_param_group({"params": new_params})
            else:
                print("[Backbone][WARN] 解凍失敗（名稱不符？）")

        avg_train = train_one_epoch(
            model, loader_train, optimizer, device, epoch, epochs,
            grad_clip=grad_clip, amp_enabled=amp_enabled, scaler=scaler,
            scheduler_iter=scheduler_iter, accumulate=accumulate,
            print_interval=print_itvl, ema=ema, detector_name=det_name,
            skip_cfg=skip_cfg
        )

        model_for_val = ema.ema if ema_enabled else model
        avg_val = validate_one_epoch(model_for_val, loader_val, device, amp_enabled=amp_enabled, detector_name=det_name)

        if 'scheduler_epoch' in locals() and scheduler_epoch is not None:
            scheduler_epoch.step()

        print(f"[Epoch {epoch+1}/{epochs}] train_loss={avg_train:.4f} | val_loss={avg_val:.4f} | lr={optimizer.param_groups[0]['lr']:.3e}")

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
                cond = (es_mode == "min" and monitor_val <= best_val) or (es_mode == "max" and monitor_val >= best_val)
                if cond:
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
