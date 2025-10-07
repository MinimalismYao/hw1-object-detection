#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
trainv7.py — YAML-driven detector training (no pretrained weights)
對應：modelv7.py、transforms.py、experiments/configs/v7.yaml

更新重點：
- 以 YAML 的 model.detector 建模（modelv7.build_detector_from_cfg）
- 自動處理單類資料的「標籤位移」：RetinaNet→0-based；FRCNN→1-based（含背景）
- 支援 Cosine + Warmup（iteration 級）與 EMA；與 v7.yaml 對齊
- 訓練完成自動以同 YAML + 本次 checkpoint 執行 eval.py
"""

import os, time, math, traceback, sys, random
from pathlib import Path
from contextlib import nullcontext
from typing import Dict, Any, List

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, LambdaLR
from tqdm import tqdm

from dataset import PigsDataset, collate_fn
from transforms import get_transforms
from modelv7 import build_detector_from_cfg  # v7：新的模型工廠
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


class ModelEMA:
    """滑動平均（EMA），每 step 更新（不含 BN/Dropout 狀態）。"""
    def __init__(self, model: torch.nn.Module, decay: float = 0.9995, device=None):
        self.decay = decay
        self.ema = self._clone_model(model).to(device or next(model.parameters()).device)
        self.ema.eval()
    def _clone_model(self, model):
        import copy
        ema = copy.deepcopy(model)
        for p in ema.parameters():
            p.requires_grad_(False)
        return ema
    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        d = self.decay
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if k in msd and v.dtype.is_floating_point:
                v.copy_(v * d + msd[k] * (1.0 - d))


def _targets_to_device(targets: List[dict], device):
    """把 target dict 內所有 tensor 欄位搬到指定 device，並確保 boxes=float32。"""
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
    RetinaNet (tv): num_classes 為「不含背景」，前景類別用 0 開始 -> labels=0
    Faster R-CNN (tv): num_classes「含背景」，前景從 1 開始 -> labels=1
    """
    dn = (detector_name or "").lower()
    out = []
    for t in targets:
        tt = dict(t)
        if "labels" in tt and isinstance(tt["labels"], torch.Tensor) and tt["labels"].numel() > 0:
            if "retinanet" in dn or "ssd" in dn:
                tt["labels"] = torch.zeros_like(tt["labels"])         # 0-based
            else:
                tt["labels"] = torch.ones_like(tt["labels"])          # 1-based（背景=0）
        out.append(tt)
    return out


# ---------------- Sanity Check ----------------
@torch.no_grad()
def quick_sanity_check(model, device, cfg: Dict[str, Any], detector_name: str):
    tfm = get_transforms(train=True, max_side=cfg["sanity_check"]["max_side"])
    ds = PigsDataset(cfg["data"]["train_img_dir"], cfg["data"]["train_gt"], transforms=tfm)
    loader = DataLoader(ds, batch_size=cfg["sanity_check"]["batch_size"], shuffle=True, num_workers=0, collate_fn=collate_fn)
    model.train()
    images, targets = next(iter(loader))
    images  = [im.to(device, non_blocking=True) for im in images]
    targets = _targets_to_device([{k: v for k, v in t.items()} for t in targets], device)
    targets = _remap_labels_for_detector(detector_name, targets)
    loss = sum(model(images, targets).values())
    print(f"[Sanity] one-step OK, loss={loss.item():.4f}")


# ---------------- Scheduler（Cosine + Warmup） ----------------
def build_iter_scheduler(optimizer, cfg: Dict[str, Any], total_iters: int):
    scfg = cfg.get("scheduler", {})
    name = str(scfg.get("name", scfg.get("type", "cosine"))).lower()
    if name not in ("cosine", "cosineanneal", "cosineannealing", "cosineannealinglr"):
        return None  # 交給 StepLR（epoch 級）

    wcfg = scfg.get("warmup", {}) or {}
    warmup_steps = int(wcfg.get("steps", 0))
    start_factor = float(wcfg.get("start_factor", 0.1))
    min_lr = 0.0

    base_lrs = [pg["lr"] for pg in optimizer.param_groups]

    def lr_lambda(cur_iter):
        # 0 ~ warmup_steps: 線性從 start_factor → 1.0
        if warmup_steps > 0 and cur_iter < warmup_steps:
            return start_factor + (1.0 - start_factor) * (cur_iter / float(max(1, warmup_steps)))
        # 其後：cosine 到 0
        progress = (cur_iter - warmup_steps) / float(max(1, total_iters - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        # 余弦從 1 → 0
        return 0.5 * (1.0 + math.cos(math.pi * (1.0 - progress)))

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    # 覆寫初始學習率（LambdaLR 只回傳倍率）
    for i, pg in enumerate(optimizer.param_groups):
        pg["initial_lr"] = base_lrs[i]
    return scheduler


# ---------------- Train / Val ----------------
def train_one_epoch(model, loader, optimizer, device, epoch_idx, total_epochs,
                    grad_clip=10.0, amp=False, scaler=None, scheduler_iter=None, accumulate=1,
                    print_interval=10, ema: ModelEMA = None, detector_name: str = ""):
    model.train()
    total_loss, n = 0.0, 0
    loss_smooth = SmoothedValue()
    autocast_ctx = torch.amp.autocast("cuda") if (amp and device.type == "cuda") else nullcontext()

    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(loader, total=len(loader), ncols=120, desc=f"Epoch {epoch_idx+1}/{total_epochs}")
    for step, (images, targets) in enumerate(pbar, 1):
        images  = [img.to(device, non_blocking=True) for img in images]
        targets = _targets_to_device([{k: v for k, v in t.items()} for t in targets], device)
        targets = _remap_labels_for_detector(detector_name, targets)

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
            if ema is not None:
                ema.update(model)

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
def validate_one_epoch(model, loader, device, amp=False, detector_name: str = ""):
    # 為了取得 torchvision detection loss，保持 train()（與 v6 相同做法）
    model.train()
    total, n = 0.0, 0
    autocast_ctx = torch.amp.autocast("cuda") if (amp and device.type == "cuda") else nullcontext()
    for images, targets in loader:
        images  = [img.to(device, non_blocking=True) for img in images]
        targets = _targets_to_device([{k: v for k, v in t.items()} for t in targets], device)
        targets = _remap_labels_for_detector(detector_name, targets)
        with autocast_ctx:
            loss = sum(model(images, targets).values())
        total += float(loss.item()); n += 1
    return total / max(1, n)


# ---------------- Main ----------------
def main():
    print("[DBG] Enter main()")
    project_root = Path(__file__).resolve().parents[1]
    cfg_path = project_root / "experiments/configs/v7.yaml"  # v7 預設
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

    device = torch.device("cuda" if (cfg.get("device", {}).get("cuda", True) and torch.cuda.is_available()) else "cpu")
    print(f"[Device] {device}")

    # 3) 模型（從 YAML 的 model.detector 建模）
    det_name = str(cfg.get("model", {}).get("detector", "fasterrcnn_r101_fpn_v7"))
    model = build_detector_from_cfg(cfg).to(device)
    print(f"[DBG] model built OK — detector={det_name}")

    # 4) Sanity check
    if cfg["sanity_check"]["enabled"]:
        quick_sanity_check(model, device, cfg, det_name)

    # 5) 資料集 & 轉換
    # 注意：下列引數需與你的 transforms.py 簽名相容（保持你原本使用法）
    train_tfms = get_transforms(
        train=True,
        max_side=cfg["augment"]["max_side"],
        flip_p=cfg["augment"]["horizontal_flip"] if "horizontal_flip" in cfg["augment"] else cfg["augment"]["flip_p"],
        hsv=cfg["augment"].get("hsv", [0.0, 0.0, 0.0]),
        resize=cfg["augment"].get("resize_scales", cfg["augment"].get("resize", [1280])),
        mosaic=cfg["augment"].get("mosaic", False),
        min_box_size=float(cfg["data"]["min_box_wh"][0]),
        color_jitter_prob=cfg["augment"].get("color_jitter_prob", 0.0),
        color_jitter=cfg["augment"].get("color_jitter", [0,0,0,0]),
        zoom_in_cfg=cfg["augment"].get("zoom_in_crop", {"enabled": False}),
        gaussian_blur_prob=cfg["augment"].get("gaussian_blur_prob", 0.0),
        gaussian_blur_sigma=cfg["augment"].get("gaussian_blur_sigma", [0.1, 1.0]),
        random_noise_prob=cfg["augment"].get("random_noise_prob", 0.0),
        normalize_mean_std=cfg["augment"].get("normalize_mean_std", [[0.485,0.456,0.406],[0.229,0.224,0.225]])
    )
    val_tfms = get_transforms(
        train=False,
        max_side=cfg["augment"]["max_side"],
        min_box_size=float(cfg["data"]["min_box_wh"][0]),
        normalize_mean_std=cfg["augment"].get("normalize_mean_std", [[0.485,0.456,0.406],[0.229,0.224,0.225]])
    )

    ds_train = PigsDataset(cfg["data"]["train_img_dir"], cfg["data"]["train_gt"], transforms=train_tfms)
    ds_val   = PigsDataset(cfg["data"]["val_img_dir"],   cfg["data"]["val_gt"],   transforms=val_tfms)

    loader_train = DataLoader(
        ds_train,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=bool(cfg["dataloader"]["shuffle"]),
        num_workers=int(cfg["data"]["num_workers"]),
        pin_memory=bool(cfg["dataloader"]["pin_memory"]),
        persistent_workers=bool(cfg["dataloader"]["persistent_workers"]),
        prefetch_factor=int(cfg["dataloader"]["prefetch_factor"]),
        collate_fn=collate_fn,
    )
    loader_val = DataLoader(
        ds_val,
        batch_size=max(1, int(cfg["sanity_check"]["batch_size"])),
        shuffle=False,
        num_workers=int(cfg["data"]["num_workers"]),
        pin_memory=bool(cfg["dataloader"]["pin_memory"]),
        persistent_workers=bool(cfg["dataloader"]["persistent_workers"]),
        prefetch_factor=int(cfg["dataloader"]["prefetch_factor"]),
        collate_fn=collate_fn,
    )
    print(f"[Data] train={len(ds_train)} | val={len(ds_val)}")

    # 6) Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    opt_name = str(cfg["optimizer"].get("name", cfg["optimizer"].get("type", "sgd"))).lower()
    if opt_name == "adamw":
        optimizer = torch.optim.AdamW(
            params,
            lr=float(cfg["optimizer"]["lr"]),
            weight_decay=float(cfg["optimizer"]["weight_decay"])
        )
    else:
        optimizer = torch.optim.SGD(
            params,
            lr=float(cfg["optimizer"]["lr"]),
            momentum=float(cfg["optimizer"]["momentum"]),
            weight_decay=float(cfg["optimizer"]["weight_decay"]),
        )

    # 7) Scheduler（iteration 級 Cosine + Warmup；或 epoch 級 StepLR）
    epochs = int(cfg["train"]["epochs"])
    total_iters = len(loader_train) * max(1, epochs)
    scheduler_iter = None
    scfg = cfg.get("scheduler", {})
    sname = str(scfg.get("name", scfg.get("type", "cosine"))).lower()
    if sname in ("cosine", "cosineanneal", "cosineannealing", "cosineannealinglr"):
        scheduler_iter = build_iter_scheduler(optimizer, cfg, total_iters)
        scheduler_epoch = None
    else:
        scheduler_iter = None
        scheduler_epoch = StepLR(
            optimizer,
            step_size=int(scfg.get("step_size", 8)),
            gamma=float(scfg.get("gamma", 0.1))
        )

    # 8) AMP / Accumulate / EMA / EarlyStop
    use_amp = bool(cfg["train"]["amp"])
    scaler = torch.amp.GradScaler(enabled=use_amp)
    accumulate = int(cfg["train"].get("accumulate", 1))
    grad_clip = float(cfg["train"].get("grad_clip", cfg["train"].get("grad_clip_norm", 0.0))) or None
    print_interval = int(cfg["logging"].get("print_interval", 10))

    ema_cfg = cfg["train"].get("model_ema", {})
    ema_enabled = bool(ema_cfg.get("enabled", False))
    ema_decay   = float(ema_cfg.get("decay", 0.9995))
    ema = ModelEMA(model, decay=ema_decay, device=device) if ema_enabled else None

    es_cfg = cfg["train"].get("early_stop", {})
    es_enabled = bool(es_cfg.get("enabled", False))
    es_monitor = str(es_cfg.get("monitor", "val_loss"))
    es_mode    = str(es_cfg.get("mode", "min"))
    es_patience= int(es_cfg.get("patience", 8))
    es_savebest= bool(es_cfg.get("save_best", True))
    early_stopper = EarlyStopping(mode=es_mode, patience=es_patience) if es_enabled else None

    # 9) 訓練回圈
    best_val = float("inf") if es_mode == "min" else -float("inf")
    ckpt_dir = Path(cfg["checkpoint"]["dir"]); ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_last = Path(cfg["checkpoint"]["save_full_path"])
    ckpt_best = ckpt_dir / cfg["checkpoint"]["name"].replace(".pth", "_best.pth")

    print(f"[Train] epochs={epochs}, batch={cfg['train']['batch_size']}, amp={use_amp}, accumulate={accumulate}")
    global_iter = 0
    for epoch in range(epochs):
        avg_train = train_one_epoch(
            model, loader_train, optimizer, device, epoch, epochs,
            grad_clip=grad_clip, amp=use_amp, scaler=scaler,
            scheduler_iter=scheduler_iter, accumulate=accumulate,
            print_interval=print_interval, ema=ema, detector_name=det_name
        )
        # 使用 EMA 模型驗證（若有）
        model_for_val = ema.ema if ema_enabled else model
        avg_val = validate_one_epoch(model_for_val, loader_val, device, amp=use_amp, detector_name=det_name)

        # per-epoch scheduler（若採 StepLR）
        if 'scheduler_epoch' in locals() and scheduler_epoch is not None:
            scheduler_epoch.step()

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
                # 落地一次 best
                if es_savebest:
                    torch.save((ema.ema if ema_enabled else model).state_dict(), ckpt_best)
                    print(f"[Save][best] {ckpt_best}")
                break

        # 保存 best（每 epoch 比較）
        if es_savebest and ( (es_mode=="min" and monitor_val<=best_val) or (es_mode=="max" and monitor_val>=best_val) ):
            best_val = monitor_val
            torch.save((ema.ema if ema_enabled else model).state_dict(), ckpt_best)
            print(f"[Save][best] {ckpt_best}")

    # 10) save last / or per-epoch
    save_last_only = bool(cfg["checkpoint"].get("save_last_only", True))
    to_save = (ema.ema if ema_enabled else model).state_dict()
    if save_last_only:
        torch.save(to_save, ckpt_last)
        print(f"[Save][last] {ckpt_last}")
    else:
        # 若非僅存最後，名稱帶 epoch
        save_name = f"{cfg['project']['run_name']}_epoch{epoch+1}.pth"
        save_path = ckpt_dir / save_name
        torch.save(to_save, save_path)
        print(f"[Save][epoch] {save_path}")

    # 11) 訓練後自動評估（以同一份 YAML + 本次 checkpoint）
    run_eval_after = bool(cfg.get("eval", {}).get("run_after_train", True))
    if run_eval_after:
        try:
            ckpt_use = ckpt_best if (es_savebest and ckpt_best.exists()) else ckpt_last
            print(f"[Post-Eval] Using checkpoint: {ckpt_use}")

            import importlib
            eval_mod = importlib.import_module("eval")

            # ---- 通用接管 ----
            from modelv7 import build_detector_from_cfg as _factory_for_eval
            def _patched_builder(**kwargs):
                cfg_in = kwargs.get("cfg")
                # 允許 eval 直接呼叫不帶參數
                return _factory_for_eval(cfg_in or cfg)

            # 若 eval.py 有 get_fasterrcnn_r50_fpn，改它指向 build_detector_from_cfg
            if hasattr(eval_mod, "get_fasterrcnn_r50_fpn"):
                setattr(eval_mod, "get_fasterrcnn_r50_fpn", _patched_builder)
            setattr(eval_mod, "build_detector_from_cfg", _patched_builder)

            # 指定 eval 讀同一份 YAML，並覆寫 checkpoint 路徑
            rel_cfg_path = str(cfg_path.relative_to(project_root))
            setattr(eval_mod, "CFG_PATH", rel_cfg_path)
            setattr(eval_mod, "OVERRIDES", [f"checkpoint.save_full_path={ckpt_use}"])

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
