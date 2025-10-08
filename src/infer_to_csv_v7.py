#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/infer_to_csv_v7.py · Kaggle 提交（對應 modelv7/v7.yaml）
- 讀 v7.yaml（可在檔頭覆寫）
- 僅用 torchvision 的 GeneralizedRCNNTransform（不做外部 resize）
- Image_ID：檔名數字（容忍前導零）；排序以數字為主、字串為備援
- PredictionString：每框 6 欄 "score x y w h 0"；無檢出→空字串
- 內建自檢：每列欄數為 6 的倍數、不得出現 NaN/inf
"""

from pathlib import Path
import os, glob, csv
from typing import List
from PIL import Image
import torch
import torchvision
from torchvision.transforms import functional as TF
from torchvision.ops import nms as hard_nms

# ======== 參數（檔頭可調） ========
CFG_PATH = "experiments/configs/v7.yaml"   # 你的 YAML
WEIGHTS_PATH = "experiments/logs/fasterrcnn_v7/fasterrcnn_v7_best.pth"  # 若為空會嘗試讀 YAML 的 checkpoint.save_full_path
IMG_DIR = "data/test/img"                  # 測試影像資料夾
OUT_CSV = "submissions/fasterrcnn_v7_submission.csv"  # 輸出檔名（自動建立資料夾）

# 推論門檻與後處理（僅影響推論，不影響已訓練權重）
SCORE_THRESH = 0.05                        # 建議 0.01~0.05
NMS_THRESH = 0.50                          # 再做一次 Hard-NMS（保險用）
MAX_DETS_PER_IMG = 300                     # 每張圖最多輸出
CLIP_TO_IMAGE = True                       # 產出前裁邊界
ROUND_DECIMALS = 1                         # CSV 小數位
CLASS_ID = 0                               # 單類別 pig → 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 顯示與限速
SHOW_PROGRESS = True
DRYRUN_MAX_IMAGES = None   # 例如 50；None 表示全量
# =================================

# ---- 匯入你的專案工具 ----
from config import load_cfg
from modelv7 import build_detector_from_cfg


def _list_images(img_dir: str) -> List[str]:
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.JPEG", "*.PNG", "*.BMP")
    files: List[str] = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(img_dir, ext)))

    def _key(fp: str):
        base = os.path.splitext(os.path.basename(fp))[0]
        try:
            s = base.lstrip("0")
            return int(s) if s != "" else 0
        except ValueError:
            return base

    files = sorted(files, key=_key)
    if DRYRUN_MAX_IMAGES is not None:
        files = files[:DRYRUN_MAX_IMAGES]
    return files


def _xyxy_to_xywh(box: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = box.unbind(-1)
    w = (x2 - x1).clamp(min=0)
    h = (y2 - y1).clamp(min=0)
    return torch.stack([x1, y1, w, h], dim=-1)


def _clip_boxes_xyxy(boxes: torch.Tensor, W: int, H: int) -> torch.Tensor:
    boxes[:, 0].clamp_(0, W - 1)
    boxes[:, 2].clamp_(0, W - 1)
    boxes[:, 1].clamp_(0, H - 1)
    boxes[:, 3].clamp_(0, H - 1)
    # 修正可能出現的 x2<x1 / y2<y1
    x1 = torch.min(boxes[:, 0], boxes[:, 2])
    x2 = torch.max(boxes[:, 0], boxes[:, 2])
    y1 = torch.min(boxes[:, 1], boxes[:, 3])
    y2 = torch.max(boxes[:, 1], boxes[:, 3])
    boxes[:, 0] = x1; boxes[:, 2] = x2
    boxes[:, 1] = y1; boxes[:, 3] = y2
    return boxes


def _format_row(scores: torch.Tensor, boxes_xywh: torch.Tensor) -> str:
    parts: List[str] = []
    for s, b in zip(scores.tolist(), boxes_xywh.tolist()):
        x, y, w, h = b
        parts += [
            f"{s:.6f}",
            f"{x:.{ROUND_DECIMALS}f}",
            f"{y:.{ROUND_DECIMALS}f}",
            f"{w:.{ROUND_DECIMALS}f}",
            f"{h:.{ROUND_DECIMALS}f}",
            str(CLASS_ID),
        ]
    return " ".join(parts)


def _is_valid_pred_string(pred_str: str) -> bool:
    if pred_str.strip() == "":
        return True
    toks = pred_str.strip().split()
    if len(toks) % 6 != 0:
        return False
    # 逐組檢查是否為數值/整數
    for i in range(0, len(toks), 6):
        try:
            float(toks[i])          # score
            float(toks[i+1])        # x
            float(toks[i+2])        # y
            float(toks[i+3])        # w
            float(toks[i+4])        # h
            int(toks[i+5])          # class id
        except Exception:
            return False
    return True


def _image_id_from_path(fp: str) -> str:
    base = os.path.splitext(os.path.basename(fp))[0]
    # 若不是純數字也可直接回 base；Kaggle 官方通常是數字
    return str(int(base.lstrip("0") or "0")) if base.isdigit() or base.lstrip("0").isdigit() else base


def _ensure_outdir(path: str):
    out_path = Path(path)
    if out_path.parent and not out_path.parent.exists():
        out_path.parent.mkdir(parents=True, exist_ok=True)


def _safe_load_state_dict(ckpt_path: str, device: str):
    # 盡量用新版安全載入；舊版 PyTorch 自動回退
    try:
        sd = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        sd = torch.load(ckpt_path, map_location=device)

    # 常見封裝鍵位
    if isinstance(sd, dict):
        if "state_dict" in sd and isinstance(sd["state_dict"], dict):
            sd = sd["state_dict"]
        elif "model" in sd and isinstance(sd["model"], dict):
            sd = sd["model"]

        # 剝除常見前綴
        def _strip_prefix(d, prefixes=("module.", "model.")):
            out = {}
            for k, v in d.items():
                nk = k
                for p in prefixes:
                    if nk.startswith(p):
                        nk = nk[len(p):]
                out[nk] = v
            return out

        if all(isinstance(k, str) for k in sd.keys()):
            sd = _strip_prefix(sd)

    return sd


def main():
    print("[Infer] loading cfg:", CFG_PATH)
    cfg = load_cfg(CFG_PATH)

    # 權重路徑（優先使用 WEIGHTS_PATH，否則嘗試 YAML）
    ckpt_from_yaml = None
    try:
        ckpt_from_yaml = cfg["checkpoint"]["save_full_path"]
    except Exception:
        ckpt_from_yaml = None
    ckpt = WEIGHTS_PATH or ckpt_from_yaml
    if not ckpt or not Path(ckpt).exists():
        raise FileNotFoundError(f"weights not found: {ckpt}")

    # 建模（與 v7.yaml 完整相容）
    model = build_detector_from_cfg(cfg).to(DEVICE)
    model.eval()
    try:
        torch.set_float32_matmul_precision("high")  # PyTorch 2.x
    except Exception:
        pass
    print(f"[Model] built OK · device={DEVICE}")

    # 載入權重
    sd = _safe_load_state_dict(ckpt, DEVICE)
    model.load_state_dict(sd, strict=True)
    print(f"[Load] weights loaded:", ckpt)

    # 列出影像
    img_files = _list_images(IMG_DIR)
    print(f"[Data] test images = {len(img_files)} | dir={IMG_DIR}")

    _ensure_outdir(OUT_CSV)
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Image_ID", "PredictionString"])

        iterator = img_files
        if SHOW_PROGRESS:
            from tqdm import tqdm
            iterator = tqdm(img_files, desc="Infer(v7)", ncols=100)

        torch.backends.cudnn.benchmark = True

        for fp in iterator:
            img = Image.open(fp).convert("RGB")
            W, H = img.size
            tensor = TF.to_tensor(img).to(DEVICE)  # [C,H,W], [0,1]

            with torch.no_grad():
                out = model([tensor])[0]

            boxes = out.get("boxes", torch.empty((0, 4), device=DEVICE))
            scores = out.get("scores", torch.empty((0,), device=DEVICE))

            # 過濾 NaN/inf
            if boxes.numel() > 0:
                finite_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores)
                boxes = boxes[finite_mask]
                scores = scores[finite_mask]

            # score 門檻
            keep = scores >= float(SCORE_THRESH)
            boxes = boxes[keep]
            scores = scores[keep]

            # Hard-NMS（單類別）
            if boxes.numel() > 0:
                keep_idx = hard_nms(boxes, scores, float(NMS_THRESH))
                boxes = boxes[keep_idx]
                scores = scores[keep_idx]

            # 依分數排序、截斷
            if scores.numel() > 0:
                order = torch.argsort(scores, descending=True)[:int(MAX_DETS_PER_IMG)]
                boxes = boxes[order]
                scores = scores[order]

            # 裁邊界、轉 xywh
            if CLIP_TO_IMAGE and boxes.numel() > 0:
                boxes = _clip_boxes_xyxy(boxes, W, H)
            boxes_xywh = _xyxy_to_xywh(boxes) if boxes.numel() > 0 else torch.empty((0, 4), device=DEVICE)

            image_id = _image_id_from_path(fp)
            pred_str = _format_row(scores, boxes_xywh)

            # 自檢：6 的倍數、不得 NaN/inf
            if not _is_valid_pred_string(pred_str):
                print(f"[WARN] invalid PredictionString on image {image_id}; fallback to empty.")
                pred_str = ""

            writer.writerow([image_id, pred_str])

    print(f"[Done] wrote CSV -> {OUT_CSV}")


if __name__ == "__main__":
    main()
