#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/infer_to_csv_v7.py · Kaggle 提交（對應 modelv7/v7.yaml）
- 讀 v7.yaml（可在檔頭覆寫）
- 僅用 torchvision 的 GeneralizedRCNNTransform（不做外部 resize）
- Image_ID：→『去前導零的數字字串』（如 "00000001" → "1"）
- PredictionString：每框 6 欄 "score x y w h 0"；無檢出→空字串
- 自檢：每列為 6 的倍數、不得出現 NaN/inf
"""

from pathlib import Path
import os, glob, csv
from typing import List
from PIL import Image
import torch
from torchvision.transforms import functional as TF
from torchvision.ops import nms as hard_nms

# ======== 檔頭可調 ========
CFG_PATH = "experiments/configs/v7.yaml"
WEIGHTS_PATH = "experiments/logs/fasterrcnn_v7/fasterrcnn_v7_best.pth"
IMG_DIR = "data/test/img"
OUT_CSV = "submissions/fasterrcnn_v7_submission.csv"

# 推論後處理（外部控制，不依賴模型內門檻）
SCORE_THRESH = 0.05
NMS_THRESH   = 0.60
MAX_DETS_PER_IMG = 300
CLIP_TO_IMAGE = True
ROUND_DECIMALS = 2
CLASS_ID = 0                 # 單類別 pig
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SHOW_PROGRESS = True
DRYRUN_MAX_IMAGES = None
# ==========================

# 專案工具
from config import load_cfg
from modelv7 import build_detector_from_cfg


def _list_images(img_dir: str) -> List[str]:
    exts = ("*.jpg","*.jpeg","*.png","*.bmp","*.JPG","*.JPEG","*.PNG","*.BMP")
    files: List[str] = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(img_dir, ext)))

    def _key(fp: str):
        base = os.path.splitext(os.path.basename(fp))[0]
        s = base.lstrip("0")
        try:
            return int(s) if s != "" else 0
        except ValueError:
            return base

    files = sorted(files, key=_key)
    if DRYRUN_MAX_IMAGES is not None:
        files = files[:DRYRUN_MAX_IMAGES]
    return files


def _xyxy_to_xywh(boxes: torch.Tensor) -> torch.Tensor:
    # x1,y1,x2,y2 -> x,y,w,h，並確保 w,h >= 1.0
    x1, y1, x2, y2 = boxes.unbind(-1)
    w = (x2 - x1).clamp(min=1.0)
    h = (y2 - y1).clamp(min=1.0)
    return torch.stack([x1, y1, w, h], dim=-1)


def _clip_boxes_xyxy(boxes: torch.Tensor, W: int, H: int) -> torch.Tensor:
    boxes[:, 0].clamp_(0, W - 1)
    boxes[:, 2].clamp_(0, W - 1)
    boxes[:, 1].clamp_(0, H - 1)
    boxes[:, 3].clamp_(0, H - 1)
    # 修正 x2<x1 / y2<y1
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
    for i in range(0, len(toks), 6):
        try:
            float(toks[i]); float(toks[i+1]); float(toks[i+2]); float(toks[i+3]); float(toks[i+4]); int(toks[i+5])
        except Exception:
            return False
    return True


def _image_id_from_path(fp: str) -> str:
    # Kaggle 期望「無前導零的數字字串」
    base = os.path.splitext(os.path.basename(fp))[0]
    if base.isdigit() or base.lstrip("0").isdigit():
        s = base.lstrip("0")
        return s if s != "" else "0"
    return base


def _ensure_outdir(path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _safe_load_state_dict(ckpt_path: str, device: str):
    try:
        sd = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        sd = torch.load(ckpt_path, map_location=device)
    if isinstance(sd, dict):
        if "state_dict" in sd and isinstance(sd["state_dict"], dict):
            sd = sd["state_dict"]
        elif "model" in sd and isinstance(sd["model"], dict):
            sd = sd["model"]
        def _strip_prefix(d, prefixes=("module.", "model.")):
            out = {}
            for k, v in d.items():
                nk = k
                for p in prefixes:
                    if nk.startswith(p): nk = nk[len(p):]
                out[nk] = v
            return out
        if all(isinstance(k, str) for k in sd.keys()):
            sd = _strip_prefix(sd)
    return sd


def main():
    print("[Infer] loading cfg:", CFG_PATH)
    cfg = load_cfg(CFG_PATH)

    # 權重路徑（優先 WEIGHTS_PATH）
    ckpt_from_yaml = None
    try:
        ckpt_from_yaml = cfg["checkpoint"]["save_full_path"]
    except Exception:
        ckpt_from_yaml = None
    ckpt = WEIGHTS_PATH or ckpt_from_yaml
    if not ckpt or not Path(ckpt).exists():
        raise FileNotFoundError(f"weights not found: {ckpt}")

    # 建模
    model = build_detector_from_cfg(cfg).to(DEVICE)
    model.eval()
    # 關閉內部 score 門檻，外部再控（避免「雙重門檻」）
    try:
        if hasattr(model, "roi_heads") and hasattr(model.roi_heads, "score_thresh"):
            model.roi_heads.score_thresh = 0.0
    except Exception:
        pass

    # 載入權重
    sd = _safe_load_state_dict(ckpt, DEVICE)
    model.load_state_dict(sd, strict=True)
    print(f"[Model] device={DEVICE} | weights={ckpt}")

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
            x = TF.to_tensor(img).to(DEVICE)

            with torch.inference_mode():
                out = model([x])[0]

            boxes = out.get("boxes", torch.empty((0, 4), device=DEVICE))
            scores = out.get("scores", torch.empty((0,), device=DEVICE))

            # 過濾 NaN/inf
            if boxes.numel() > 0:
                finite = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores)
                boxes, scores = boxes[finite], scores[finite]

            # 外部 score 門檻
            keep = scores >= float(SCORE_THRESH)
            boxes, scores = boxes[keep], scores[keep]

            # Hard-NMS
            if boxes.numel() > 0:
                keep_idx = hard_nms(boxes, scores, float(NMS_THRESH))
                boxes, scores = boxes[keep_idx], scores[keep_idx]

            # 依分數排序、截斷
            if scores.numel() > 0:
                order = torch.argsort(scores, descending=True)[:int(MAX_DETS_PER_IMG)]
                boxes, scores = boxes[order], scores[order]

            # 裁邊界、轉 xywh（w,h>=1.0）
            if CLIP_TO_IMAGE and boxes.numel() > 0:
                boxes = _clip_boxes_xyxy(boxes, W, H)
            boxes_xywh = _xyxy_to_xywh(boxes) if boxes.numel() > 0 else torch.empty((0, 4), device=DEVICE)

            image_id = _image_id_from_path(fp)
            pred_str = _format_row(scores, boxes_xywh)

            # 自檢
            if not _is_valid_pred_string(pred_str):
                print(f"[WARN] invalid PredictionString on image {image_id}; fallback to empty.")
                pred_str = ""

            writer.writerow([image_id, pred_str])

    print(f"[Done] wrote CSV -> {OUT_CSV}")


if __name__ == "__main__":
    main()
