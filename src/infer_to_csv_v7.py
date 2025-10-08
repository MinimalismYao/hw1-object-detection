#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/infer_to_csv_v7.py · Kaggle 提交（對應 modelv7/v7.yaml）

特色
- 讀 v7.yaml（可在檔頭覆寫路徑）
- 不做外部 resize；交給 torchvision GeneralizedRCNNTransform（由 YAML 的 min/max_size 控）
- Image_ID：無前導零的數字字串（"00000001"→"1"），非純數字則用原字串
- PredictionString：每框 6 欄 "score x y w h 0"；無檢出→空字串
- 面積感知門檻：小框/超小框需要更高信心（抑制背景誤檢）
- 可選 Soft-NMS（Gaussian/Linear），預設關
- 嚴格自檢：6 欄倍數、不得 NaN/Inf

只需改檔頭常數，不用 CLI 參數。
"""

from pathlib import Path
import os, glob, csv
from typing import List, Tuple
from PIL import Image
import torch
from torchvision.transforms import functional as TF
from torchvision.ops import nms as hard_nms

# ======== 檔頭可調 ========
CFG_PATH = "experiments/configs/v7.yaml"
WEIGHTS_PATH = "experiments/logs/fasterrcnn_v7/fasterrcnn_v7_best.pth"
IMG_DIR = "data/test/img"
OUT_CSV = "submissions/fasterrcnn_v7_submission.csv"

# 推論超參（外部控制，不依賴模型內門檻）
SCORE_THRESH = 0.25          # 全域最低分數門檻（中/大框）
NMS_THRESH   = 0.55          # Hard-NMS IoU（僅當 USE_SOFT_NMS=False）
MAX_DETS_PER_IMG = 80        # 每張圖最多保留
CLIP_TO_IMAGE = True
ROUND_DECIMALS = 2           # bbox 輸出小數位
CLASS_ID = 0                 # 單類別 pig
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SHOW_PROGRESS = True
DRYRUN_MAX_IMAGES = None     # 例如 50；None 表示全量

# 面積感知門檻（相對於整張圖面積 W*H）
TINY_AREA_RATIO  = 0.0003    # 超小框：< 0.03%
SMALL_AREA_RATIO = 0.0010    # 小框：  < 0.10%
TINY_MIN_SCORE   = 0.35      # 超小框最低分
SMALL_MIN_SCORE  = 0.25      # 小框最低分

# Soft-NMS（可選）
USE_SOFT_NMS = False         # 需要時改 True
SOFT_NMS_METHOD = "gaussian" # "gaussian" 或 "linear"
SOFT_NMS_IOU = 0.55          # 僅 linear 會用到；gaussian 只看 sigma
SOFT_NMS_SIGMA = 0.5         # 高斯σ（越大越寬鬆）
# ==========================

# 專案工具
from config import load_cfg
from modelv7 import build_detector_from_cfg


# ---------- 基礎工具 ----------
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


# ---------- Soft-NMS（單類別） ----------
def soft_nms_single(boxes: torch.Tensor,
                    scores: torch.Tensor,
                    method: str = "gaussian",
                    iou_thresh: float = 0.5,
                    sigma: float = 0.5,
                    topk: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    簡單 Soft-NMS（CPU/Tensor 皆可），回傳抑制後的 boxes/scores（已排序）
    method: "gaussian" 或 "linear"
    """
    if boxes.numel() == 0:
        return boxes, scores
    # 轉 CPU 浮點（演算法是逐步更新分數；CPU 反而比較穩定）
    b = boxes.detach().float().cpu()
    s = scores.detach().float().cpu()
    x1, y1, x2, y2 = b[:,0], b[:,1], b[:,2], b[:,3]
    areas = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)

    order = torch.argsort(s, descending=True)
    keep_boxes = []
    keep_scores = []

    while order.numel() > 0:
        i = order[0]
        bb = b[i]
        ss = s[i].item()
        keep_boxes.append(bb)
        keep_scores.append(ss)

        if topk is not None and len(keep_boxes) >= topk:
            break

        rest = order[1:]
        if rest.numel() == 0:
            break

        xx1 = torch.maximum(bb[0], x1[rest])
        yy1 = torch.maximum(bb[1], y1[rest])
        xx2 = torch.minimum(bb[2], x2[rest])
        yy2 = torch.minimum(bb[3], y2[rest])

        w = (xx2 - xx1).clamp(min=0)
        h = (yy2 - yy1).clamp(min=0)
        inter = w * h
        iou = inter / (areas[i] + areas[rest] - inter + 1e-6)

        if method == "linear":
            weight = torch.ones_like(iou)
            mask = iou > iou_thresh
            weight[mask] = 1.0 - iou[mask]
        else:  # gaussian
            weight = torch.exp(- (iou * iou) / sigma)

        s[rest] = s[rest] * weight
        # 重新排序剩餘的
        order = torch.argsort(s[rest], descending=True)
        rest = rest[order]
        order = rest

    kb = torch.stack(keep_boxes, dim=0).to(boxes.device)
    ks = torch.tensor(keep_scores, device=boxes.device)
    return kb, ks


# ---------- 主流程 ----------
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

            # 過濾 NaN/Inf
            if boxes.numel() > 0:
                finite = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores)
                boxes, scores = boxes[finite], scores[finite]

            # 先以「較寬鬆」的門檻過濾（避免太多垃圾進 NMS）
            pre_keep = scores >= max(0.05, min(0.2, SCORE_THRESH * 0.5))
            boxes, scores = boxes[pre_keep], scores[pre_keep]

            # NMS（Soft 或 Hard）
            if boxes.numel() > 0:
                if USE_SOFT_NMS:
                    boxes, scores = soft_nms_single(
                        boxes, scores,
                        method=SOFT_NMS_METHOD,
                        iou_thresh=float(SOFT_NMS_IOU),
                        sigma=float(SOFT_NMS_SIGMA),
                        topk=None
                    )
                else:
                    keep_idx = hard_nms(boxes, scores, float(NMS_THRESH))
                    boxes, scores = boxes[keep_idx], scores[keep_idx]

            # 依分數排序、截斷（先粗略 Top-K，後面還會面積門檻再細修一次）
            if scores.numel() > 0:
                order = torch.argsort(scores, descending=True)[:max(MAX_DETS_PER_IMG*2, 200)]
                boxes, scores = boxes[order], scores[order]

            # 裁邊界、轉 xywh
            if CLIP_TO_IMAGE and boxes.numel() > 0:
                boxes = _clip_boxes_xyxy(boxes, W, H)
            boxes_xywh = _xyxy_to_xywh(boxes) if boxes.numel() > 0 else torch.empty((0, 4), device=DEVICE)

            # 面積感知門檻（小框更嚴）
            if boxes_xywh.numel() > 0:
                areas = (boxes_xywh[:, 2] * boxes_xywh[:, 3])
                A = float(W * H)
                tiny_mask  = areas < (TINY_AREA_RATIO * A)
                small_mask = (~tiny_mask) & (areas < (SMALL_AREA_RATIO * A))
                large_mask = ~(tiny_mask | small_mask)

                mask = torch.zeros_like(scores, dtype=torch.bool)
                if tiny_mask.any():
                    mask |= (tiny_mask & (scores >= TINY_MIN_SCORE))
                if small_mask.any():
                    mask |= (small_mask & (scores >= SMALL_MIN_SCORE))
                if large_mask.any():
                    mask |= (large_mask & (scores >= float(SCORE_THRESH)))

                boxes_xywh, scores = boxes_xywh[mask], scores[mask]

            # 最終 Top-K 截斷
            if scores.numel() > 0 and scores.numel() > MAX_DETS_PER_IMG:
                order = torch.argsort(scores, descending=True)[:int(MAX_DETS_PER_IMG)]
                boxes_xywh, scores = boxes_xywh[order], scores[order]

            image_id = _image_id_from_path(fp)
            pred_str = _format_row(scores, boxes_xywh)

            # 自檢：6 的倍數、不得 NaN/Inf
            if not _is_valid_pred_string(pred_str):
                print(f"[WARN] invalid PredictionString on image {image_id}; fallback to empty.")
                pred_str = ""

            writer.writerow([image_id, pred_str])

    print(f"[Done] wrote CSV -> {OUT_CSV}")


if __name__ == "__main__":
    main()
