#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/infer_to_csv.py  ·  Kaggle 提交保守穩妥版
- Image_ID: 使用 enumerate 從 1 開始的整數（和你 0.17033 的那版一致）
- PredictionString: 一律輸出 6 欄位的倍數 => "conf x y w h class"（class 固定 0）
"""

from pathlib import Path
import csv
from typing import List

import torch
import torchvision
from torchvision.ops import nms
from torchvision.transforms import functional as TF
from PIL import Image
from tqdm import tqdm

from config import load_cfg
from modelv6 import get_fasterrcnn_r50_fpn

# ========= 覆寫設定（可留空） =========
CFG_PATH = "experiments/configs/v6.yaml"
OVERRIDES = [
    "checkpoint.save_full_path=experiments/logs/fasterrcnn_v6/fasterrcnn_v6_best.pth",
    # "project.run_name=submit_v6",
]
# ====================================

def list_images_sorted(img_dir: str):
    p = Path(img_dir)
    files = [*p.glob("*.jpg"), *p.glob("*.jpeg"), *p.glob("*.png"), *p.glob("*.bmp"),
             *p.glob("*.JPG"), *p.glob("*.JPEG"), *p.glob("*.PNG"), *p.glob("*.BMP")]
    def _key(fp: Path):
        stem = fp.stem.lstrip("0")
        return (0, int(stem)) if stem.isdigit() else (1, fp.stem)
    return sorted(files, key=_key)

def resize_keep_max_side(img: Image.Image, max_side: int):
    w, h = img.size
    m = max(w, h)
    if m <= max_side:
        return img, 1.0
    s = float(max_side) / m
    new_w, new_h = int(round(w * s)), int(round(h * s))
    return img.resize((new_w, new_h), Image.BILINEAR), s

def xyxy_to_xywh(boxes: torch.Tensor) -> torch.Tensor:
    xywh = boxes.clone()
    xywh[:, 2] = xywh[:, 2] - xywh[:, 0]
    xywh[:, 3] = xywh[:, 3] - xywh[:, 1]
    return xywh

def _state_to_fp32(state):
    for k, v in list(state.items()):
        if isinstance(v, torch.Tensor) and v.is_floating_point() and v.dtype == torch.float16:
            state[k] = v.float()
    return state

def soft_nms_gaussian(boxes, scores, iou_thresh, sigma, score_thresh) -> List[int]:
    boxes = boxes.clone().cpu()
    scores = scores.clone().cpu()
    keep_scores = scores.clone()
    order = scores.argsort(descending=True).tolist()
    kept = []
    while order:
        i = order.pop(0)
        kept.append(i)
        if not order:
            break
        ious = torchvision.ops.box_iou(boxes[i].unsqueeze(0), boxes[order]).squeeze(0)
        decay = torch.exp(-(ious ** 2) / sigma)
        keep_scores[order] = keep_scores[order] * decay
        order = [j for j in order if keep_scores[j] >= score_thresh]
        order.sort(key=lambda j: float(keep_scores[j]), reverse=True)
    kept.sort(key=lambda j: float(keep_scores[j]), reverse=True)
    return kept

def soft_nms_linear_or_hard(boxes, scores, iou_thresh, method, score_thresh) -> List[int]:
    if method == "hard":
        return nms(boxes, scores, iou_thresh).cpu().tolist()
    boxes = boxes.clone().cpu()
    scores = scores.clone().cpu()
    keep_scores = scores.clone()
    order = scores.argsort(descending=True).tolist()
    kept = []
    while order:
        i = order.pop(0)
        kept.append(i)
        if not order:
            break
        ious = torchvision.ops.box_iou(boxes[i].unsqueeze(0), boxes[order]).squeeze(0)
        decay = torch.ones_like(ious)
        mask = ious > iou_thresh
        decay[mask] = 1 - ious[mask]
        keep_scores[order] = keep_scores[order] * decay
        order = [j for j in order if keep_scores[j] >= score_thresh]
        order.sort(key=lambda j: float(keep_scores[j]), reverse=True)
    kept.sort(key=lambda j: float(keep_scores[j]), reverse=True)
    return kept

@torch.inference_mode()
def main():
    project_root = Path(__file__).resolve().parents[1]
    cfg = load_cfg(str(project_root / CFG_PATH), overrides=OVERRIDES)

    ckpt_cfg = cfg["checkpoint"]
    ckpt_path = Path(ckpt_cfg.get("save_full_path") or (Path(ckpt_cfg["dir"]) / ckpt_cfg["name"]))
    test_dir = Path(cfg["data"]["test_img_dir"])
    out_csv = Path(cfg["infer"]["submission_csv"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() and cfg["device"]["cuda"] else "cpu")
    assert ckpt_path.exists(), f"找不到權重檔：{ckpt_path}"
    assert test_dir.exists(), "找不到測試影像資料夾"

    # 模型（與訓練同 cfg）
    model = get_fasterrcnn_r50_fpn(
        num_classes=int(cfg["model"]["num_classes"]),
        freeze_backbone=bool(cfg["model"]["freeze_backbone"]),
        pretrained_backbone=bool(cfg["model"].get("pretrained_backbone", False)),
        cfg=cfg,
    ).to(device)
    try:
        state = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(str(ckpt_path), map_location="cpu")
    model.load_state_dict(_state_to_fp32(state), strict=True)
    model.eval()

    # 推論設定
    score_thr = float(cfg["infer"]["score_thr"])
    nms_iou   = float(cfg["infer"]["nms_iou"])
    max_side  = int(cfg["augment"]["max_side"])
    max_det   = int(cfg.get("infer", {}).get("postproc", {}).get("topk_per_image",
                      int(cfg.get("eval", {}).get("max_det", 100))))
    # 後處理
    pp = cfg.get("infer", {}).get("postproc", {})
    soft_cfg = pp.get("soft_nms", {})
    use_soft = bool(soft_cfg.get("enabled", False))
    soft_method = str(soft_cfg.get("method", "gaussian"))
    soft_sigma  = float(soft_cfg.get("sigma", 0.5))
    soft_iou    = float(soft_cfg.get("iou_thresh", nms_iou))
    soft_score_floor = float(soft_cfg.get("score_thresh", 0.0))

    area_cfg = pp.get("area_aware_score", {})
    area_aware = bool(area_cfg.get("enabled", False))
    small_thr  = float(area_cfg.get("small_thr", score_thr))
    medium_thr = float(area_cfg.get("medium_thr", score_thr))
    large_thr  = float(area_cfg.get("large_thr", score_thr))
    small_area = float(area_cfg.get("small_area", 32**2))
    large_area = float(area_cfg.get("large_area", 96**2))

    imgs = list_images_sorted(str(test_dir))
    assert len(imgs) > 0, "測試資料夾沒有影像"

    print(f"[Infer] images={len(imgs)} thr={score_thr} nms={nms_iou} max_side={max_side} max_det={max_det}")
    print(f"[CKPT ] {ckpt_path}")
    print(f"[OUT  ] {out_csv}")

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Image_ID", "PredictionString"])

        for img_id, fp in tqdm(list(enumerate(imgs, start=1)), ncols=100, desc="Infer"):
            pil = Image.open(fp).convert("RGB")
            W0, H0 = pil.size
            resized, scale = resize_keep_max_side(pil, max_side)
            tensor = TF.to_tensor(resized).to(device)

            out = model([tensor])[0]
            boxes  = out["boxes"]
            scores = out["scores"]
            labels = out["labels"]

            # 只前景
            keep = (labels == 1)
            boxes, scores = boxes[keep], scores[keep]

            # 初步閾值
            if boxes.numel() > 0:
                k0 = scores >= score_thr
                boxes, scores = boxes[k0], scores[k0]

            # (Soft-)NMS
            if boxes.numel() > 0:
                if use_soft:
                    if soft_method == "gaussian":
                        keep_idx = soft_nms_gaussian(boxes, scores, iou_thresh=soft_iou, sigma=soft_sigma, score_thresh=soft_score_floor)
                    else:
                        keep_idx = soft_nms_linear_or_hard(boxes, scores, iou_thresh=soft_iou, method=soft_method, score_thresh=soft_score_floor)
                else:
                    keep_idx = nms(boxes, scores, nms_iou).cpu().tolist()
                boxes, scores = boxes[keep_idx], scores[keep_idx]

            # 回到原圖、clip
            if boxes.numel() > 0 and scale != 1.0:
                boxes = boxes / float(scale)
            if boxes.numel() > 0:
                x1 = boxes[:, 0].clamp_(0, W0 - 1)
                y1 = boxes[:, 1].clamp_(0, H0 - 1)
                x2 = boxes[:, 2].clamp_(0, W0 - 1)
                y2 = boxes[:, 3].clamp_(0, H0 - 1)
                boxes = torch.stack([x1, y1, x2, y2], dim=1)

            # 面積感知第二道閾值（原圖空間）
            if boxes.numel() > 0 and area_aware:
                areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                thr_vec = torch.full_like(areas, medium_thr, dtype=torch.float32)
                thr_vec[areas <  small_area] = small_thr
                thr_vec[areas >= large_area] = large_thr
                k_area = scores >= thr_vec
                boxes, scores = boxes[k_area], scores[k_area]

            # 轉 xywh、Top-K、組 PredictionString（一定輸出 6*n 欄）
            parts = []
            if boxes.numel() > 0:
                boxes = xyxy_to_xywh(boxes)
                if boxes.shape[0] > max_det:
                    vals, idxs = torch.topk(scores, k=max_det)
                    boxes, scores = boxes[idxs], vals
                boxes_np = boxes.cpu().numpy()
                scores_np = scores.cpu().numpy()
                for (x, y, w_, h_), conf in zip(boxes_np, scores_np):
                    if w_ <= 0 or h_ <= 0:
                        continue
                    parts += [f"{conf:.4f}", f"{int(round(x))}", f"{int(round(y))}",
                              f"{int(round(w_))}", f"{int(round(h_))}", "0"]  # class 固定 0

            predstr = " ".join(parts)  # 無檢出 -> 空字串
            w.writerow([img_id, predstr])

    print(f"[Done] CSV saved -> {out_csv}")

if __name__ == "__main__":
    main()
