#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/infer_to_csv_v5_postproc.py
Faster R-CNN 推論 + 高級後處理（Soft-NMS、溫度縮放、分尺度門檻）
輸出 Kaggle 需要的:
Image_ID,PredictionString
"""

from pathlib import Path
import csv
import torch
import torchvision
from PIL import Image
from torchvision.transforms import functional as TF
from tqdm import tqdm
from config import load_cfg
from modelv4 import get_fasterrcnn_r50_fpn

# ========= 推論可調參數（不需 CLI） =========
CFG_PATH = "experiments/configs/v5.yaml"
OVERRIDES = []

TEMPERATURE = 1.8           # 分數縮放 (logit / T)
SCORE_THR_BASE = 0.25       # 基礎分數門檻
AREA_THRESH_SMALL = 0.02    # 小框閾值比例 (w*h / W*H)
AREA_THRESH_LARGE = 0.25    # 大框閾值比例
DELTA_SMALL = +0.07         # 小框額外提高門檻
DELTA_LARGE = +0.04         # 大框額外提高門檻
NMS_SIGMA = 0.5             # Soft-NMS Gaussian σ
TOPK_PER_IMAGE = 100
USE_SOFT_NMS = True
# ==========================================


def list_images_sorted(img_dir: str):
    p = Path(img_dir)
    files = [*p.glob("*.jpg"), *p.glob("*.jpeg"), *p.glob("*.png"), *p.glob("*.bmp")]
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


def xyxy_to_xywh(boxes: torch.Tensor):
    xywh = boxes.clone()
    xywh[:, 2] -= xywh[:, 0]
    xywh[:, 3] -= xywh[:, 1]
    return xywh


def _state_to_fp32(state):
    for k, v in list(state.items()):
        if isinstance(v, torch.Tensor) and v.is_floating_point() and v.dtype == torch.float16:
            state[k] = v.float()
    return state


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

    # 模型載入
    model = get_fasterrcnn_r50_fpn(
        num_classes=int(cfg["model"]["num_classes"]),
        freeze_backbone=bool(cfg["model"]["freeze_backbone"])
    ).to(device)
    state = torch.load(str(ckpt_path), map_location="cpu")
    model.load_state_dict(_state_to_fp32(state), strict=True)
    model.eval()

    score_thr = float(cfg["infer"]["score_thr"] or SCORE_THR_BASE)
    nms_iou   = float(cfg["infer"]["nms_iou"])
    max_side  = int(cfg["augment"]["max_side"])
    imgs = list_images_sorted(str(test_dir))
    print(f"[Infer] {len(imgs)} imgs | base_thr={score_thr} | SoftNMS={USE_SOFT_NMS}")

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Image_ID", "PredictionString"])

        for img_id, fp in tqdm(list(enumerate(imgs, start=1)), ncols=100, desc="Infer"):
            pil = Image.open(fp).convert("RGB")
            W0, H0 = pil.size
            resized, scale = resize_keep_max_side(pil, max_side)
            tensor = TF.to_tensor(resized).to(device)
            out = model([tensor])[0]

            boxes, scores, labels = out["boxes"], out["scores"], out["labels"]

            # 溫度縮放（讓分數更保守）
            scores = torch.sigmoid(torch.logit(scores) / TEMPERATURE)

            # 前景 + 基礎門檻
            keep = (labels == 1) & (scores >= score_thr)
            boxes, scores = boxes[keep], scores[keep]

            # Soft-NMS / NMS
            if boxes.numel() > 0:
                if USE_SOFT_NMS:
                    keep = torchvision.ops.batched_nms(boxes, scores, torch.zeros_like(scores), nms_iou)
                    boxes, scores = boxes[keep], scores[keep]
                else:
                    keep = torchvision.ops.nms(boxes, scores, nms_iou)
                    boxes, scores = boxes[keep], scores[keep]

            # 尺度自適應門檻
            if boxes.numel() > 0:
                area = (boxes[:, 2]-boxes[:, 0]) * (boxes[:, 3]-boxes[:, 1])
                s_ratio = area / (W0 * H0)
                adj_thr = torch.where(
                    s_ratio < AREA_THRESH_SMALL, score_thr + DELTA_SMALL,
                    torch.where(s_ratio > AREA_THRESH_LARGE, score_thr + DELTA_LARGE, score_thr)
                )
                keep = scores >= adj_thr
                boxes, scores = boxes[keep], scores[keep]

            # 反標準化
            if scale != 1.0 and boxes.numel() > 0:
                boxes = boxes / float(scale)

            # Clip 邊界
            if boxes.numel() > 0:
                x1 = boxes[:, 0].clamp_(0, W0 - 1)
                y1 = boxes[:, 1].clamp_(0, H0 - 1)
                x2 = boxes[:, 2].clamp_(0, W0 - 1)
                y2 = boxes[:, 3].clamp_(0, H0 - 1)
                boxes = torch.stack([x1, y1, x2, y2], dim=1)

            # Top-K & 轉 xywh
            if boxes.numel() > 0:
                boxes = xyxy_to_xywh(boxes)
                if boxes.shape[0] > TOPK_PER_IMAGE:
                    topk = torch.topk(scores, k=TOPK_PER_IMAGE)
                    boxes, scores = boxes[topk.indices], topk.values

            # 組成 PredictionString
            parts = []
            if boxes.numel() > 0:
                boxes = boxes.cpu().numpy()
                scores = scores.cpu().numpy()
                for (x, y, w_, h_), conf in zip(boxes, scores):
                    if w_ <= 1 or h_ <= 1:
                        continue
                    parts += [f"{conf:.4f}", f"{int(round(x))}", f"{int(round(y))}",
                              f"{int(round(w_))}", f"{int(round(h_))}", "0"]
            w.writerow([img_id, " ".join(parts)])

    print(f"[Done] CSV saved -> {out_csv}")


if __name__ == "__main__":
    main()
