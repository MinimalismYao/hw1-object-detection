#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/infer_to_csv_v5_final.py
Faster R-CNN 推論（v5） + 後處理強化：
- Temperature Scaling（分數重校準）
- 面積感知門檻（small/large 加嚴）
- Soft-NMS（Gaussian，可切換 Hard-NMS）
- Top-K per image
- 反標準化、邊界裁切、去除不合法框
輸出 Kaggle 需要的 CSV：
Image_ID,PredictionString
"""

from pathlib import Path
import csv
import math
import torch
import torchvision
from PIL import Image
from torchvision.transforms import functional as TF
from torchvision.ops import nms as hard_nms
from tqdm import tqdm

from config import load_cfg
from modelv4 import get_fasterrcnn_r50_fpn


# ========= 可在這裡快速調整的區域（不需 CLI） =========
CFG_PATH = "experiments/configs/v5.yaml"
OVERRIDES = [
    # 覆寫 YAML 也能在此設定（例）：
    # "checkpoint.save_full_path=experiments/logs/fasterrcnn_v5/fasterrcnn_v5_best.pth",
    # "infer.submission_csv=submissions/fasterrcnn_v5_submission.csv",
    "infer.score_thr=0.30",   # 基礎分數門檻（由 0.03 → 0.30 以抑制低質量框）
    "infer.nms_iou=0.55",     # NMS IoU（較嚴以壓重疊）
    "eval.max_det=60",        # 每張圖最多保留幾個框（Top-K）
]

# 後處理強化參數
TEMPERATURE = 2.0            # 分數重校準：logit / T 後再 sigmoid
USE_SOFT_NMS = True          # True=Soft-NMS；False=Hard-NMS
SOFT_NMS_SIGMA = 0.30        # Soft-NMS Gaussian σ（越小抑制越強）
SOFT_NMS_MIN_SCORE = 1e-4    # Soft-NMS 內部最小分數（落到此值會丟棄）

# 面積感知門檻（在「resized 空間」計算面積比，與縮放無關）
AREA_THRESH_SMALL = 0.02     # (w*h)/(W*H) < 0.02 視為小框
AREA_THRESH_LARGE = 0.25     # (w*h)/(W*H) > 0.25 視為大框
DELTA_SMALL = 0.10           # 小框在基礎門檻上再 +0.10
DELTA_LARGE = 0.05           # 大框在基礎門檻上再 +0.05

TOPK_PER_IMAGE = 60          # 每張圖最終最多輸出框數
MIN_BOX_WH = 2               # 最小合法寬高（像素），<2 視為無效
# =====================================================


def list_images_sorted(img_dir: str):
    p = Path(img_dir)
    files = [*p.glob("*.jpg"), *p.glob("*.jpeg"), *p.glob("*.png"), *p.glob("*.bmp")]
    def _key(fp: Path):
        stem = fp.stem.lstrip("0")
        return (0, int(stem)) if stem.isdigit() else (1, fp.stem)
    return sorted(files, key=_key)


def resize_keep_max_side(img: Image.Image, max_side: int):
    """回傳 (resized_img, scale)。scale = resized / original"""
    w, h = img.size
    m = max(w, h)
    if m <= max_side:
        return img, 1.0
    s = float(max_side) / m
    new_w, new_h = int(round(w * s)), int(round(h * s))
    return img.resize((new_w, new_h), Image.BILINEAR), s


def xyxy_to_xywh(boxes: torch.Tensor) -> torch.Tensor:
    xywh = boxes.clone()
    xywh[:, 2] -= xywh[:, 0]
    xywh[:, 3] -= xywh[:, 1]
    return xywh


def _state_to_fp32(state):
    for k, v in list(state.items()):
        if isinstance(v, torch.Tensor) and v.is_floating_point() and v.dtype == torch.float16:
            state[k] = v.float()
    return state


def _temperature_scale(scores: torch.Tensor, T: float) -> torch.Tensor:
    # scores 為 [0,1] 之間的置信度；先轉回 logit 再除以 T，最後 sigmoid
    # 避免數值問題：限制 scores 在 (eps, 1-eps)
    eps = 1e-6
    s = scores.clamp(min=eps, max=1 - eps)
    logit = torch.log(s / (1 - s))
    return torch.sigmoid(logit / T)


def _soft_nms_gaussian(boxes: torch.Tensor,
                       scores: torch.Tensor,
                       iou_thr: float = 0.5,
                       sigma: float = 0.5,
                       min_score: float = 1e-4) -> torch.Tensor:
    """
    Soft-NMS (Gaussian) — 回傳保留索引。
    參考：Soft-NMS (ICCV 2017) — 以簡潔 PyTorch 實作。
    注意：此版本是 O(N^2)；用於每張圖 N<=數百 框時足夠。
    """
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)

    x1, y1, x2, y2 = boxes.unbind(1)
    areas = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)

    order = scores.argsort(descending=True)
    keep = []

    while order.numel() > 0:
        i = order[0]
        keep.append(i.item())

        if order.numel() == 1:
            break

        rest = order[1:]
        xx1 = torch.maximum(x1[i], x1[rest])
        yy1 = torch.maximum(y1[i], y1[rest])
        xx2 = torch.minimum(x2[i], x2[rest])
        yy2 = torch.minimum(y2[i], y2[rest])

        w = (xx2 - xx1).clamp(min=0)
        h = (yy2 - yy1).clamp(min=0)
        inter = w * h
        iou = inter / (areas[i] + areas[rest] - inter + 1e-6)

        # Gaussian 下降：score_j *= exp(-(iou^2)/sigma)
        decay = torch.exp(-(iou * iou) / (sigma + 1e-6))
        new_scores = scores[rest] * decay
        # 低於門檻的直接丟棄
        mask = new_scores >= min_score

        # 更新分數並重新排序
        scores = scores.clone()
        scores[rest] = new_scores
        order = torch.cat([rest[mask], rest[~mask]*0])  # 保持長度，先挑 mask==True 的
        # 只保留分數仍 >= min_score 的元素並依分數排序
        order = rest[mask][scores[rest[mask]].argsort(descending=True)]

    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


@torch.inference_mode()
def main():
    # 讀設定
    project_root = Path(__file__).resolve().parents[1]
    cfg = load_cfg(str(project_root / CFG_PATH), overrides=OVERRIDES)

    ckpt_cfg = cfg["checkpoint"]
    ckpt_path = Path(ckpt_cfg.get("save_full_path") or (Path(ckpt_cfg["dir"]) / ckpt_cfg["name"]))
    test_dir = Path(cfg["data"]["test_img_dir"])
    out_csv = Path(cfg["infer"]["submission_csv"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    assert ckpt_path.exists(), f"找不到權重檔：{ckpt_path}"
    assert test_dir.exists(), "找不到測試影像資料夾"

    device = torch.device("cuda" if torch.cuda.is_available() and cfg["device"]["cuda"] else "cpu")

    # 建模
    model = get_fasterrcnn_r50_fpn(
        num_classes=int(cfg["model"]["num_classes"]),
        freeze_backbone=bool(cfg["model"]["freeze_backbone"])
    ).to(device)
    # 安全載入
    try:
        state = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(str(ckpt_path), map_location="cpu")
    model.load_state_dict(_state_to_fp32(state), strict=True)
    model.eval()

    # 推論參數
    score_thr = float(cfg["infer"]["score_thr"])
    nms_iou   = float(cfg["infer"]["nms_iou"])
    max_side  = int(cfg["augment"]["max_side"])
    max_det   = int(cfg.get("eval", {}).get("max_det", TOPK_PER_IMAGE))

    imgs = list_images_sorted(str(test_dir))
    assert len(imgs) > 0, "測試資料夾沒有影像"

    print(f"[Infer] images={len(imgs)} | base_thr={score_thr} | NMS={('Soft' if USE_SOFT_NMS else 'Hard')} "
          f"| nms_iou={nms_iou} | T={TEMPERATURE} | topk={max_det}")
    print(f"[CKPT ] {ckpt_path}")
    print(f"[OUT  ] {out_csv}")

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Image_ID", "PredictionString"])

        for img_id, fp in tqdm(list(enumerate(imgs, start=1)), ncols=100, desc="Infer"):
            pil = Image.open(fp).convert("RGB")
            W0, H0 = pil.size
            resized, scale = resize_keep_max_side(pil, max_side)
            Wr, Hr = resized.size

            tensor = TF.to_tensor(resized).to(device)
            out = model([tensor])[0]

            boxes  = out["boxes"]      # xyxy on resized space
            scores = out["scores"]
            labels = out["labels"]

            if boxes.numel() == 0:
                w.writerow([img_id, ""])
                continue

            # 1) 溫度縮放（讓分數更保守）
            scores = _temperature_scale(scores, TEMPERATURE)

            # 2) 只保留前景類 + 基礎門檻
            keep = (labels == 1) & (scores >= score_thr)
            boxes, scores = boxes[keep], scores[keep]

            if boxes.numel() == 0:
                w.writerow([img_id, ""])
                continue

            # 3) NMS（Soft 或 Hard）
            if USE_SOFT_NMS:
                keep_idx = _soft_nms_gaussian(
                    boxes, scores, iou_thr=nms_iou, sigma=SOFT_NMS_SIGMA, min_score=SOFT_NMS_MIN_SCORE
                )
            else:
                keep_idx = hard_nms(boxes, scores, nms_iou)
            boxes, scores = boxes[keep_idx], scores[keep_idx]

            if boxes.numel() == 0:
                w.writerow([img_id, ""])
                continue

            # 4) 面積感知門檻（在 resized 空間計算面積比）
            area = (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
            s_ratio = area / float(Wr * Hr)
            adj_thr = torch.where(
                s_ratio < AREA_THRESH_SMALL, score_thr + DELTA_SMALL,
                torch.where(s_ratio > AREA_THRESH_LARGE, score_thr + DELTA_LARGE, score_thr)
            )
            keep = scores >= adj_thr
            boxes, scores = boxes[keep], scores[keep]

            if boxes.numel() == 0:
                w.writerow([img_id, ""])
                continue

            # 5) 反標準化回原圖座標
            if scale != 1.0:
                boxes = boxes / float(scale)

            # 邊界裁切 + 去除過小框
            x1 = boxes[:, 0].clamp_(0, W0 - 1)
            y1 = boxes[:, 1].clamp_(0, H0 - 1)
            x2 = boxes[:, 2].clamp_(0, W0 - 1)
            y2 = boxes[:, 3].clamp_(0, H0 - 1)
            boxes = torch.stack([x1, y1, x2, y2], dim=1)

            boxes_xywh = xyxy_to_xywh(boxes)
            bw, bh = boxes_xywh[:, 2], boxes_xywh[:, 3]
            keep = (bw > MIN_BOX_WH) & (bh > MIN_BOX_WH)
            boxes_xywh, scores = boxes_xywh[keep], scores[keep]

            if boxes_xywh.numel() == 0:
                w.writerow([img_id, ""])
                continue

            # 6) Top-K（每張圖最多保留）
            if boxes_xywh.shape[0] > max_det:
                topk = torch.topk(scores, k=max_det)
                boxes_xywh, scores = boxes_xywh[topk.indices], topk.values

            # 7) 組 PredictionString（class 固定 0；座標轉整數像素）
            parts = []
            b_np = boxes_xywh.cpu().numpy()
            s_np = scores.cpu().numpy()
            for (x, y, w_, h_), conf in zip(b_np, s_np):
                parts += [f"{conf:.4f}", f"{int(round(x))}", f"{int(round(y))}",
                          f"{int(round(w_))}", f"{int(round(h_))}", "0"]

            w.writerow([img_id, " ".join(parts)])

    print(f"[Done] CSV saved -> {out_csv}")


if __name__ == "__main__":
    main()
