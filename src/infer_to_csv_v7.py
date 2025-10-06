#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/infer_to_csv_v7.py  ·  Kaggle 提交（RetinaNet/泛用）穩妥版
- 與訓練同一份 YAML（含 detector/anchors/尺寸/NMS/Tiling）
- Image_ID: 使用 enumerate 從 1 開始的整數（沿用你 0.17033 那版做法）
- PredictionString: 一律輸出 6 欄位倍數 => "conf x y w h class"（class 固定 0）
"""

from pathlib import Path
import csv
from typing import List, Tuple

import torch
import torchvision
from torchvision.ops import nms
from torchvision.transforms import functional as TF
from PIL import Image
from tqdm import tqdm

from config import load_cfg
from modelv7 import build_detector_from_cfg  # ← v7：可切換架構的工廠

# ========= 覆寫設定（可留空） =========
CFG_PATH = "experiments/configs/v7.yaml"
OVERRIDES = [
    # 指定要用的 checkpoint；若留空會用 YAML 的 checkpoint 配置
    # "checkpoint.save_full_path=experiments/logs/retinanet_v7/retinanet_v7_best.pth",
    # "project.run_name=submit_v7",
]
# ====================================


def list_images_sorted(img_dir: str) -> list[Path]:
    p = Path(img_dir)
    files = [*p.glob("*.jpg"), *p.glob("*.jpeg"), *p.glob("*.png"), *p.glob("*.bmp"),
             *p.glob("*.JPG"), *p.glob("*.JPEG"), *p.glob("*.PNG"), *p.glob("*.BMP")]
    def _key(fp: Path):
        stem = fp.stem.lstrip("0")
        return (0, int(stem)) if stem.isdigit() else (1, fp.stem)
    return sorted(files, key=_key)


def resize_keep_max_side(img: Image.Image, max_side: int) -> Tuple[Image.Image, float]:
    """把影像縮到最長邊=max_side，回傳 (image, scale)。"""
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


def run_nms_like_cfg(boxes: torch.Tensor, scores: torch.Tensor, cfg: dict) -> Tuple[torch.Tensor, torch.Tensor]:
    """依 YAML 的 infer/postproc 執行 Soft-NMS 或 Hard-NMS。"""
    score_thr = float(cfg["infer"]["score_thr"])
    nms_iou   = float(cfg["infer"]["nms_iou"])
    max_det   = int(cfg.get("infer", {}).get("postproc", {}).get("topk_per_image",
                      int(cfg.get("eval", {}).get("max_det", 100))))
    pp = cfg.get("infer", {}).get("postproc", {})
    soft_cfg = pp.get("soft_nms", {})
    use_soft = bool(soft_cfg.get("enabled", False))
    soft_method = str(soft_cfg.get("method", "gaussian"))
    soft_sigma  = float(soft_cfg.get("sigma", 0.5))
    soft_iou    = float(soft_cfg.get("iou_thresh", nms_iou))
    soft_score_floor = float(soft_cfg.get("score_thresh", 0.0))

    # 初步門檻
    if boxes.numel() > 0:
        k0 = scores >= score_thr
        boxes, scores = boxes[k0], scores[k0]

    # NMS
    if boxes.numel() > 0:
        if use_soft:
            if soft_method == "gaussian":
                keep_idx = soft_nms_gaussian(boxes, scores, iou_thresh=soft_iou, sigma=soft_sigma, score_thresh=soft_score_floor)
            else:
                keep_idx = soft_nms_linear_or_hard(boxes, scores, iou_thresh=soft_iou, method=soft_method, score_thresh=soft_score_floor)
        else:
            keep_idx = nms(boxes, scores, nms_iou).cpu().tolist()
        boxes, scores = boxes[keep_idx], scores[keep_idx]

    # Top-K
    if boxes.numel() > 0 and boxes.shape[0] > max_det:
        vals, idxs = torch.topk(scores, k=max_det)
        boxes, scores = boxes[idxs], vals

    return boxes, scores


def area_aware_threshold(boxes: torch.Tensor, scores: torch.Tensor, cfg: dict) -> Tuple[torch.Tensor, torch.Tensor]:
    """按面積再做一層分數門檻（原圖座標）。"""
    pp = cfg.get("infer", {}).get("postproc", {})
    area_cfg = pp.get("area_aware_score", {})
    if not bool(area_cfg.get("enabled", False)) or boxes.numel() == 0:
        return boxes, scores

    small_thr  = float(area_cfg.get("small_thr", 0.05))
    medium_thr = float(area_cfg.get("medium_thr", 0.05))
    large_thr  = float(area_cfg.get("large_thr", 0.05))
    small_area = float(area_cfg.get("small_area", 32**2))
    large_area = float(area_cfg.get("large_area", 96**2))

    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    thr_vec = torch.full_like(areas, medium_thr, dtype=torch.float32)
    thr_vec[areas <  small_area] = small_thr
    thr_vec[areas >= large_area] = large_thr
    k = scores >= thr_vec
    return boxes[k], scores[k]


def infer_single_image_no_tiling(model, pil: Image.Image, device, cfg: dict) -> Tuple[torch.Tensor, torch.Tensor]:
    """舊路徑：不切塊，維持原有縮放→NMS→面積門檻流程。"""
    W0, H0 = pil.size
    max_side = int(cfg["augment"]["max_side"])
    resized, scale = resize_keep_max_side(pil, max_side)
    tensor = TF.to_tensor(resized).to(device)

    out = model([tensor])[0]
    boxes, scores, labels = out["boxes"], out["scores"], out["labels"]
    keep = (labels == 1)
    boxes, scores = boxes[keep], scores[keep]

    # 還原座標
    if boxes.numel() > 0 and scale != 1.0:
        boxes = boxes / float(scale)
    if boxes.numel() > 0:
        x1 = boxes[:, 0].clamp_(0, W0 - 1)
        y1 = boxes[:, 1].clamp_(0, H0 - 1)
        x2 = boxes[:, 2].clamp_(0, W0 - 1)
        y2 = boxes[:, 3].clamp_(0, H0 - 1)
        boxes = torch.stack([x1, y1, x2, y2], dim=1)

    # NMS + 面積門檻 + TopK
    boxes, scores = run_nms_like_cfg(boxes, scores, cfg)
    boxes, scores = area_aware_threshold(boxes, scores, cfg)
    return boxes, scores


def make_grid_tiles(W: int, H: int, tile: int, overlap: float) -> List[Tuple[int, int, int, int]]:
    """回傳一組 tile 區塊 (x0,y0,x1,y1) 覆蓋整張圖。"""
    stride = max(1, int(round(tile * (1.0 - overlap))))
    xs = list(range(0, max(1, W - tile + 1), stride))
    ys = list(range(0, max(1, H - tile + 1), stride))
    if len(xs) == 0: xs = [0]
    if len(ys) == 0: ys = [0]
    if xs[-1] + tile < W: xs.append(W - tile)
    if ys[-1] + tile < H: ys.append(H - tile)

    boxes = []
    for y in ys:
        for x in xs:
            x0, y0 = max(0, x), max(0, y)
            x1, y1 = min(W, x0 + tile), min(H, y0 + tile)
            boxes.append((x0, y0, x1, y1))
    return boxes


def infer_single_image_tiling(model, pil: Image.Image, device, cfg: dict) -> Tuple[torch.Tensor, torch.Tensor]:
    """切塊推論：每塊縮到 max_side，推論後還原到原圖，全域再做 NMS/面積門檻。"""
    W0, H0 = pil.size
    tcfg = cfg.get("infer", {}).get("tiling", {})
    tile_size = int(tcfg.get("tile_size", 1024))
    overlap   = float(tcfg.get("overlap", 0.15))
    max_side  = int(cfg["augment"]["max_side"])  # 每塊推論時也縮到這個，避免 model.transform 再縮放

    all_boxes, all_scores = [], []

    for (x0, y0, x1, y1) in make_grid_tiles(W0, H0, tile_size, overlap):
        crop = pil.crop((x0, y0, x1, y1))
        resized, scale = resize_keep_max_side(crop, max_side)
        tensor = TF.to_tensor(resized).to(device)

        out = model([tensor])[0]
        boxes, scores, labels = out["boxes"], out["scores"], out["labels"]
        keep = (labels == 1)
        boxes, scores = boxes[keep], scores[keep]

        # 還原到「原圖」座標
        if boxes.numel() > 0:
            if scale != 1.0:
                boxes = boxes / float(scale)
            boxes[:, [0, 2]] += float(x0)
            boxes[:, [1, 3]] += float(y0)
            all_boxes.append(boxes)
            all_scores.append(scores)

    if len(all_boxes) == 0:
        return torch.zeros((0, 4), dtype=torch.float32), torch.zeros((0,), dtype=torch.float32)

    boxes = torch.cat(all_boxes, dim=0)
    scores = torch.cat(all_scores, dim=0)

    # 全域 NMS + 面積門檻 + TopK（依 YAML）
    boxes, scores = run_nms_like_cfg(boxes, scores, cfg)
    boxes, scores = area_aware_threshold(boxes, scores, cfg)
    return boxes, scores


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

    # ---- 模型（與訓練同 cfg；自動依 YAML 選 detector）----
    model = build_detector_from_cfg(cfg).to(device)
    try:
        state = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(str(ckpt_path), map_location="cpu")
    model.load_state_dict(_state_to_fp32(state), strict=True)
    model.eval()

    # ---- 推論設定 ----
    max_side  = int(cfg["augment"]["max_side"])
    max_det   = int(cfg.get("infer", {}).get("postproc", {}).get("topk_per_image",
                      int(cfg.get("eval", {}).get("max_det", 100))))
    tcfg = cfg.get("infer", {}).get("tiling", {})
    use_tiling = bool(tcfg.get("enabled", False))

    imgs = list_images_sorted(str(test_dir))
    assert len(imgs) > 0, "測試資料夾沒有影像"

    print(f"[Infer] images={len(imgs)} tiling={use_tiling} max_side={max_side} max_det={max_det}")
    print(f"[CKPT ] {ckpt_path}")
    print(f"[OUT  ] {out_csv}")

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Image_ID", "PredictionString"])

        for img_id, fp in tqdm(list(enumerate(imgs, start=1)), ncols=100, desc="Infer"):
            pil = Image.open(fp).convert("RGB")

            if use_tiling:
                boxes, scores = infer_single_image_tiling(model, pil, device, cfg)
            else:
                boxes, scores = infer_single_image_no_tiling(model, pil, device, cfg)

            # 轉 xywh、組 PredictionString（一定輸出 6*n 欄；class 固定 0）
            parts: List[str] = []
            if boxes.numel() > 0:
                boxes = xyxy_to_xywh(boxes)
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
