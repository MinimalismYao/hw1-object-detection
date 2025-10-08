#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/vis_pred_v7.py
從指定資料夾讀取影像 → 用 v7 模型推論 → 畫出半透明淡色的預測框（可選 GT）→ 存檔。

使用方式（專案根目錄）：
  python src/vis_pred_v7.py

只需修改檔頭的設定區即可。
"""

import os, glob, math
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torchvision
from torchvision.ops import nms as hard_nms

# 專案內模組
from config import load_cfg
from modelv7 import build_detector_from_cfg

# ========== 可自行修改的設定（v7 專用） ==========
CFG_PATH    = "experiments/configs/v7.yaml"     # v7 的 YAML
CKPT_PATH   = "experiments/logs/fasterrcnn_v7/fasterrcnn_v7_best.pth"  # 權重（建議用 best）
IMG_DIR     = "data/mini_test"                  # 要可視化的影像資料夾（train/val/test/mini_test 都可）
OUT_DIR     = "experiments/vis_pred_v7"         # 輸出的圖片資料夾
MAX_IMAGES  = 50                                 # 最多輸出幾張（None 表示全部）
DRAW_GT     = False                              # 是否同時畫出 GT 框
GT_TXT      = "data/val/gt_val.txt"             # 若 DRAW_GT=True，指向相對應的 gt 檔（frame,x,y,w,h）

# 若以下兩個為 None，則會讀 YAML 的 infer.score_thr / infer.nms_iou
SCORE_THR_OVERRIDE = None                        # 例：0.40；None=使用 YAML
NMS_IOU_OVERRIDE   = None                        # 例：0.50；None=使用 YAML
# ===============================================

# 三種淺色（BGR）
PASTEL_BW_COLORS = [
    (200, 230, 255),  # 淺藍
    (210, 240, 210),  # 淺綠
    (245, 220, 235),  # 淺粉
]

_VALID_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP")
BBox = Tuple[float, float, float, float]  # (x1,y1,x2,y2)

def _numeric_sort_key(stem: str):
    """依數字檔名排序（容忍前導零）；若非數字，放到後面並依字串排序。"""
    s2 = stem.lstrip("0")
    if s2 == "":
        return (0, stem)
    try:
        return (int(s2), stem)
    except Exception:
        return (math.inf, stem)

def _list_images(img_dir: str) -> List[str]:
    paths: List[str] = []
    for ext in _VALID_EXTS:
        paths.extend(glob.glob(os.path.join(img_dir, f"*{ext}")))
    paths.sort(key=lambda p: _numeric_sort_key(os.path.splitext(os.path.basename(p))[0]))
    return paths

def _read_gt_txt(gt_txt: str) -> Dict[int, List[Tuple[int,int,int,int]]]:
    """讀取 gt.txt → {img_id: [(x,y,w,h), ...]}"""
    d: Dict[int, List[Tuple[int,int,int,int]]] = {}
    if not os.path.isfile(gt_txt):
        return d
    with open(gt_txt, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 5:
                continue
            frame, l, t, w, h = parts
            try:
                img_id = int(float(frame))
                l, t, w, h = map(int, (l, t, w, h))
            except Exception:
                continue
            if w <= 0 or h <= 0:
                continue
            d.setdefault(img_id, []).append((l, t, w, h))
    return d

def _load_model_from_cfg(cfg, ckpt_path: str, device: torch.device):
    """依 v7.yaml 建模並載入權重；支援多種 checkpoint 字典鍵。"""
    model = build_detector_from_cfg(cfg).to(device)
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)  # torch>=2.4 參數
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        sd = ckpt.get("model_ema", None) or ckpt.get("model", None) or ckpt.get("model_state", None) or ckpt
        model.load_state_dict(sd, strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    model.eval()
    return model

def _pil_to_cv_bgr(img: Image.Image) -> np.ndarray:
    return np.array(img)[:, :, ::-1].copy()

def _draw_translucent_box(
    img_bgr: np.ndarray,
    box_xyxy: Tuple[int,int,int,int],
    color_bgr: Tuple[int,int,int],
    alpha: float = 0.25,
    thickness: int = 2,
    label: Optional[str] = None,
):
    """畫半透明矩形 + 邊框 + 文字"""
    x1, y1, x2, y2 = [int(v) for v in box_xyxy]
    H, W = img_bgr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W - 1, x2), min(H - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return img_bgr

    overlay = img_bgr.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color_bgr, -1)           # 填滿
    cv2.addWeighted(overlay, alpha, img_bgr, 1 - alpha, 0, img_bgr)      # 混合
    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color_bgr, thickness)     # 外框

    if label:
        ((tw, th), baseline) = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        bar_h = th + baseline + 6
        y_top = max(0, y1 - bar_h)
        cv2.rectangle(img_bgr, (x1, y_top), (x1 + tw + 6, y_top + bar_h), color_bgr, -1)
        cv2.putText(img_bgr, label, (x1 + 3, y_top + bar_h - baseline - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
    return img_bgr

def _postprocess(boxes_xyxy: torch.Tensor,
                 scores: torch.Tensor,
                 labels: torch.Tensor,
                 score_thr: float,
                 nms_iou: float,
                 max_det: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """依 YAML 的 infer 設定做門檻與 NMS。單類別任務可直接整體 NMS。"""
    keep = scores >= score_thr
    boxes_xyxy = boxes_xyxy[keep]
    scores     = scores[keep]
    labels     = labels[keep]
    if boxes_xyxy.numel() == 0:
        return boxes_xyxy, scores, labels
    keep_idx = hard_nms(boxes_xyxy, scores, nms_iou)
    if max_det is not None:
        keep_idx = keep_idx[:max_det]
    return boxes_xyxy[keep_idx], scores[keep_idx], labels[keep_idx]

@torch.inference_mode()
def main():
    # 檢查
    assert os.path.isdir(IMG_DIR), f"找不到資料夾：{IMG_DIR}"
    assert os.path.isfile(CKPT_PATH), f"找不到權重：{CKPT_PATH}"
    if DRAW_GT:
        assert os.path.isfile(GT_TXT), f"DRAW_GT=True，但找不到 GT：{GT_TXT}"

    os.makedirs(OUT_DIR, exist_ok=True)

    # 讀 YAML 與裝置
    cfg = load_cfg(CFG_PATH)
    use_cuda = True
    try:
        use_cuda = bool(cfg["device"]["cuda"])
    except Exception:
        pass
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    # 從 YAML 取 infer 設定（可被 override）
    infer_cfg = cfg.get("infer", {})
    score_thr = float(SCORE_THR_OVERRIDE if SCORE_THR_OVERRIDE is not None
                      else infer_cfg.get("score_thr", 0.40))
    nms_iou   = float(NMS_IOU_OVERRIDE if NMS_IOU_OVERRIDE is not None
                      else infer_cfg.get("nms_iou", 0.50))
    max_det   = int(cfg.get("model", {}).get("roi", {}).get("detections_per_img", 100))

    # 建模並載入權重（完全對齊 v7）
    model = _load_model_from_cfg(cfg, CKPT_PATH, device)

    # 讀圖清單（依數字檔名排序，容忍前導零）
    img_files = _list_images(IMG_DIR)
    if MAX_IMAGES is not None:
        img_files = img_files[:int(MAX_IMAGES)]

    tfm_to_tensor = torchvision.transforms.ToTensor()
    gt_map = _read_gt_txt(GT_TXT) if DRAW_GT else {}

    for fp in tqdm(img_files, ncols=100, desc="VisPredV7"):
        fname = os.path.basename(fp)
        stem  = os.path.splitext(fname)[0]
        # 嘗試解析 frame id（僅作 GT 對應使用）
        try:
            frame_id = int(stem.lstrip("0") or "0")
        except Exception:
            frame_id = None

        # 讀圖（不做外部 resize，交給模型內 transform）
        pil_img = Image.open(fp).convert("RGB")
        x = tfm_to_tensor(pil_img).to(device).unsqueeze(0)

        # 推論
        out = model(x)[0]
        boxes_xyxy = out["boxes"].detach().cpu()
        scores     = out["scores"].detach().cpu()
        labels     = out["labels"].detach().cpu()

        # 後處理（門檻 + 額外 NMS，以 YAML 為準）
        boxes_xyxy, scores, labels = _postprocess(
            boxes_xyxy, scores, labels, score_thr, nms_iou, max_det
        )

        # 畫預測框
        img_bgr = _pil_to_cv_bgr(pil_img)
        for i, (b, s) in enumerate(zip(boxes_xyxy, scores)):
            x1, y1, x2, y2 = [int(round(v)) for v in b.tolist()]
            if x2 <= x1 or y2 <= y1:
                continue
            color = PASTEL_BW_COLORS[i % len(PASTEL_BW_COLORS)]
            label = f"pig {float(s):.2f}"
            img_bgr = _draw_translucent_box(img_bgr, (x1, y1, x2, y2), color,
                                            alpha=0.25, thickness=2, label=label)

        # 可選：畫 GT（固定偏青色）
        if DRAW_GT and (frame_id is not None) and (frame_id in gt_map):
            for (l, t, w, h) in gt_map[frame_id]:
                x1, y1, x2, y2 = l, t, l + w, t + h
                img_bgr = _draw_translucent_box(img_bgr, (x1, y1, x2, y2),
                                                (180, 220, 220),
                                                alpha=0.18, thickness=2, label="GT")

        # 存檔
        out_path = os.path.join(OUT_DIR, fname)
        cv2.imwrite(out_path, img_bgr)

    print(f"[Done] 已輸出可視化結果到：{OUT_DIR}")
    print(f"[Cfg] score_thr={score_thr}, nms_iou={nms_iou}, max_det={max_det}, device={device}")

if __name__ == "__main__":
    main()
