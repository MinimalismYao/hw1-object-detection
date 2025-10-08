#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vis_pred.py
從指定資料夾讀取影像 → 用已訓練模型推論 → 畫出半透明淡色的預測框（可選 GT）→ 存檔。

使用方式（專案根目錄）：
  python src/vis_pred.py

只需修改檔頭的設定區即可。
"""

import os, glob
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torchvision

from modelv5 import get_fasterrcnn_r50_fpn

# ========== 可自行修改的設定 ==========
IMG_DIR     = "data/mini_test"                      # 要可視化的影像資料夾（小測試集/驗證集/任一資料夾）
CKPT_PATH   = "experiments/logs/fasterrcnn_v5/fasterrcnn_v5.pth"  # 權重
OUT_DIR     = "experiments/vis_pred"              # 輸出的圖片資料夾
MAX_SIDE    = 1024                                  # 推論時最長邊縮放（保持等比）
SCORE_THR   = 0.8                                 # 顯示框的分數門檻
MAX_IMAGES  = 50                                   # 最多輸出幾張（None 表示全部）
DRAW_GT     = False                                # 是否同時畫出 GT 框
GT_TXT      = "data/val/gt_val.txt"               # 若 DRAW_GT=True，這裡要指向相對應的 gt 檔（frame,x,y,w,h）
# =====================================

# 三種淺色（BGR）
PASTEL_BW_COLORS = [
    (200, 230, 255),  # 淺橙藍(淡藍)
    (210, 240, 210),  # 淺綠
    (245, 220, 235),  # 淺粉
]

BBox = Tuple[float, float, float, float]  # (x1,y1,x2,y2)

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

def _load_model(ckpt_path: str, device: torch.device):
    model = get_fasterrcnn_r50_fpn(num_classes=2, freeze_backbone=True).to(device)
    # 安全載入（兼容舊版）
    try:
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model

def _load_and_resize_pil(fp: str, max_side: int):
    img = Image.open(fp).convert("RGB")
    w, h = img.size
    scale = min(1.0, float(max_side) / max(w, h))
    if scale < 1.0:
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        img = img.resize((new_w, new_h), resample=Image.BILINEAR)
    return img, (w, h), scale

def _to_numpy_img_bgr(pil_img: Image.Image) -> np.ndarray:
    # PIL RGB -> OpenCV BGR
    arr = np.array(pil_img)[:, :, ::-1].copy()
    return arr

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
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img_bgr.shape[1]-1, x2), min(img_bgr.shape[0]-1, y2)
    if x2 <= x1 or y2 <= y1:
        return img_bgr

    overlay = img_bgr.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color_bgr, -1)  # 填滿
    cv2.addWeighted(overlay, alpha, img_bgr, 1 - alpha, 0, img_bgr)  # 混合
    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color_bgr, thickness)  # 外框

    if label:
        ((tw, th), baseline) = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        bar_h = th + baseline + 6
        cv2.rectangle(img_bgr, (x1, y1 - bar_h), (x1 + tw + 6, y1), color_bgr, -1)
        cv2.putText(img_bgr, label, (x1 + 3, y1 - baseline - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
    return img_bgr

@torch.inference_mode()
def main():
    # 檢查
    assert os.path.isdir(IMG_DIR), f"找不到資料夾：{IMG_DIR}"
    assert os.path.isfile(CKPT_PATH), f"找不到權重：{CKPT_PATH}"
    if DRAW_GT:
        assert os.path.isfile(GT_TXT), f"DRAW_GT=True，但找不到 GT：{GT_TXT}"

    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    model = _load_model(CKPT_PATH, device)
    tfm = torchvision.transforms.ToTensor()

    gt_map = _read_gt_txt(GT_TXT) if DRAW_GT else {}

    img_files = sorted(glob.glob(os.path.join(IMG_DIR, "*.jpg")))
    if MAX_IMAGES is not None:
        img_files = img_files[:MAX_IMAGES]

    for fp in tqdm(img_files, ncols=100, desc="VisPred"):
        fname = os.path.basename(fp)
        fid = int(os.path.splitext(fname)[0])

        # 讀圖 + 等比縮放
        pil_img, (W, H), scale = _load_and_resize_pil(fp, MAX_SIDE)
        x = tfm(pil_img).to(device).unsqueeze(0)

        # 推論
        out = model(x)[0]
        boxes = out["boxes"].detach().cpu().numpy()
        scores = out["scores"].detach().cpu().numpy()

        # 還原座標
        if scale < 1.0:
            inv = 1.0 / scale
            boxes[:, [0, 2]] *= inv
            boxes[:, [1, 3]] *= inv

        # 過濾 + 轉為 int
        keep = scores >= SCORE_THR
        boxes = boxes[keep]
        scores = scores[keep]

        img_bgr = _to_numpy_img_bgr(pil_img if scale == 1.0 else pil_img.resize((W, H), Image.BILINEAR))

        # 畫預測框（固定三色輪流）
        for i, (b, s) in enumerate(zip(boxes, scores)):
            x1, y1, x2, y2 = [int(round(v)) for v in b]
            w_box, h_box = max(0, x2 - x1), max(0, y2 - y1)
            if w_box <= 0 or h_box <= 0:
                continue
            color = PASTEL_BW_COLORS[i % len(PASTEL_BW_COLORS)]
            label = f"pig {s:.2f}"
            img_bgr = _draw_translucent_box(img_bgr, (x1, y1, x2, y2), color, alpha=0.25, thickness=2, label=label)

        # 可選：畫 GT
        if DRAW_GT and fid in gt_map:
            for (l, t, w, h) in gt_map[fid]:
                x1, y1, x2, y2 = l, t, l + w, t + h
                # 用固定顏色區分 GT（偏青色）
                img_bgr = _draw_translucent_box(img_bgr, (x1, y1, x2, y2), (180, 220, 220),
                                                alpha=0.18, thickness=2, label="GT")

        # 存檔
        out_path = os.path.join(OUT_DIR, fname)
        cv2.imwrite(out_path, img_bgr)

    print(f"[Done] 已輸出可視化結果到：{OUT_DIR}")

if __name__ == "__main__":
    main()
