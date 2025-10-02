#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vis_gt.py
直接執行即可跑：讀取 TAICA HW1 的 gt.txt，將標註框畫到影像上並輸出。

"""

import os
import csv
import glob
from typing import Dict, List, Tuple, Optional

import cv2
import matplotlib.pyplot as plt

# -----------------------
# 預設設定（改這裡就好）
# -----------------------
GT_PATH   = "data/train/gt.txt"          # 標註檔
IMG_DIR   = "data/train/img"             # 影像資料夾
OUT_DIR   = "experiments/vis_gt"         # 輸出資料夾
MAX_IMAGES = 50                          # 最多可視化幾張
SHOW = False                             # 是否顯示 (True/False)

BBox = Tuple[int, int, int, int]  # (left, top, width, height)


# ---------------------------------------------------------------------
# 讀取 gt.txt
# ---------------------------------------------------------------------
def read_gt(gt_path: str) -> Dict[str, List[BBox]]:
    boxes: Dict[str, List[BBox]] = {}
    if not os.path.isfile(gt_path):
        raise FileNotFoundError(f"找不到 gt.txt：{gt_path}")

    with open(gt_path, "r", newline="") as f:
        for line in f:
            if not line.strip():
                continue
            row = [x.strip() for x in line.split(",")]
            if len(row) != 5:
                continue
            frame, l, t, w, h = row
            frame_id = str(int(float(frame)))
            l, t, w, h = map(int, (l, t, w, h))
            boxes.setdefault(frame_id, []).append((l, t, w, h))
    return boxes


# ---------------------------------------------------------------------
# 找到對應圖片路徑
# ---------------------------------------------------------------------
def find_image_path(img_dir: str, frame_id: str) -> Optional[str]:
    n = int(frame_id)
    candidate = os.path.join(img_dir, f"{n:08d}.jpg")
    if os.path.exists(candidate):
        return candidate
    hits = glob.glob(os.path.join(img_dir, f"{n:08d}.*"))
    return hits[0] if hits else None


# ---------------------------------------------------------------------
# 畫框
# ---------------------------------------------------------------------
def draw_boxes_on_image(img_bgr, bboxes: List[BBox], color=(0, 255, 0), thickness=2):
    h_img, w_img = img_bgr.shape[:2]
    for (l, t, w, h) in bboxes:
        x1, y1 = l, t
        x2, y2 = l + w, t + h
        # 裁切避免超框
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img - 1, x2), min(h_img - 1, y2)
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, thickness)
    return img_bgr


# ---------------------------------------------------------------------
# 可視化單張
# ---------------------------------------------------------------------
def visualize_one(frame_id: str, img_dir: str, bboxes: List[BBox], save_path: str, show: bool):
    img_path = find_image_path(img_dir, frame_id)
    if img_path is None:
        print(f"[Warn] 找不到影像 frame={frame_id}")
        return

    img = cv2.imread(img_path)
    if img is None:
        print(f"[Warn] 無法讀取影像：{img_path}")
        return

    img_drawn = draw_boxes_on_image(img.copy(), bboxes)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, img_drawn)

    if show:
        plt.imshow(cv2.cvtColor(img_drawn, cv2.COLOR_BGR2RGB))
        plt.title(f"Frame {frame_id} — {len(bboxes)} boxes")
        plt.axis("off")
        plt.show()


# ---------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------
def main():
    boxes = read_gt(GT_PATH)
    frame_ids = sorted(boxes.keys(), key=lambda s: int(s))

    saved = 0
    for fid in frame_ids:
        save_path = os.path.join(OUT_DIR, f"{fid}.jpg")
        visualize_one(fid, IMG_DIR, boxes[fid], save_path, SHOW)
        saved += 1
        if saved >= MAX_IMAGES:
            break

    print(f"[Done] 共輸出 {saved} 張到：{OUT_DIR}")


if __name__ == "__main__":
    main()
