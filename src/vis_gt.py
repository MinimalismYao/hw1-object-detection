#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vis_gt.py
讀取 TAICA HW1 的 gt.txt，將 bbox 畫在對應影像上並輸出。
格式：<frame>, <bb_left>, <bb_top>, <bb_width>, <bb_height>

使用範例（專案根目錄）：
  python src/vis_gt.py \
    --gt data/train/gt.txt \
    --img-dir data/train/img \
    --out-dir experiments/vis_gt \
    --max-images 12 \
    --show 0

注意：
- 會自動偵測影像命名（1.jpg / 000001.jpg / .png ...）。
- 若 data/train/img 不存在，會自動退回 data/train。
- 需要套件：opencv-python、matplotlib
"""

import os
import csv
import glob
import argparse
from typing import Dict, List, Tuple, Optional

import cv2
import matplotlib.pyplot as plt

BBox = Tuple[int, int, int, int]  # (left, top, width, height)


# -----------------------------
# 1) 讀取與解析 gt.txt
# -----------------------------
def read_gt(gt_path: str) -> Dict[str, List[BBox]]:
    """
    讀取 gt.txt，回傳：{frame_id(str): [(l, t, w, h), ...]}
    - 容錯：忽略空行、容許逗號後空白
    - frame 轉成 "1","2"... 的純數字字串（避免 '1.0'）
    """
    boxes: Dict[str, List[BBox]] = {}
    if not os.path.isfile(gt_path):
        raise FileNotFoundError(f"gt.txt not found: {gt_path}")

    with open(gt_path, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            # 兼容「1, 307, 50, 96, 18」與「1,307,50,96,18」
            row = [x.strip() for x in row if x is not None]
            if len(row) != 5:
                row = [x.strip() for x in ",".join(row).split(",")]
            if len(row) != 5:
                # 可改成 print(row) 調查異常行
                continue

            frame, l, t, w, h = row
            # 轉 frame 成純整數再轉回字串：'1.0' -> '1'
            frame_id = str(int(float(frame)))
            l, t, w, h = map(int, (l, t, w, h))
            boxes.setdefault(frame_id, []).append((l, t, w, h))

    return boxes


# -----------------------------
# 2) 找到正確的影像目錄與檔名
# -----------------------------
def resolve_img_dir(img_dir_hint: str) -> str:
    """
    若傳入 data/train/img 但不存在，嘗試退回 data/train。
    """
    if os.path.isdir(img_dir_hint):
        return img_dir_hint
    fallback = os.path.dirname(img_dir_hint)
    if os.path.isdir(fallback):
        return fallback
    raise FileNotFoundError(
        f"Image directory not found. Tried: {img_dir_hint} and {fallback}"
    )


def find_image_path(img_dir: str, frame_id: str) -> Optional[str]:
    """
    依序嘗試多種命名與副檔名：
    1.jpg / 1.png / 000001.jpg / 000001.png / 大小寫副檔名
    若皆找不到，使用萬用字元做最後嘗試。
    """
    n = int(frame_id)
    candidates = [os.path.join(img_dir, f"{n:08d}.jpg"),]
    for p in candidates:
        if os.path.exists(p):
            return p

    # 最後廣義搜尋（第一個命中的就用）
    wildcards = [os.path.join(img_dir, f"{n:08d}.*")]
    
    for wc in wildcards:
        hits = glob.glob(wc)
        if hits:
            return hits[0]
    return None


# -----------------------------
# 3) 繪製與批次可視化
# -----------------------------
def draw_boxes_on_image(
    img_bgr,
    bboxes: List[BBox],
    color=(0, 255, 0),
    thickness: int = 2,
    clip: bool = True,
):
    """
    在單張影像上畫出所有 bbox。
    - clip=True：將框裁切到影像邊界內，避免負座標或超出邊界。
    """
    h_img, w_img = img_bgr.shape[:2]
    for (l, t, w, h) in bboxes:
        x1, y1 = l, t
        x2, y2 = l + w, t + h

        if clip:
            x1 = max(0, min(w_img - 1, x1))
            y1 = max(0, min(h_img - 1, y1))
            x2 = max(0, min(w_img - 1, x2))
            y2 = max(0, min(h_img - 1, y2))

        cv2.rectangle(img_bgr, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
    return img_bgr


def visualize_one(
    frame_id: str,
    img_dir: str,
    bboxes: List[BBox],
    save_path: Optional[str] = None,
    show: bool = False,
):
    """
    可視化單張：自動找檔名，畫框並存檔/顯示。
    """
    img_dir = resolve_img_dir(img_dir)
    img_path = find_image_path(img_dir, frame_id)
    if img_path is None:
        raise FileNotFoundError(
            f"Image not found for frame {frame_id}. "
            f"Tried in {img_dir} with patterns like 1.jpg / 000001.jpg / .png"
        )

    img = cv2.imread(img_path)
    if img is None:
        raise RuntimeError(f"Failed to read image: {img_path}")

    img_drawn = draw_boxes_on_image(img.copy(), bboxes)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, img_drawn)

    if show:
        plt.figure(figsize=(8, 6))
        plt.imshow(cv2.cvtColor(img_drawn, cv2.COLOR_BGR2RGB))
        plt.title(f"Frame {frame_id} — {len(bboxes)} boxes\n{os.path.basename(img_path)}")
        plt.axis("off")
        plt.show()


def visualize_batch(
    gt_path: str,
    train_img_dir: str,
    out_dir: str,
    max_images: int = 10,
    show: bool = False,
    sort_frame_ids: bool = True,
):
    """
    讀 gt.txt，對前 max_images 張「有標註」的 frame 產生可視化結果。
    """
    boxes = read_gt(gt_path)

    # 決定處理順序
    keys = list(boxes.keys())
    if sort_frame_ids:
        # 依數字排序：'10' 會排在 '2' 之後
        keys = sorted(keys, key=lambda s: int(s))

    os.makedirs(out_dir, exist_ok=True)

    count = 0
    for frame_id in keys:
        save_path = os.path.join(out_dir, f"{frame_id}.jpg")
        try:
            visualize_one(
                frame_id=frame_id,
                img_dir=train_img_dir,
                bboxes=boxes[frame_id],
                save_path=save_path,
                show=show,
            )
            count += 1
            if count >= max_images:
                break
        except Exception as e:
            print(f"[Warn] Skip frame {frame_id}: {e}")

    print(f"[Done] Saved {count} visualized images to: {out_dir}")


# -----------------------------
# 4) CLI
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Visualize TAICA HW1 gt.txt bounding boxes on images."
    )
    p.add_argument("--gt", type=str, default="data/train/gt.txt", help="Path to gt.txt")
    p.add_argument(
        "--img-dir",
        type=str,
        default="data/train/img",
        help="Image folder (will fallback to its parent if not found)",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="experiments/vis_gt",
        help="Output folder for visualized images",
    )
    p.add_argument(
        "--max-images",
        type=int,
        default=12,
        help="How many frames to visualize (with labels)",
    )
    p.add_argument(
        "--show",
        type=int,
        default=0,
        choices=[0, 1],
        help="Show with matplotlib (1) or just save (0)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    visualize_batch(
        gt_path=args.gt,
        train_img_dir=args.img_dir,
        out_dir=args.out_dir,
        max_images=args.max_images,
        show=bool(args.show),
    )


if __name__ == "__main__":
    main()
