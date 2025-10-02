# src/vis_gt.py
# 解析 gt.txt 並可視化 bbox（不使用任何預訓練與外部資料，僅讀官方提供標註）
# 格式參考：<frame>, <bb_left>, <bb_top>, <bb_width>, <bb_height>

import os
import csv
from typing import Dict, List, Tuple
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import glob

BBox = Tuple[float, float, float, float]  # (left, top, width, height)

def read_gt(gt_path: str) -> Dict[str, List[BBox]]:
    """
    讀取 gt.txt，回傳：{frame_id(str): [(l, t, w, h), ...]}
    - 允許空白與逗號後空格
    - 忽略空行
    """
    boxes: Dict[str, List[BBox]] = {}
    with open(gt_path, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            # 清除可能的空白
            row = [x.strip() for x in row if x is not None]
            # 有些標註可能寫成 "1, 12, 34, 56, 78"
            if len(row) != 5:
                # 嘗試用手動 split 再次解析（以防混合空白）
                row = [x.strip() for x in ",".join(row).split(",")]
            if len(row) != 5:
                # 仍然不對就跳過或 raise
                # 你也可以改成 print(row) 檢查異常行
                continue

            frame, l, t, w, h = row
            # 統一 frame 為字串（對應影像檔名）
            frame_id = str(int(float(frame)))  # 兼容 "1" 或 "1.0"
            l, t, w, h = map(float, (l, t, w, h))

            boxes.setdefault(frame_id, []).append((l, t, w, h))
    return boxes


def draw_boxes_on_image(img_bgr, bboxes: List[BBox], color=(0, 255, 0), thickness=2, clip=True):
    """
    在單張影像上畫出所有 bbox；預設會裁到影像邊界（因為 sample_submission 可能含負座標）
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

        cv2.rectangle(img_bgr, (int(round(x1)), int(round(y1))),
                      (int(round(x2)), int(round(y2))), color, thickness)
    return img_bgr


def visualize_one(frame_id: str, img_dir: str, bboxes: List[BBox], save_path: str = None, show: bool = True):
    """
    可視化某一張 frame：從 img_dir 讀 <frame_id>.jpg，畫框後存檔/顯示
    """
    # 你也可以改規則，例如 f"{frame_id:06d}.jpg"
    img_path = os.path.join(img_dir, f"{int(frame_id):07d}.jpg")
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")

    img = cv2.imread(img_path)  # BGR
    if img is None:
        raise RuntimeError(f"Failed to read image: {img_path}")

    img_drawn = draw_boxes_on_image(img.copy(), bboxes)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, img_drawn)

    if show:
        # 用 matplotlib 以 RGB 顯示
        plt.figure(figsize=(8, 6))
        plt.imshow(cv2.cvtColor(img_drawn, cv2.COLOR_BGR2RGB))
        plt.title(f"Frame {frame_id} — {len(bboxes)} boxes")
        plt.axis("off")
        plt.show()


def visualize_batch(gt_path: str,
                    train_img_dir: str,
                    out_dir: str = "./experiments/vis_gt",
                    max_images: int = 10,
                    show: bool = False):
    """
    讀 gt.txt，對前 max_images 張有標註的影像輸出可視化結果到 out_dir
    """
    boxes = read_gt(gt_path)
    os.makedirs(out_dir, exist_ok=True)
    count = 0
    for frame_id, bboxes in boxes.items():
        save_path = os.path.join(out_dir, f"{frame_id}.jpg")
        visualize_one(frame_id, train_img_dir, bboxes, save_path=save_path, show=show)
        count += 1
        if count >= max_images:
            break
    print(f"[Done] Saved {count} visualized images to: {out_dir}")


if __name__ == "__main__":
    # 預設資料結構：data/train/img 與 data/train/gt.txt
    GT = "data/train/gt.txt"
    IMG_DIR = "data/train/img"
    visualize_batch(GT, IMG_DIR, out_dir="./experiments/vis_gt", max_images=12, show=False)
