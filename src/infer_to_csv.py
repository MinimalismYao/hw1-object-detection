#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
infer_to_csv.py
Test set 推論 → 產生 Kaggle submission.csv
格式（單一類別）: "score x y w h 0 ..."（0 是 class id）

使用方式：
  直接修改檔頭常數（CKPT_PATH, IMG_DIR, OUT_CSV, ...），然後執行：
      python src/infer_to_csv.py
"""

import os, csv, glob
import torch
import numpy as np
from PIL import Image
import torchvision
from tqdm import tqdm

from model import get_fasterrcnn_r50_fpn

# ========= 可自行修改的設定 =========
IMG_DIR     = "data/test/img"                             # 要推論的影像資料夾
CKPT_PATH   = "experiments/logs/fasterrcnn_r50fpn_final_v1.pth" # 權重檔
OUT_CSV     = "submission_v1.csv"                            # 輸出檔名
MAX_SIDE    = 800                                         # 推論時最長邊縮放
SCORE_THR   = 0.10                                        # 分數閾值
MAX_IMAGES  = None                                        # 限制最多推論幾張（None 表示全部）
# ===================================


def load_model(ckpt_path, device):
    model = get_fasterrcnn_r50_fpn(num_classes=2, freeze_backbone=True).to(device)

    # 載入權重（你存的是 state_dict）
    try:
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # （相容性保護）某些 torchvision 版本這些是 dict，有些是方法/物件
    try:
        pre_nms = getattr(model.rpn, "pre_nms_top_n", None)
        post_nms = getattr(model.rpn, "post_nms_top_n", None)
        if isinstance(pre_nms, dict) and isinstance(post_nms, dict):
            pre_nms["testing"]  = min(3000, pre_nms.get("testing", 1000))
            post_nms["testing"] = min(1000, post_nms.get("testing", 300))
    except Exception:
        pass

    return model


def load_image(fp, max_side=800):
    img = Image.open(fp).convert("RGB")
    w, h = img.size
    scale = min(1.0, float(max_side) / max(w, h))
    if scale < 1.0:
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        img = img.resize((new_w, new_h), resample=Image.BILINEAR)
    return img, (w, h), scale


@torch.inference_mode()
def run_infer_to_strings(model, img_dir, max_side=800, score_thr=0.10, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tfm = torchvision.transforms.ToTensor()

    # 支援多種副檔名
    img_files = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        img_files.extend(glob.glob(os.path.join(img_dir, ext)))
    # 檔名以數字排序（00000001.jpg → 1）
    def _key(fp):
        base = os.path.splitext(os.path.basename(fp))[0]
        try:
            return int(base)
        except ValueError:
            return base
    img_files = sorted(img_files, key=_key)

    if MAX_IMAGES is not None:
        img_files = img_files[:MAX_IMAGES]

    rows = []

    for fp in tqdm(img_files, desc="Infer", ncols=100):
        fname = os.path.basename(fp)
        fid_str = os.path.splitext(fname)[0]
        try:
            fid = int(fid_str)
        except ValueError:
            # 非數字命名：Kaggle 通常需要數字 Image_ID，這裡略過避免壞掉
            continue

        img_resized, (orig_w, orig_h), scale = load_image(fp, max_side=max_side)
        x = tfm(img_resized).to(device).unsqueeze(0)

        out = model(x)[0]
        boxes = out["boxes"].detach().cpu().numpy()
        scores = out["scores"].detach().cpu().numpy()

        # 還原到原圖座標
        if scale < 1.0:
            inv = 1.0 / scale
            boxes[:, [0, 2]] *= inv
            boxes[:, [1, 3]] *= inv

        # 分數過濾
        keep = scores >= score_thr
        boxes = boxes[keep]
        scores = scores[keep]

        # x1y1x2y2 → xywh 並裁邊
        boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, orig_w - 1)
        boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, orig_h - 1)
        xywh = boxes.copy()
        xywh[:, 2] = np.maximum(0.0, boxes[:, 2] - boxes[:, 0])
        xywh[:, 3] = np.maximum(0.0, boxes[:, 3] - boxes[:, 1])

        parts = []
        for b, s in zip(xywh, scores):
            if b[2] <= 0 or b[3] <= 0:
                continue
            parts.extend([
                f"{float(s):.6f}",
                f"{float(b[0]):.2f}",
                f"{float(b[1]):.2f}",
                f"{float(b[2]):.2f}",
                f"{float(b[3]):.2f}",
                "0"  # 單一類別 id
            ])
        pred_str = " ".join(parts)
        rows.append((fid, pred_str))

    rows.sort(key=lambda x: x[0])
    return rows


def write_submission(rows, out_csv):
    d = os.path.dirname(out_csv)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Image_ID", "PredictionString"])
        for fid, pred in rows:
            w.writerow([fid, pred])


def main():
    # 基本檢查
    assert os.path.isdir(IMG_DIR), f"找不到影像資料夾：{IMG_DIR}"
    assert os.path.isfile(CKPT_PATH), f"找不到權重：{CKPT_PATH}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    print(f"[Config] ckpt={CKPT_PATH}  img_dir={IMG_DIR}  out={OUT_CSV}  "
          f"max_side={MAX_SIDE}  thr={SCORE_THR}")

    model = load_model(CKPT_PATH, device)
    rows = run_infer_to_strings(model, IMG_DIR, max_side=MAX_SIDE, score_thr=SCORE_THR, device=device)
    write_submission(rows, OUT_CSV)
    print(f"[Done] Wrote submission to: {OUT_CSV}")


if __name__ == "__main__":
    main()
