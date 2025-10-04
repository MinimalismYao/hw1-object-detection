#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
infer_to_csv.py
在 test set 跑推論並輸出 Kaggle 提交檔 submission.csv

Kaggle PredictionString 格式（單一類別）：
  "score x y w h 0 score x y w h 0 ..."
  其中最後的 0 是 class id（單類別）

預設結構：
  data/test/img  下有 00000001.jpg ...
  模型權重：experiments/logs/fasterrcnn_r50fpn_e5.pth

使用方式（有預設）：
  python src/infer_to_csv.py
或客製：
  python src/infer_to_csv.py --img-dir data/test/img \
      --checkpoint experiments/logs/fasterrcnn_r50fpn_e5.pth \
      --out submission.csv --max-side 800 --score-thr 0.1
"""
import os, csv, glob, argparse
import torch
import numpy as np
from PIL import Image
import torchvision

from model import get_fasterrcnn_r50_fpn

@torch.no_grad()
def run_infer_to_strings(model, img_dir, max_side=800, score_thr=0.1, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    tfm = torchvision.transforms.ToTensor()
    img_files = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    rows = []

    for fp in img_files:
        fname = os.path.basename(fp)
        fid_str = os.path.splitext(fname)[0]
        fid = int(fid_str)

        img = Image.open(fp).convert("RGB")
        w, h = img.size

        # 等比縮放
        scale = min(1.0, float(max_side) / max(w, h))
        if scale < 1.0:
            new_w, new_h = int(round(w * scale)), int(round(h * scale))
            img_resized = img.resize((new_w, new_h), resample=Image.BILINEAR)
        else:
            img_resized = img

        x = tfm(img_resized).to(device).unsqueeze(0)
        out = model(x)[0]
        boxes = out["boxes"].detach().cpu().numpy()
        scores = out["scores"].detach().cpu().numpy()

        # 還原回原圖座標
        if scale < 1.0:
            inv = 1.0 / scale
            boxes[:, [0, 2]] *= inv
            boxes[:, [1, 3]] *= inv

        # 過濾
        keep = scores >= score_thr
        boxes = boxes[keep]
        scores = scores[keep]

        # x1y1x2y2 → xywh 並裁邊
        boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, w - 1)
        boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, h - 1)
        xywh = boxes.copy()
        xywh[:, 2] = np.maximum(0.0, boxes[:, 2] - boxes[:, 0])
        xywh[:, 3] = np.maximum(0.0, boxes[:, 3] - boxes[:, 1])

        parts = []
        for b, s in zip(xywh, scores):
            if b[2] <= 0 or b[3] <= 0:
                continue
            # Kaggle 單類別格式：score x y w h 0
            parts.extend([
                f"{float(s):.6f}",
                f"{float(b[0]):.2f}",
                f"{float(b[1]):.2f}",
                f"{float(b[2]):.2f}",
                f"{float(b[3]):.2f}",
                "0"
            ])
        pred_str = " ".join(parts)  # 若沒有框，留空字串

        rows.append((fid, pred_str))

    # 依 Image_ID 升冪（保險）
    rows.sort(key=lambda x: x[0])
    return rows

def write_submission(rows, out_csv):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True) if os.path.dirname(out_csv) else None
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Image_ID", "PredictionString"])
        for fid, pred in rows:
            writer.writerow([fid, pred])

def parse_args():
    p = argparse.ArgumentParser("Inference on test set → submission.csv for Kaggle")
    p.add_argument("--img-dir", type=str, default="data/test/img")
    p.add_argument("--checkpoint", type=str, default="experiments/logs/fasterrcnn_r50fpn_e5.pth")
    p.add_argument("--out", type=str, default="submission.csv")
    p.add_argument("--max-side", type=int, default=800)
    p.add_argument("--score-thr", type=float, default=0.10)
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 構建/載入模型
    model = get_fasterrcnn_r50_fpn(num_classes=2, freeze_backbone=True).to(device)
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(args.checkpoint)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)

    rows = run_infer_to_strings(model, args.img-dir if hasattr(args, 'img-dir') else args.img_dir,
                                max_side=args.max_side, score_thr=args.score_thr, device=device)
    write_submission(rows, args.out)
    print(f"[Done] Wrote submission to: {args.out}")

if __name__ == "__main__":
    main()
