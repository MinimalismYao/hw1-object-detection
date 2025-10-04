#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
infer_to_csv.py
Test set 推論 → 產生 Kaggle submission.csv
格式（單一類別）: "score x y w h 0 ..."（0 是 class id）

預設：
  --img-dir data/test/img
  --checkpoint experiments/logs/fasterrcnn_r50fpn_e5.pth
"""
import os, csv, glob, argparse
import torch
import numpy as np
from PIL import Image
import torchvision
from tqdm import tqdm

from model import get_fasterrcnn_r50_fpn

def load_model(ckpt_path, device):
    model = get_fasterrcnn_r50_fpn(num_classes=2, freeze_backbone=True).to(device)

    # 載入權重（你存的是 state_dict，安全）
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    # ---- 可選加速：僅在屬性是 dict 時才調整；否則略過，避免不同版本 torchvision 出錯 ----
    try:
        pre_nms = getattr(model.rpn, "pre_nms_top_n", None)
        post_nms = getattr(model.rpn, "post_nms_top_n", None)
        if isinstance(pre_nms, dict) and isinstance(post_nms, dict):
            pre_nms["testing"] = min(3000, pre_nms.get("testing", 1000))
            post_nms["testing"] = min(1000, post_nms.get("testing", 300))
    except Exception:
        # 不同版本結構差異，安全忽略
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

    img_files = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    rows = []

    for fp in tqdm(img_files, desc="Infer", ncols=100):
        fname = os.path.basename(fp)
        fid_str = os.path.splitext(fname)[0]
        fid = int(fid_str)

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
                "0"  # class id
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

def parse_args():
    p = argparse.ArgumentParser("Inference on test set → submission.csv for Kaggle")
    p.add_argument("--img_dir", type=str, default="data/test/img")
    p.add_argument("--checkpoint", type=str, default="experiments/logs/fasterrcnn_r50fpn_e5.pth")
    p.add_argument("--out", type=str, default="submission.csv")
    p.add_argument("--max_side", type=int, default=800)
    p.add_argument("--score_thr", type=float, default=0.10)
    return p.parse_args()

def main():
    # 內建預設（可改）
    DEFAULT_CKPT = "experiments/logs/fasterrcnn_r50fpn_final_v1.pth"
    DEFAULT_IMG  = "data/test/img"
    DEFAULT_OUT  = "submission_v1.csv"
    DEFAULT_MAX  = 800
    DEFAULT_THR  = 0.10

    args = parse_args()
    ckpt = args.checkpoint or DEFAULT_CKPT
    imgd = args.img_dir   or DEFAULT_IMG
    outc = args.out       or DEFAULT_OUT
    mside = args.max_side if args.max_side else DEFAULT_MAX
    thr   = args.score_thr if args.score_thr is not None else DEFAULT_THR

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    if not os.path.isfile(ckpt):
        raise FileNotFoundError(ckpt)
    if not os.path.isdir(imgd):
        raise FileNotFoundError(imgd)

    model = load_model(ckpt, device)
    rows = run_infer_to_strings(model, imgd, max_side=mside, score_thr=thr, device=device)
    write_submission(rows, outc)
    print(f"[Done] Wrote submission to: {outc}")



if __name__ == "__main__":
    main()

