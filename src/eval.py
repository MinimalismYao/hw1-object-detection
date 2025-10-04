#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval.py
對驗證集 (val) 進行推論並計算 mAP@50 與 mAP@50:95
使用 pycocotools COCO API。

需求：
- pycocotools
- torchvision
- dataset.py / model.py / transforms.py (專案已有)

輸出：
- 結果印在螢幕
- 存到 experiments/eval_results.txt
"""

import os, glob, torch, json
import numpy as np
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from PIL import Image

import torchvision
from dataset import PigsDataset, collate_fn
from transforms import get_transforms
from model import get_fasterrcnn_r50_fpn




# -----------------------------
# 載入模型
# -----------------------------
def load_model(ckpt_path, device):
    model = get_fasterrcnn_r50_fpn(num_classes=2, freeze_backbone=True).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


# -----------------------------
# 建立 COCO GT dict
# -----------------------------
def build_coco_from_gt(gt_txt, img_dir):
    images, annotations = [], []
    ann_id, seen = 1, set()

    with open(gt_txt, "r") as f:
        for line in f:
            if not line.strip():
                continue
            frame, l, t, w, h = [x.strip() for x in line.split(",")]
            img_id = int(float(frame))
            file_name = f"{img_id:08d}.jpg"

            if img_id not in seen:
                with Image.open(os.path.join(img_dir, file_name)) as im:
                    W, H = im.size
                images.append({"id": img_id, "file_name": file_name,
                               "width": W, "height": H})
                seen.add(img_id)

            bbox = [int(l), int(t), int(w), int(h)]
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": 1,
                "bbox": bbox,
                "area": int(w) * int(h),
                "iscrowd": 0,
            })
            ann_id += 1

    coco_dict = {
        "info": {"description": "TAICA HW1 val"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "pig"}],
    }

    coco_gt = COCO()
    coco_gt.dataset = coco_dict
    coco_gt.createIndex()
    return coco_gt


# -----------------------------
# 推論並轉 COCO 格式
# -----------------------------
@torch.inference_mode()
def infer_and_build_dt(model, img_dir, device, max_side=800, score_thr=0.05):
    tfm = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()])

    img_files = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    results = []

    for fp in tqdm(img_files, desc="Infer", ncols=100):
        fname = os.path.basename(fp)
        fid = int(os.path.splitext(fname)[0])

        img = Image.open(fp).convert("RGB")
        W, H = img.size
        scale = min(1.0, float(max_side) / max(W, H))
        if scale < 1.0:
            new_w, new_h = int(W * scale), int(H * scale)
            img = img.resize((new_w, new_h), resample=Image.BILINEAR)

        x = torchvision.transforms.ToTensor()(img).to(device).unsqueeze(0)
        out = model(x)[0]

        boxes = out["boxes"].detach().cpu().numpy()
        scores = out["scores"].detach().cpu().numpy()

        if scale < 1.0:
            inv = 1.0 / scale
            boxes[:, [0, 2]] *= inv
            boxes[:, [1, 3]] *= inv

        keep = scores >= score_thr
        boxes, scores = boxes[keep], scores[keep]

        for b, s in zip(boxes, scores):
            x1, y1, x2, y2 = b
            w, h = x2 - x1, y2 - y1
            if w <= 0 or h <= 0:
                continue
            results.append({
                "image_id": fid,
                "category_id": 1,
                "bbox": [float(x1), float(y1), float(w), float(h)],
                "score": float(s),
            })

    return results


# -----------------------------
# 主程式
# -----------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = "experiments/logs/fasterrcnn_r50fpn_e5.pth"   # 👈 你可以改成想要的權重
    img_dir = "data/val/img"
    gt_txt = "data/val/gt_val.txt"

    if not os.path.isfile(ckpt):
        raise FileNotFoundError(ckpt)

    model = load_model(ckpt, device)

    # === 建 GT ===
    coco_gt = build_coco_from_gt(gt_txt, img_dir)

    # === 建 DT ===
    coco_dt_list = infer_and_build_dt(model, img_dir, device)
    coco_dt = coco_gt.loadRes(coco_dt_list)

    # === Eval ===
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    m50 = coco_eval.stats[1]   # mAP@50
    m5095 = coco_eval.stats[0] # mAP@50:95

    out_path = "experiments/eval_results.txt"
    with open(out_path, "w") as f:
        f.write(f"mAP@50: {m50:.4f}\n")
        f.write(f"mAP@50:95: {m5095:.4f}\n")
    print(f"[Done] Results saved to {out_path}")


if __name__ == "__main__":
    main()
