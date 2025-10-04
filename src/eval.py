#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval.py
在 validation set 計算 mAP@50 與 mAP@50:95（使用 pycocotools）。
預設資料結構：
  data/val/
    ├─ img/               # 驗證影像（檔名如 00000001.jpg）
    └─ gt_val.txt         # 內容: frame, l, t, w, h  （與 train 的 gt.txt 同格式）

模型：
  experiments/logs/fasterrcnn_r50fpn_e5.pth  （可改）

使用方式（有預設）：
  python src/eval.py
或客製：
  python src/eval.py --img-dir data/val/img --gt data/val/gt_val.txt \
                     --checkpoint experiments/logs/fasterrcnn_r50fpn_e5.pth \
                     --max-side 800 --score-thr 0.05
"""
import os, json, argparse, glob
import torch
import torchvision
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from model import get_fasterrcnn_r50_fpn

def read_gt_txt(gt_path):
    """
    讀 <frame,l,t,w,h> 轉成 dict: {fid_str: [[l,t,w,h], ...]}
    """
    boxes_by_id = {}
    with open(gt_path, "r", newline="") as f:
        for line in f:
            if not line.strip():
                continue
            v = [x.strip() for x in line.split(",")]
            if len(v) != 5:
                continue
            frame, l, t, w, h = v
            fid = f"{int(float(frame)):08d}"
            l, t, w, h = map(int, (l, t, w, h))
            if w <= 0 or h <= 0:
                continue
            boxes_by_id.setdefault(fid, []).append([l, t, w, h])
    return boxes_by_id

def build_coco_gt(img_dir, gt_path, out_json_path=None):
    """
    把 val 的標註轉成 COCO JSON 給 pycocotools 用。
    單一類別：pig，category_id=1
    """
    boxes_by_id = read_gt_txt(gt_path)
    # 收集 val 圖片 id（依檔案存在）
    img_files = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    images = []
    annotations = []
    ann_id = 1
    for fp in img_files:
        fname = os.path.basename(fp)
        fid = os.path.splitext(fname)[0]  # 00000001
        try:
            im = Image.open(fp).convert("RGB")
            w, h = im.size
        except Exception:
            continue
        images.append({
            "id": int(fid),
            "file_name": fname,
            "width": w,
            "height": h
        })
        for l, t, bw, bh in boxes_by_id.get(fid, []):
            annotations.append({
                "id": ann_id,
                "image_id": int(fid),
                "category_id": 1,
                "bbox": [float(l), float(t), float(bw), float(bh)],
                "area": float(bw * bh),
                "iscrowd": 0
            })
            ann_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "pig"}]
    }
    if out_json_path:
        os.makedirs(os.path.dirname(out_json_path), exist_ok=True)
        with open(out_json_path, "w") as f:
            json.dump(coco, f)
    return coco

@torch.no_grad()
def run_inference(model, img_dir, max_side=800, score_thr=0.05, device=None):
    """
    對 val 圖片跑推論，回傳 COCO dt list
    COCO dt 欄位：image_id, category_id, bbox[x,y,w,h], score
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    img_files = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    results = []
    tfm = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])

    for fp in img_files:
        fname = os.path.basename(fp)
        fid = int(os.path.splitext(fname)[0])
        img = Image.open(fp).convert("RGB")
        # 等比縮放：長邊不超過 max_side
        w, h = img.size
        scale = min(1.0, float(max_side) / max(w, h))
        if scale < 1.0:
            new_w, new_h = int(round(w * scale)), int(round(h * scale))
            img_resized = img.resize((new_w, new_h), resample=Image.BILINEAR)
        else:
            img_resized = img

        x = tfm(img_resized).to(device).unsqueeze(0)  # 1,C,H,W
        out = model(x)[0]
        boxes = out["boxes"].detach().cpu().numpy()
        scores = out["scores"].detach().cpu().numpy()

        # 把 boxes 從 resized 空間還原回原圖座標
        if scale < 1.0:
            inv = 1.0 / scale
            boxes[:, [0, 2]] *= inv
            boxes[:, [1, 3]] *= inv

        # 過濾低分數
        keep = scores >= score_thr
        boxes = boxes[keep]
        scores = scores[keep]

        # x1y1x2y2 → xywh，並裁切到影像內
        boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, w - 1)
        boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, h - 1)
        xywh = boxes.copy()
        xywh[:, 2] = np.maximum(0.0, boxes[:, 2] - boxes[:, 0])
        xywh[:, 3] = np.maximum(0.0, boxes[:, 3] - boxes[:, 1])

        for b, s in zip(xywh, scores):
            if b[2] <= 0 or b[3] <= 0:
                continue
            results.append({
                "image_id": fid,
                "category_id": 1,         # 單一類別
                "bbox": [float(b[0]), float(b[1]), float(b[2]), float(b[3])],
                "score": float(s)
            })
    return results

def coco_eval_from_dicts(coco_gt_dict, coco_dt_list):
    # 建 COCO api
    coco_gt = COCO()
    coco_gt.dataset = coco_gt_dict
    coco_gt.createIndex()

    coco_dt = coco_gt.loadRes(coco_dt_list)

    # mAP50:95
    E = COCOeval(coco_gt, coco_dt, iouType='bbox')
    E.evaluate(); E.accumulate(); E.summarize()
    map_50_95 = E.stats[0]  # AP @[.5:.95]

    # mAP50
    E50 = COCOeval(coco_gt, coco_dt, iouType='bbox')
    E50.params.iouThrs = np.array([0.5])
    E50.evaluate(); E50.accumulate(); E50.summarize()
    map_50 = E50.stats[0]

    return map_50, map_50_95

def parse_args():
    p = argparse.ArgumentParser("Eval mAP on validation set (pycocotools).")
    p.add_argument("--img-dir", type=str, default="data/val/img")
    p.add_argument("--gt", type=str, default="data/val/gt_val.txt")
    p.add_argument("--checkpoint", type=str, default="experiments/logs/fasterrcnn_r50fpn_e5.pth")
    p.add_argument("--max-side", type=int, default=800)
    p.add_argument("--score-thr", type=float, default=0.05)
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

    # 準備 COCO GT
    coco_gt = build_coco_gt(args.img_dir, args.gt)

    # 推論 → COCO dt
    coco_dt = run_inference(model, args.img_dir, max_side=args.max_side,
                            score_thr=args.score_thr, device=device)

    # 評估
    m50, m5095 = coco_eval_from_dicts(coco_gt, coco_dt)
    print(f"\n[Eval] mAP@50 = {m50:.4f}")
    print(f"[Eval] mAP@50:95 = {m5095:.4f}")

if __name__ == "__main__":
    main()
