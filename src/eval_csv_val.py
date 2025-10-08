#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/eval_csv_val.py
- 讀取 gt_val.txt（frame_id,x,y,w,h）與 val_pred_diag.csv（Image_ID,PredictionString）
- 轉為 COCO 格式並用 pycocotools 計分（bbox mAP）
- 假設 val 影像命名為零補數字，如 000123.jpg；若不同，調整 FMT/ID 轉換處
"""

from pathlib import Path
import json, csv
from collections import defaultdict
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# ===== 可調參數 =====
GT_FILE = Path("data/val/gt_val.txt")                  # <frame_id>,<x>,<y>,<w>,<h>
IMG_DIR = Path("data/val/img")
PRED_CSV = Path("experiments/eval_results/val_pred_diag.csv")
IMG_EXT = ".jpg"                                       # 依你的資料調整（.jpg/.png）

# 若你的檔名長度不是 6 位，改這裡
ZERO_PAD = 6

def _int_id_from_str(s: str) -> int:
    # 將 "000123" → 123；全是 0 則回 0
    s2 = s.lstrip("0")
    return int(s2) if s2 != "" else 0

def build_coco_gt():
    # 收集所有 val 影像的寬高（依檔名 000123.jpg）
    wh_cache = {}
    for p in IMG_DIR.iterdir():
        if p.suffix.lower() in (".jpg",".jpeg",".png",".bmp"):
            try:
                with Image.open(p) as im:
                    w, h = im.size
                wh_cache[p.name] = (w, h)
            except Exception:
                pass

    images = []
    annotations = []
    categories = [{"id": 1, "name": "pig"}]
    ann_id = 1
    seen_img_ids = set()

    for line in GT_FILE.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) < 5:
            continue
        frame_id, x, y, w, h = parts[:5]
        img_id = _int_id_from_str(frame_id)
        file_name = f"{img_id:0{ZERO_PAD}d}{IMG_EXT}"

        if img_id not in seen_img_ids:
            W, H = wh_cache.get(file_name, (None, None))
            images.append({
                "id": img_id,
                "file_name": file_name,
                "width": W,
                "height": H
            })
            seen_img_ids.add(img_id)

        annotations.append({
            "id": ann_id,
            "image_id": img_id,
            "category_id": 1,
            "bbox": [float(x), float(y), float(w), float(h)],
            "area": float(w) * float(h),
            "iscrowd": 0
        })
        ann_id += 1

    coco_gt = {
        "info": {"description": "VAL GT for pig detection", "version": "1.0"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    tmp = Path("experiments/eval_results/_tmp_gt.json")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(coco_gt))
    return COCO(str(tmp))

def load_pred_as_coco_json():
    preds = []
    with open(PRED_CSV, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id_str = row["Image_ID"].strip()
            image_id = _int_id_from_str(image_id_str)
            ps = row["PredictionString"].strip()
            if ps == "":
                continue
            nums = [float(x) for x in ps.split()]
            if len(nums) % 6 != 0:
                raise ValueError(f"PredictionString length not multiple of 6 for Image_ID={image_id_str}")
            for i in range(0, len(nums), 6):
                score, x, y, w, h, _cls = nums[i:i+6]
                # 類別固定 1（pig）；若你之後多類別再調整
                preds.append({
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [x, y, w, h],
                    "score": score
                })
    tmp = Path("experiments/eval_results/_tmp_pred.json")
    tmp.write_text(json.dumps(preds))
    return tmp

def main():
    coco_gt = build_coco_gt()
    pred_json = load_pred_as_coco_json()
    coco_dt = coco_gt.loadRes(str(pred_json))
    ev = COCOeval(coco_gt, coco_dt, "bbox")
    ev.evaluate(); ev.accumulate(); ev.summarize()

if __name__ == "__main__":
    main()
