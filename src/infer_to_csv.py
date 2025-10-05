#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/infer_to_csv.py
---------------------------------
依 Kaggle submission 格式輸出：
Image_ID,PredictionString
1,<conf_1> <x_1> <y_1> <w_1> <h_1> <class_1> <conf_2> <x_2> <y_2> <w_2> <h_2> <class_2> ...
"""

import os
from pathlib import Path
import glob
import pandas as pd

import torch
import torchvision
from PIL import Image
from tqdm import tqdm

from config import load_cfg
from model import get_fasterrcnn_r50_fpn


# ========= 可在這裡快速覆寫設定（可留空） =========
CFG_PATH = "experiments/configs/default.yaml"
OVERRIDES = [
    # "infer.score_thr=0.05",
    # "infer.nms_iou=0.5",
    # "checkpoint.name=fasterrcnn_v3.pth",
    # "infer.submission_csv=submissions/fasterrcnn_v3_submission.csv",
]
# =================================================


def list_images(img_dir: str):
    """列出所有影像檔（依檔名排序，支援多格式）"""
    img_dir = Path(img_dir)
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    files = []
    for ext in exts:
        files.extend(img_dir.glob(ext))
    def _key(fp: Path):
        try:
            return int(fp.stem)
        except ValueError:
            return fp.stem
    return sorted(files, key=_key)


def resize_keep_max_side(pil_img: Image.Image, max_side: int) -> Image.Image:
    """等比縮放：讓最長邊 = max_side；若原圖已小於 max_side 就不放大"""
    w, h = pil_img.size
    if max(w, h) <= max_side:
        return pil_img
    scale = float(max_side) / max(w, h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    return pil_img.resize((new_w, new_h), resample=Image.BILINEAR)


def xyxy_to_xywh(boxes: torch.Tensor) -> torch.Tensor:
    """(x1,y1,x2,y2) -> (x,y,w,h)"""
    boxes = boxes.clone()
    boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
    boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
    return boxes


@torch.inference_mode()
def main():
    # === 讀取設定 ===
    project_root = Path(__file__).resolve().parents[1]
    cfg = load_cfg(str(project_root / CFG_PATH), overrides=OVERRIDES)

    ckpt_path = Path(cfg["checkpoint"]["dir"]) / cfg["checkpoint"]["name"]
    img_dir = Path(cfg["data"]["test_img_dir"])
    out_csv = Path(cfg["infer"]["submission_csv"])
    assert ckpt_path.exists(), f"找不到權重檔：{ckpt_path}"
    assert img_dir.exists(), f"找不到測試影像資料夾：{img_dir}"

    device = torch.device("cuda" if torch.cuda.is_available() and cfg["device"]["cuda"] else "cpu")

    # === 載入模型 ===
    model = get_fasterrcnn_r50_fpn(
        num_classes=cfg["model"]["num_classes"],
        freeze_backbone=cfg["model"]["freeze_backbone"]
    ).to(device)
    try:
        state = torch.load(str(ckpt_path), map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(state)
    model.eval()

    # === 推論設定 ===
    score_thr = float(cfg["infer"]["score_thr"])
    nms_iou   = float(cfg["infer"]["nms_iou"])
    max_side  = int(cfg["augment"]["max_side"])

    img_files = list_images(str(img_dir))
    if len(img_files) == 0:
        raise FileNotFoundError(f"未找到任何影像於 {img_dir}")

    to_tensor = torchvision.transforms.ToTensor()
    results = []

    print(f"[Infer] Using {len(img_files)} images from {img_dir}")
    print(f"[Infer] Score threshold={score_thr}, NMS IoU={nms_iou}, max_side={max_side}")
    print(f"[Infer] Output CSV: {out_csv}")

    for fp in tqdm(img_files, ncols=100, desc="Infer"):
        image_id = int(fp.stem)
        img = Image.open(fp).convert("RGB")
        img = resize_keep_max_side(img, max_side=max_side)
        x = to_tensor(img).to(device).unsqueeze(0)

        out = model(x)[0]
        boxes  = out["boxes"].detach().cpu()
        scores = out["scores"].detach().cpu()

        # 閾值過濾
        keep = scores >= score_thr
        boxes, scores = boxes[keep], scores[keep]

        # NMS
        if len(boxes) > 0:
            keep_idx = torchvision.ops.nms(boxes, scores, nms_iou)
            boxes, scores = boxes[keep_idx], scores[keep_idx]

        boxes_xywh = xyxy_to_xywh(boxes)
        parts = []
        for b, s in zip(boxes_xywh.tolist(), scores.tolist()):
            x1, y1, w, h = [round(float(v), 2) for v in b]
            conf = round(float(s), 4)
            cls = 0  # pigs 固定 class=0
            # 按規範順序：<conf> <bb_left> <bb_top> <bb_width> <bb_height> <class>
            parts.extend([conf, x1, y1, w, h, cls])
        pred_str = " ".join(map(str, parts))
        results.append([image_id, pred_str])

    # === 寫出 CSV（Kaggle submission 格式） ===
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results, columns=["Image_ID", "PredictionString"])
    df.to_csv(out_csv, index=False)
    print(f"[Done] 已輸出 submission：{out_csv} ({len(results)} 張影像)")


if __name__ == "__main__":
    main()
