#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/infer_to_csv.py
輸出 Kaggle 需要的:
Image_ID,PredictionString
每張圖都要有一列；<class> 固定 0；座標需反標準化回原圖空間。
"""

from pathlib import Path
import csv
import torch
import torchvision
from PIL import Image
from torchvision.ops import nms
from torchvision.transforms import functional as TF
from tqdm import tqdm

from config import load_cfg
from modelv4 import get_fasterrcnn_r50_fpn

# ========= 覆寫設定（可留空） =========
CFG_PATH = "experiments/configs/v5.yaml"
OVERRIDES = [
    #"checkpoint.save_full_path=experiments/logs/fasterrcnn_r50fpn_final_v2.pth",
    #"project.run_name=Test",
    #"project.run_name=fasterrcnn_v4_9_best",
]
# ====================================


def list_images_sorted(img_dir: str):
    p = Path(img_dir)
    files = [*p.glob("*.jpg"), *p.glob("*.jpeg"), *p.glob("*.png"), *p.glob("*.bmp")]
    # 以數字檔名優先排序，否則以字典序
    def _key(fp: Path):
        stem = fp.stem.lstrip("0")
        return (0, int(stem)) if stem.isdigit() else (1, fp.stem)
    return sorted(files, key=_key)


def resize_keep_max_side(img: Image.Image, max_side: int):
    """回傳 (resized_img, scale)。scale = resized / original"""
    w, h = img.size
    m = max(w, h)
    if m <= max_side:
        return img, 1.0
    s = float(max_side) / m
    new_w, new_h = int(round(w * s)), int(round(h * s))
    return img.resize((new_w, new_h), Image.BILINEAR), s


def xyxy_to_xywh(boxes: torch.Tensor) -> torch.Tensor:
    xywh = boxes.clone()
    xywh[:, 2] = xywh[:, 2] - xywh[:, 0]
    xywh[:, 3] = xywh[:, 3] - xywh[:, 1]
    return xywh


def _state_to_fp32(state):
    for k, v in list(state.items()):
        if isinstance(v, torch.Tensor) and v.is_floating_point() and v.dtype == torch.float16:
            state[k] = v.float()
    return state


@torch.inference_mode()
def main():
    # 讀設定
    project_root = Path(__file__).resolve().parents[1]
    cfg = load_cfg(str(project_root / CFG_PATH), overrides=OVERRIDES)

    ckpt_cfg = cfg["checkpoint"]
    ckpt_path = Path(ckpt_cfg.get("save_full_path") or (Path(ckpt_cfg["dir"]) / ckpt_cfg["name"]))
    test_dir = Path(cfg["data"]["test_img_dir"])
    out_csv = Path(cfg["infer"]["submission_csv"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    assert ckpt_path.exists(), f"找不到權重檔：{ckpt_path}"
    assert test_dir.exists(), "找不到測試影像資料夾"

    device = torch.device("cuda" if torch.cuda.is_available() and cfg["device"]["cuda"] else "cpu")

    # 模型
    model = get_fasterrcnn_r50_fpn(
        num_classes=int(cfg["model"]["num_classes"]),   # 你 YAML 已含背景（=2）
        freeze_backbone=bool(cfg["model"]["freeze_backbone"])
    ).to(device)
    try:
        state = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(str(ckpt_path), map_location="cpu")
    model.load_state_dict(_state_to_fp32(state), strict=True)
    model.eval()

    # 推論設定
    score_thr = float(cfg["infer"]["score_thr"])
    nms_iou   = float(cfg["infer"]["nms_iou"])
    max_side  = int(cfg["augment"]["max_side"])
    max_det   = int(cfg.get("eval", {}).get("max_det", 100))

    imgs = list_images_sorted(str(test_dir))
    assert len(imgs) > 0, "測試資料夾沒有影像"

    print(f"[Infer] images={len(imgs)}  thr={score_thr}  nms={nms_iou}  max_side={max_side}  max_det={max_det}")
    print(f"[CKPT ] {ckpt_path}")
    print(f"[OUT  ] {out_csv}")

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Image_ID", "PredictionString"])

        for img_id, fp in tqdm(list(enumerate(imgs, start=1)), ncols=100, desc="Infer"):
            pil = Image.open(fp).convert("RGB")
            W0, H0 = pil.size
            resized, scale = resize_keep_max_side(pil, max_side)
            tensor = TF.to_tensor(resized).to(device)

            out = model([tensor])[0]
            boxes  = out["boxes"]      # xyxy on resized space
            scores = out["scores"]
            labels = out["labels"]

            # 保留前景類（label==1），並做 NMS + threshold
            keep = (labels == 1) & (scores >= score_thr)
            boxes, scores = boxes[keep], scores[keep]

            if boxes.numel() > 0:
                keep = nms(boxes, scores, nms_iou)
                boxes, scores = boxes[keep], scores[keep]

            # 反標準化回原圖座標
            if scale != 1.0 and boxes.numel() > 0:
                boxes = boxes / float(scale)

            # clip 到原圖邊界
            if boxes.numel() > 0:
                x1 = boxes[:, 0].clamp_(0, W0 - 1)
                y1 = boxes[:, 1].clamp_(0, H0 - 1)
                x2 = boxes[:, 2].clamp_(0, W0 - 1)
                y2 = boxes[:, 3].clamp_(0, H0 - 1)
                boxes = torch.stack([x1, y1, x2, y2], dim=1)

            # 轉 xywh、取 top-k（≤100）
            if boxes.numel() > 0:
                boxes = xyxy_to_xywh(boxes)
                if boxes.shape[0] > max_det:
                    topk = torch.topk(scores, k=max_det)
                    boxes, scores = boxes[topk.indices], topk.values

            # 組 PredictionString（整數像素更保險；class 固定 0）
            parts = []
            if boxes.numel() > 0:
                boxes = boxes.cpu().numpy()
                scores = scores.cpu().numpy()
                for (x, y, w_, h_), conf in zip(boxes, scores):
                    if w_ <= 0 or h_ <= 0:
                        continue
                    parts += [f"{conf:.4f}", f"{int(round(x))}", f"{int(round(y))}",
                              f"{int(round(w_))}", f"{int(round(h_))}", "0"]

            predstr = " ".join(parts)  # 無檢出時為空字串
            w.writerow([img_id, predstr])

    print(f"[Done] CSV saved -> {out_csv}")
    

if __name__ == "__main__":
    main()
