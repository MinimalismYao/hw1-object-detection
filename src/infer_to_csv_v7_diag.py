#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/infer_to_csv_v7_diag.py
— 用來排除 Kaggle 低分的常見地雷（ID/格式/後處理/座標/變形不一致）。
— 會先在 val 集產 CSV 做煙霧測試，再在 test 產 Kaggle 用 CSV。
"""

from pathlib import Path
import csv, math
from typing import List
import torch
from PIL import Image
from torchvision.transforms import functional as TF
from torchvision.ops import nms as hard_nms

# ===== 你只要檔頭這三個路徑對一下 =====
CFG_PATH = "experiments/configs/v7.yaml"
CKPT_PATH = "experiments/logs/fasterrcnn_v7/fasterrcnn_v7_best.pth"
IMG_DIR = Path("data/test/img")                  # → 這裡是要產 Kaggle CSV 的資料夾
OUT_CSV = Path("experiments/eval_results/submission_diag.csv")

# ===== 推論後處理（先保守）=====
SCORE_THRESH = 0.05
NMS_IOU = 0.6
TOPK = 300

# ===== 煙霧測試（先在 val 集做一次 CSV→評分）=====
SMOKE_TEST_ON_VAL = True
VAL_IMG_DIR = Path("data/val/img")
VAL_CSV = Path("experiments/eval_results/val_pred_diag.csv")

def _xyxy_to_xywh(boxes, W, H):
    x1 = boxes[:, 0].clamp(0, W - 1)
    y1 = boxes[:, 1].clamp(0, H - 1)
    x2 = boxes[:, 2].clamp(0, W - 1)
    y2 = boxes[:, 3].clamp(0, H - 1)
    w = (x2 - x1).clamp(min=1.0)
    h = (y2 - y1).clamp(min=1.0)
    return torch.stack([x1, y1, w, h], dim=1)

def _format_row(image_id: str, det: torch.Tensor):
    # det: [N, 5] => score, x, y, w, h
    if det.numel() == 0:
        return [image_id, ""]
    parts: List[str] = []
    for i in range(det.shape[0]):
        s, x, y, w, h = det[i].tolist()
        if not (math.isfinite(s) and math.isfinite(x) and math.isfinite(y) and math.isfinite(w) and math.isfinite(h)):
            continue
        if w <= 0 or h <= 0:
            continue
        parts.append(f"{s:.6f} {x:.1f} {y:.1f} {w:.1f} {h:.1f} 0")
    return [image_id, " ".join(parts)]

@torch.inference_mode()
def _run(model, img_dir: Path, out_csv: Path):
    paths = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in (".jpg",".jpeg",".png",".bmp")])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Image_ID", "PredictionString"])
        for p in paths:
            im = Image.open(p).convert("RGB")
            W, H = im.size
            x = TF.to_tensor(im).cuda(non_blocking=True)
            out = model([x])[0]
            scores = out["scores"].detach().float().cuda()
            boxes  = out["boxes"].detach().float().cuda()
            keep = scores >= SCORE_THRESH
            scores, boxes = scores[keep], boxes[keep]
            if boxes.numel() > 0:
                keep_idx = hard_nms(boxes, scores, iou_threshold=NMS_IOU)
                scores, boxes = scores[keep_idx], boxes[keep_idx]
                if scores.numel() > TOPK:
                    topk_idx = torch.topk(scores, TOPK).indices
                    scores, boxes = scores[topk_idx], boxes[topk_idx]
                boxes_xywh = _xyxy_to_xywh(boxes, W, H)
                det = torch.cat([scores.view(-1,1), boxes_xywh], dim=1).cpu()
            else:
                det = torch.zeros((0,5), dtype=torch.float32)
            image_id = p.stem  # ← 原檔名（去副檔名），不要動前導零
            w.writerow(_format_row(image_id, det))
    print(f"[OK] wrote {out_csv} with {len(paths)} rows.")

def main():
    from config import load_cfg
    cfg = load_cfg(CFG_PATH)
    from modelv7 import build_detector_from_cfg
    model = build_detector_from_cfg(cfg).cuda()

    # 載入權重（可忽略 FutureWarning；我們不是載不可信檔案）
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    model.load_state_dict(ckpt)

    # ★ 關鍵：推論一定要 eval 模式
    model.eval()

    # 以防 YAML 有不同門檻，這裡再覆寫一次（若該屬性存在）
    if hasattr(model, "roi_heads") and hasattr(model.roi_heads, "score_thresh"):
        model.roi_heads.score_thresh = SCORE_THRESH

    # 保險檢查：若還是 training 就直接中斷，避免再次踩雷
    assert not model.training, "Model is still in training mode; call model.eval() before inference."

    if SMOKE_TEST_ON_VAL:
        _run(model, VAL_IMG_DIR, VAL_CSV)
        print("[SMOKE] 已輸出 val_pred_diag.csv，接著跑：python src/eval_csv_val.py")

    _run(model, IMG_DIR, OUT_CSV)


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()
