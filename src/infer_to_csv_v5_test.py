#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/infer_to_csv_v5_test.py
Faster R-CNN 推論（v5） + 後處理強化（A提案落地版）：
- Temperature Scaling（可選）
- 尺度感知：small/medium/large 各自 score 門檻與 NMS
- 反標準化、邊界裁切、去除不合法框
- 逐圖 Top-K
輸出 Kaggle 需要的 CSV：
Image_ID,PredictionString
"""

from pathlib import Path
import csv
import torch
import torchvision
from PIL import Image
from torchvision.transforms import functional as TF
from torchvision.ops import nms as hard_nms
from tqdm import tqdm

from config import load_cfg
from modelv4 import get_fasterrcnn_r50_fpn


# ========= 可在這裡快速調整的區域（不需 CLI） =========
CFG_PATH = "experiments/configs/v5.yaml"
OVERRIDES = [
    # 範例覆寫（可留空）
    # "checkpoint.save_full_path=experiments/logs/fasterrcnn_v5.2/fasterrcnn_v5.2_best.pth",
    # "infer.submission_csv=submissions/fasterrcnn_v5.2_submission.csv",
    "infer.score_thr=0.30",   # base threshold（會配合 small/large 產生加嚴）
    "infer.nms_iou=0.60",     # 基礎 NMS IoU（中型用），小/大會各自調整
    "eval.max_det=100",       # 每張圖最多保留
]

# —— Temperature Scaling（可關閉） ——
USE_TEMPERATURE = True
TEMPERATURE = 2.0            # logit / T 後再 sigmoid，T>1 讓分數更保守

# —— 面積分組閾值（在「原圖空間」判定小/中/大） ——
SMALL_PIX = 32 * 32          # area <= 32^2 → small
MED_PIX   = 96 * 96          # small < area <= 96^2 → medium，否則 large

# —— 尺度感知門檻與 NMS（可依需要微調） ——
DELTA_SMALL_THR = 0.10       # small 的 score_thr = base + 0.10
DELTA_LARGE_THR = 0.05       # large 的 score_thr = base + 0.05
SMALL_NMS = 0.50             # small 用的 NMS IoU
MED_NMS   = None             # None 代表用 infer.nms_iou
LARGE_NMS = 0.70             # large 用的 NMS IoU

# —— 其他後處理 ——
MIN_BOX_WH = 2               # 小於此寬或高直接丟棄（像素）
# =====================================================


def list_images_sorted(img_dir: str):
    p = Path(img_dir)
    files = [*p.glob("*.jpg"), *p.glob("*.jpeg"), *p.glob("*.png"), *p.glob("*.bmp")]
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
    xywh[:, 2] -= xywh[:, 0]
    xywh[:, 3] -= xywh[:, 1]
    return xywh


def _state_to_fp32(state):
    for k, v in list(state.items()):
        if isinstance(v, torch.Tensor) and v.is_floating_point() and v.dtype == torch.float16:
            state[k] = v.float()
    return state


def _temperature_scale(scores: torch.Tensor, T: float) -> torch.Tensor:
    if not USE_TEMPERATURE or T is None or T <= 0:
        return scores
    eps = 1e-6
    s = scores.clamp(min=eps, max=1 - eps)
    logit = torch.log(s / (1 - s))
    return torch.sigmoid(logit / T)


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

    # 建模
    model = get_fasterrcnn_r50_fpn(
        num_classes=int(cfg["model"]["num_classes"]),          # YAML 已含背景（=2）
        freeze_backbone=bool(cfg["model"]["freeze_backbone"])
    ).to(device)
    try:
        state = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(str(ckpt_path), map_location="cpu")
    model.load_state_dict(_state_to_fp32(state), strict=True)
    model.eval()

    # 推論參數
    base_thr = float(cfg["infer"]["score_thr"])
    med_nms = float(cfg["infer"]["nms_iou"]) if MED_NMS is None else MED_NMS
    max_side = int(cfg["augment"]["max_side"])
    max_det  = int(cfg.get("eval", {}).get("max_det", 100))

    imgs = list_images_sorted(str(test_dir))
    assert len(imgs) > 0, "測試資料夾沒有影像"

    print(f"[Infer] images={len(imgs)} | base_thr={base_thr} | NMS(S/M/L)=({SMALL_NMS}/{med_nms}/{LARGE_NMS}) "
          f"| T={TEMPERATURE if USE_TEMPERATURE else 'off'} | topk={max_det}")
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

            # 1) 只保留前景類
            keep = (labels == 1)
            boxes, scores = boxes[keep], scores[keep]

            if boxes.numel() == 0:
                w.writerow([img_id, ""])
                continue

            # 2) 溫度縮放（更保守）
            scores = _temperature_scale(scores, TEMPERATURE)

            # 3) 反標準化回「原圖」空間
            if scale != 1.0:
                boxes = boxes / float(scale)

            # 4) 邊界裁切 + 去除過小框
            x1 = boxes[:, 0].clamp_(0, W0 - 1)
            y1 = boxes[:, 1].clamp_(0, H0 - 1)
            x2 = boxes[:, 2].clamp_(0, W0 - 1)
            y2 = boxes[:, 3].clamp_(0, H0 - 1)
            boxes = torch.stack([x1, y1, x2, y2], dim=1)

            # 5) 以「原圖空間」面積分組 → 各自門檻 + 各自 NMS
            w_box = (boxes[:, 2] - boxes[:, 0]).clamp(min=0)
            h_box = (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
            area  = w_box * h_box

            small_mask = area <= SMALL_PIX
            med_mask   = (area > SMALL_PIX) & (area <= MED_PIX)
            large_mask = area > MED_PIX

            def filt(bmask, thr, iou):
                if bmask.sum() == 0:
                    return torch.empty((0,4), device=boxes.device), torch.empty((0,), device=scores.device)
                b, s = boxes[bmask], scores[bmask]
                # score 過濾
                keep_ = s >= thr
                b, s = b[keep_], s[keep_]
                if b.numel() > 0:
                    idx = hard_nms(b, s, iou)
                    b, s = b[idx], s[idx]
                return b, s

            thr_s = base_thr + DELTA_SMALL_THR
            thr_m = base_thr
            thr_l = base_thr + DELTA_LARGE_THR

            b_s, s_s = filt(small_mask, thr_s, SMALL_NMS)
            b_m, s_m = filt(med_mask,   thr_m, med_nms)
            b_l, s_l = filt(large_mask, thr_l, LARGE_NMS)

            # 6) 合併三組
            if b_s.numel() + b_m.numel() + b_l.numel() == 0:
                w.writerow([img_id, ""])
                continue
            boxes_all  = torch.cat([b_s, b_m, b_l], 0)
            scores_all = torch.cat([s_s, s_m, s_l], 0)

            # 7) 去除過小框（再次保險）
            xywh = xyxy_to_xywh(boxes_all)
            keep = (xywh[:, 2] > MIN_BOX_WH) & (xywh[:, 3] > MIN_BOX_WH)
            boxes_all, scores_all, xywh = boxes_all[keep], scores_all[keep], xywh[keep]

            if boxes_all.numel() == 0:
                w.writerow([img_id, ""])
                continue

            # 8) 全域 Top-K（≤ max_det）
            if xywh.shape[0] > max_det:
                topk = torch.topk(scores_all, k=max_det)
                xywh, scores_all = xywh[topk.indices], topk.values

            # 9) 組 PredictionString（class 固定 0，整數像素）
            parts = []
            b_np = xywh.cpu().numpy()
            s_np = scores_all.cpu().numpy()
            for (x, y, w_, h_), conf in zip(b_np, s_np):
                parts += [f"{conf:.4f}", f"{int(round(x))}", f"{int(round(y))}",
                          f"{int(round(w_))}", f"{int(round(h_))}", "0"]

            w.writerow([img_id, " ".join(parts)])

    print(f"[Done] CSV saved -> {out_csv}")


if __name__ == "__main__":
    main()
