#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/infer_to_csv_v7.py  ·  Kaggle 提交（精簡、安全、可重現）
- 與訓練配置對齊：讀 YAML + build_detector_from_cfg
- 嚴格輸出格式：PredictionString = "conf x y w h 0"
- 安全防呆：裁邊、過濾非法框、NMS、Top-K、四捨五入壓縮體積
- 影像排序：以檔名中的數字為主（容忍前導零）；無數字則回退字串序
"""

from pathlib import Path
import os, re, glob, csv, math
from typing import List, Tuple

import torch
import torchvision
from torchvision.ops import nms
from torchvision.transforms import functional as TF
from PIL import Image
from tqdm import tqdm

from config import load_cfg
from modelv7 import build_detector_from_cfg

# ========== 可調參數（集中於此，不需 CLI） ==========
# 設定檔與權重（建議覆寫為你實際路徑）
CFG_PATH = "experiments/configs/v7.yaml"
OVERRIDES = [
    # 權重位置（若 YAML 已設可留空或同名覆寫）
    "checkpoint.save_full_path=experiments/logs/fasterrcnn_v7/fasterrcnn_v7_best.pth",
    # 推論後處理（保守且通用）
    "infer.score_thr=0.25",
    "infer.nms_iou=0.50",
    "infer.postproc.topk_per_image=100",
]
# 單類別競賽固定輸出 0；若為多類別，請改成讀 labels 值
FIXED_CLASS_ID = 0
# 小數位數（壓縮 CSV 體積）
ROUND_CONF = 4
ROUND_COORD = 1
# ===============================================

_VALID_EXTS = ("*.jpg","*.jpeg","*.png","*.bmp","*.JPG","*.JPEG","*.PNG","*.BMP")
_NUM_RE = re.compile(r"(\d+)")

# ---------- 工具函式 ----------
def _numeric_id_from_path(path: str, fallback):
    """從檔名擷取數字（容忍前導零）；若無數字，回傳 fallback。"""
    stem = os.path.splitext(os.path.basename(path))[0]
    m = _NUM_RE.search(stem)
    if not m:
        return fallback
    s = m.group(1).lstrip("0")
    return int(s) if s != "" else 0

def _list_images_sorted(img_dir: str) -> List[Path]:
    p = Path(img_dir)
    files: List[Path] = []
    for pat in _VALID_EXTS:
        files.extend(p.glob(pat))
    # 先以數字排序，再以字串排序，確保穩定
    return sorted(files, key=lambda f: (_numeric_id_from_path(str(f), 10**12), f.stem))

def _xyxy_to_xywh(boxes: torch.Tensor) -> torch.Tensor:
    """(x1,y1,x2,y2) -> (x,y,w,h)"""
    xywh = boxes.clone()
    xywh[:, 2] = xywh[:, 2] - xywh[:, 0]
    xywh[:, 3] = xywh[:, 3] - xywh[:, 1]
    return xywh

def _clip_xyxy(x1, y1, x2, y2, W, H) -> Tuple[float,float,float,float]:
    x1 = max(0.0, min(float(x1), W))
    y1 = max(0.0, min(float(y1), H))
    x2 = max(0.0, min(float(x2), W))
    y2 = max(0.0, min(float(y2), H))
    return x1, y1, x2, y2

# ---------- 安全載入權重 ----------
def _safe_load_state(ckpt_path: Path) -> dict:
    """
    優先用 weights_only=True 載入，舊版 PyTorch 沒有此參數時自動回退。
    最終回傳 state_dict(dict)。若是 checkpoint 包裝，會自動取出 'state_dict' 或 'model'。
    """
    try:
        obj = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    except TypeError:
        obj = torch.load(str(ckpt_path), map_location="cpu")

    # 兼容常見儲存格式
    if isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            obj = obj["state_dict"]
        elif "model" in obj and isinstance(obj["model"], dict):
            obj = obj["model"]

    if not isinstance(obj, dict):
        raise RuntimeError("Checkpoint does not contain a valid state_dict dictionary.")
    return obj

# ---------- 主流程 ----------
@torch.inference_mode()
def main():
    torch.backends.cudnn.benchmark = True

    # 專案根路徑（src/ 的上一層）
    project_root = Path(__file__).resolve().parents[1]

    # 讀設定（允許覆寫）
    cfg = load_cfg(str(project_root / CFG_PATH), overrides=OVERRIDES)

    # 路徑與裝置
    ckpt_cfg = cfg.get("checkpoint", {})
    ckpt_path = ckpt_cfg.get("save_full_path") or (
        Path(ckpt_cfg.get("dir", "")) / ckpt_cfg.get("name", "")
    )
    ckpt_path = project_root / ckpt_path if not os.path.isabs(str(ckpt_path)) else Path(ckpt_path)

    test_dir = cfg["data"]["test_img_dir"]
    test_dir = project_root / test_dir if not os.path.isabs(test_dir) else Path(test_dir)

    out_csv = cfg.get("infer", {}).get("submission_csv", "submission_v7.csv")
    out_csv = project_root / out_csv if not os.path.isabs(out_csv) else Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("device", {}).get("cuda", True) else "cpu")

    # 檢查
    assert ckpt_path.exists(), f"[Err] 找不到權重檔：{ckpt_path}"
    assert test_dir.exists(), f"[Err] 找不到測試影像資料夾：{test_dir}"

    # 建模 + 載入權重
    model = build_detector_from_cfg(cfg).to(device).eval()
    state = _safe_load_state(ckpt_path)
    model.load_state_dict(state, strict=False)

    # 後處理參數
    score_thr = float(cfg.get("infer", {}).get("score_thr", 0.25))
    nms_iou   = float(cfg.get("infer", {}).get("nms_iou", 0.50))
    max_det   = int(cfg.get("infer", {}).get("postproc", {}).get("topk_per_image",
                   int(cfg.get("eval", {}).get("max_det", 100))))
    max_det   = max(1, min(300, max_det))  # 安全邊界

    # 影像清單
    img_files = _list_images_sorted(str(test_dir))
    assert len(img_files) > 0, f"[Err] 測試資料夾無影像：{test_dir}"

    print(f"[Infer] imgs={len(img_files)} | score_thr={score_thr} nms_iou={nms_iou} topK={max_det}")
    print(f"[CKPT ] {ckpt_path}")
    print(f"[OUT  ] {out_csv}")

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Image_ID", "PredictionString"])

        for rank_idx, fp in enumerate(tqdm(img_files, ncols=100, desc="Infer"), start=1):
            # Image_ID：優先用檔名數字（容忍前導零），無數字回退檔名
            image_id = _numeric_id_from_path(str(fp), fp.stem if fp.stem else rank_idx)

            im = Image.open(fp).convert("RGB")
            W, H = im.size
            img_tensor = TF.to_tensor(im).to(device)

            out = model([img_tensor])[0]
            boxes: torch.Tensor = out.get("boxes", torch.empty((0,4), device=device))
            scores: torch.Tensor = out.get("scores", torch.empty((0,), device=device))
            # 如需多類別，可讀 out["labels"] 用於 class_id；此作業預設單類別固定 0

            # Score 門檻
            if boxes.numel() > 0:
                keep = scores >= score_thr
                boxes, scores = boxes[keep], scores[keep]

            # NMS
            if boxes.numel() > 0:
                keep_idx = nms(boxes, scores, nms_iou)
                boxes, scores = boxes[keep_idx], scores[keep_idx]

            # Top-K 控制體積
            if boxes.shape[0] > max_det:
                topk = scores.topk(max_det).indices
                boxes, scores = boxes[topk], scores[topk]

            # 裁邊 + 去除非法框
            parts: List[str] = []
            if boxes.numel() > 0:
                b = boxes
                x1 = b[:, 0].clamp_(0, W); y1 = b[:, 1].clamp_(0, H)
                x2 = b[:, 2].clamp_(0, W); y2 = b[:, 3].clamp_(0, H)
                boxes = torch.stack([x1, y1, x2, y2], dim=1)

                # 去除 w/h <= 0、NaN
                xywh = _xyxy_to_xywh(boxes)
                wv = xywh[:, 2]; hv = xywh[:, 3]
                finite_mask = torch.isfinite(wv) & torch.isfinite(hv) & torch.isfinite(scores)
                pos_mask = (wv > 0) & (hv > 0)
                keep = finite_mask & pos_mask
                xywh, scores = xywh[keep], scores[keep]

                # 轉為 PredictionString：四捨五入壓縮體積
                if xywh.numel() > 0:
                    xywh_np = xywh.cpu().numpy()
                    scores_np = scores.cpu().numpy()
                    for (x, y, w_, h_), conf in zip(xywh_np, scores_np):
                        x1c, y1c, x2c, y2c = _clip_xyxy(x, y, x + w_, y + h_, W, H)
                        w2, h2 = x2c - x1c, y2c - y1c
                        if w2 <= 0 or h2 <= 0:
                            continue
                        conf_s = f"{float(conf):.{ROUND_CONF}f}"
                        x_s = f"{x1c:.{ROUND_COORD}f}"
                        y_s = f"{y1c:.{ROUND_COORD}f}"
                        w_s = f"{w2:.{ROUND_COORD}f}"
                        h_s = f"{h2:.{ROUND_COORD}f}"
                        parts.extend([conf_s, x_s, y_s, w_s, h_s, str(FIXED_CLASS_ID)])

            writer.writerow([image_id, " ".join(parts)])  # 若無偵測，留空字串即可

    print(f"[Done] CSV saved -> {out_csv}")

if __name__ == "__main__":
    main()
