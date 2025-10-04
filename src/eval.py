#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval.py
在驗證集 (data/val/img + data/val/gt_val.txt) 上計算 mAP@50 與 mAP@50:95，
並把「所有」 precision / recall 結果輸出到 experiments/ 目錄。

不需要 COCO JSON，直接讀 gt_val.txt（格式: frame,x,y,w,h）。
"""

import os, json, glob
from typing import List, Dict, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torchvision
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from model import get_fasterrcnn_r50_fpn


# ========= 可自行修改的設定 =========
CKPT_PATH   = "experiments/logs/best_model.pth"        # 權重
VAL_IMG_DIR = "data/val/img"                                     # 驗證影像資料夾
VAL_GT_TXT  = "data/val/gt_val.txt"                              # 驗證標註 txt
MAX_SIDE    = 800                                                # 評估時最長邊縮放
SCORE_THR   = 0.05                                               # 篩選分數門檻
OUT_TXT     = "experiments/eval_results/eval_results_e5.txt"     # 人類可讀摘要
OUT_JSON    = "experiments/eval_results/eval_details_e5.json"    # 完整陣列（精確）
# ===================================


# ---------- 工具：讀取/建立 COCO GT ----------
def _read_gt_txt(gt_txt: str) -> List[Tuple[int, int, int, int, int]]:
    """讀取 gt_val.txt -> [(img_id, x, y, w, h), ...]"""
    out = []
    with open(gt_txt, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 5:
                continue
            frame, l, t, w, h = parts
            img_id = int(float(frame))
            l, t, w, h = map(int, (l, t, w, h))
            if w <= 0 or h <= 0:
                continue
            out.append((img_id, l, t, w, h))
    return out


def build_coco_from_gt(gt_txt: str, img_dir: str) -> COCO:
    """
    由 gt_val.txt 動態建立 COCO 資料結構（無需 JSON 檔）。
    - 會讀取每張影像尺寸 (PIL)
    """
    rows = _read_gt_txt(gt_txt)
    images_map: Dict[int, Dict] = {}
    annotations: List[Dict] = []

    ann_id = 1
    for (img_id, l, t, w, h) in rows:
        file_name = f"{img_id:08d}.jpg"  # 依你的命名規則
        if img_id not in images_map:
            fp = os.path.join(img_dir, file_name)
            if not os.path.isfile(fp):
                hits = glob.glob(os.path.join(img_dir, f"{img_id:08d}.*"))
                if not hits:
                    # 找不到影像就略過該影像的標註
                    continue
                fp = hits[0]
                file_name = os.path.basename(fp)
            with Image.open(fp) as im:
                W, H = im.size
            images_map[img_id] = {"id": img_id, "file_name": file_name, "width": W, "height": H}

        annotations.append({
            "id": ann_id,
            "image_id": img_id,
            "category_id": 1,            # 單類別：pig
            "bbox": [l, t, w, h],        # COCO: [x,y,w,h]
            "area": w * h,
            "iscrowd": 0
        })
        ann_id += 1

    coco_dict = {
        "info": {"description": "TAICA HW1 - Validation (from gt_val.txt)"},
        "licenses": [],
        "images": list(images_map.values()),
        "annotations": annotations,
        "categories": [{"id": 1, "name": "pig"}],
    }

    coco_gt = COCO()
    coco_gt.dataset = coco_dict
    coco_gt.createIndex()
    return coco_gt


# ---------- 模型 ----------
def load_model(ckpt_path: str, device: torch.device):
    model = get_fasterrcnn_r50_fpn(num_classes=2, freeze_backbone=True).to(device)
    # 安全載入權重（新舊 torch 皆可）
    try:
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


# ---------- 推論並轉成 COCO DT 清單 ----------
@torch.inference_mode()
def infer_to_coco_dt(model, coco_gt: COCO, img_dir: str, device: torch.device,
                     max_side: int = 800, score_thr: float = 0.05):
    tfm = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    img_ids = sorted(coco_gt.getImgIds())
    results = []

    for img_id in tqdm(img_ids, desc="Infer", ncols=100):
        info = coco_gt.loadImgs(img_id)[0]
        file_name = info["file_name"]
        fp = os.path.join(img_dir, file_name)
        if not os.path.isfile(fp):
            hits = glob.glob(os.path.join(img_dir, os.path.splitext(file_name)[0] + ".*"))
            if not hits:
                continue
            fp = hits[0]

        img = Image.open(fp).convert("RGB")
        W, H = img.size
        scale = min(1.0, float(max_side) / max(W, H))
        if scale < 1.0:
            new_w, new_h = int(round(W * scale)), int(round(H * scale))
            img = img.resize((new_w, new_h), resample=Image.BILINEAR)

        x = tfm(img).to(device).unsqueeze(0)
        out = model(x)[0]

        boxes = out["boxes"].detach().cpu().numpy()
        scores = out["scores"].detach().cpu().numpy()

        # 還原座標
        if scale < 1.0:
            inv = 1.0 / scale
            boxes[:, [0, 2]] *= inv
            boxes[:, [1, 3]] *= inv

        # 過濾分數
        keep = scores >= score_thr
        boxes, scores = boxes[keep], scores[keep]

        # 轉 COCO bbox
        for b, s in zip(boxes, scores):
            x1, y1, x2, y2 = [float(v) for v in b]
            w_box, h_box = max(0.0, x2 - x1), max(0.0, y2 - y1)
            if w_box <= 0.0 or h_box <= 0.0:
                continue
            results.append({
                "image_id": img_id,
                "category_id": 1,
                "bbox": [x1, y1, w_box, h_box],
                "score": float(s),
            })

    return results


# ---------- 將 COCOeval 結果完整寫檔 ----------
def dump_cocoeval(coco_eval: COCOeval, out_txt: str, out_json: str):
    os.makedirs(os.path.dirname(out_txt), exist_ok=True)

    # 1) 人類可讀摘要
    with open(out_txt, "w") as f:
        f.write("===== COCO Evaluation Summary =====\n\n")
        metrics = [
            "AP @[ IoU=0.50:0.95 | area=all | maxDets=100 ]",
            "AP @[ IoU=0.50      | area=all | maxDets=100 ]",
            "AP @[ IoU=0.75      | area=all | maxDets=100 ]",
            "AP @[ IoU=0.50:0.95 | area=small | maxDets=100 ]",
            "AP @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]",
            "AP @[ IoU=0.50:0.95 | area=large | maxDets=100 ]",
            "AR @[ IoU=0.50:0.95 | area=all | maxDets=1 ]",
            "AR @[ IoU=0.50:0.95 | area=all | maxDets=10 ]",
            "AR @[ IoU=0.50:0.95 | area=all | maxDets=100 ]",
            "AR @[ IoU=0.50:0.95 | area=small | maxDets=100 ]",
            "AR @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]",
            "AR @[ IoU=0.50:0.95 | area=large | maxDets=100 ]",
        ]
        for i, m in enumerate(metrics):
            f.write(f"{m:<70} = {coco_eval.stats[i]:.6f}\n")

        f.write("\n[說明]\n")
        f.write("- AP: 平均精度；AR: 平均召回；主指標為 AP@[0.50:0.95]（COCO 標準）。\n")
        f.write("- precision 的維度為 [IoU x recall x category x area x maxDets]。\n")
        f.write("- recall    的維度為 [IoU x category x area x maxDets]（每個 IoU 的最大可達召回）。\n")
        f.write("- 詳細陣列請見 eval_details.json（可用來畫曲線）。\n")

    # 2) 遞迴轉換為原生 Python 型別，方便 JSON 序列化
    def _to_py(obj):
        """把 numpy/tensor/容器遞迴轉成原生 Python（list/float/int/None 等）。"""
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj

        # numpy
        try:
            import numpy as _np
            if isinstance(obj, _np.generic):
                return obj.item()
            if isinstance(obj, _np.ndarray):
                return obj.tolist()
        except Exception:
            pass

        # torch
        try:
            import torch as _torch
            if isinstance(obj, _torch.Tensor):
                return obj.detach().cpu().tolist()
        except Exception:
            pass

        # 容器
        if isinstance(obj, (list, tuple, set)):
            return [_to_py(x) for x in obj]
        if isinstance(obj, dict):
            return {k: _to_py(v) for k, v in obj.items()}

        # 其他支援 tolist 的
        if hasattr(obj, "tolist"):
            try:
                return obj.tolist()
            except Exception:
                pass

        # fallback
        try:
            return float(obj)
        except Exception:
            try:
                return int(obj)
            except Exception:
                return str(obj)

    details = {
        "precision": _to_py(coco_eval.eval.get("precision")),  # [T,R,K,A,M]
        "recall":    _to_py(coco_eval.eval.get("recall")),     # [T,K,A,M]
        "scores":    _to_py(coco_eval.eval.get("scores")),     # [T,R,K,A,M]
        "params": {
            "iouThrs":    _to_py(coco_eval.params.iouThrs),    # len T
            "recThrs":    _to_py(coco_eval.params.recThrs),    # len R
            "catIds":     _to_py(coco_eval.params.catIds),     # 我們只有一類
            "areaRngLbl": _to_py(coco_eval.params.areaRngLbl), # ["all","small","medium","large"]
            "maxDets":    _to_py(coco_eval.params.maxDets),    # [1,10,100]
        },
    }

    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(details, f, indent=2)

    print(f"[Done] Results saved to:\n  - {out_txt}\n  - {out_json}")



# ---------- 主流程 ----------
def main():
    # 檢查路徑
    if not os.path.isfile(CKPT_PATH):
        raise FileNotFoundError(CKPT_PATH)
    if not os.path.isdir(VAL_IMG_DIR):
        raise FileNotFoundError(VAL_IMG_DIR)
    if not os.path.isfile(VAL_GT_TXT):
        raise FileNotFoundError(VAL_GT_TXT)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    # 1) 準備 GT
    coco_gt = build_coco_from_gt(VAL_GT_TXT, VAL_IMG_DIR)

    # 2) 載入模型
    model = load_model(CKPT_PATH, device)

    # 3) 推論 → COCO DT
    coco_dt_list = infer_to_coco_dt(model, coco_gt, VAL_IMG_DIR, device,
                                    max_side=MAX_SIDE, score_thr=SCORE_THR)

    # 4) COCO 評估
    coco_dt = coco_gt.loadRes(coco_dt_list)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()  # 會印在終端機

    # 5) 存檔（摘要 + 完整陣列）
    os.makedirs(os.path.dirname(OUT_TXT), exist_ok=True)
    dump_cocoeval(coco_eval, OUT_TXT, OUT_JSON)


if __name__ == "__main__":
    main()
