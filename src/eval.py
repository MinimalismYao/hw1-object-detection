#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/eval.py
---------------------------------
單類別（class=0=pig）COCO 風格指標（AP/AR）並以 COCO 官方摘要格式輸出。
只讀 data/val/img 與 data/val/gt_val.txt。

特性：
- ✅ 與訓練「同一份 YAML」建模（anchors/min-max-size/NMS 等完全一致）
- ✅ 收集階段即做 Top-K（與 COCO maxDets 對齊）
- ✅ 產出更完整的 details JSON（每個 IoU 的 AP/AR 陣列 + S/M/L 區間）
"""

from pathlib import Path
import json
from collections import defaultdict

import torch
import torchvision
from PIL import Image
from tqdm import tqdm
import numpy as np

from config import load_cfg

# ========= 可以在這裡快速覆寫設定（可留空） =========
CFG_PATH = "experiments/configs/v7.yaml"
OVERRIDES = [
    # 範例：
    "checkpoint.save_full_path=experiments/logs/fasterrcnn_v7/fasterrcnn_v7_best.pth",
    # "project.run_name=fasterrcnn_v7_eval",
]
# =================================================


# ---------- 通用模型工廠（v7 優先、v6 後備） ----------
def _import_builder():
    try:
        from modelv7 import build_detector_from_cfg as _b
        return _b
    except Exception:
        pass
    try:
        from modelv6 import build_detector_from_cfg as _b
        return _b
    except Exception:
        return None

_BUILD_DET = _import_builder()

def build_detector_from_cfg(cfg):
    assert _BUILD_DET is not None, "找不到模型工廠：請確認 src/modelv7.py（或 modelv6.py）存在。"
    return _BUILD_DET(cfg)


# ---------- 小工具 ----------
def _cfg_get(d, path, default=None):
    cur = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def resize_keep_max_side(pil_img: Image.Image, max_side: int) -> Image.Image:
    w, h = pil_img.size
    if max(w, h) <= max_side:
        return pil_img
    scale = float(max_side) / max(w, h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    return pil_img.resize((new_w, new_h), resample=Image.BILINEAR)

def load_gt(gt_txt_path: Path):
    gt_map = defaultdict(list)
    with open(gt_txt_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split(",") if "," in s else s.split()
            if len(parts) < 5:
                continue
            fid = int(parts[0]); x, y, w, h = map(float, parts[1:5])
            if w <= 0 or h <= 0:
                continue
            gt_map[fid].append([x, y, w, h])
    for k in list(gt_map.keys()):
        gt_map[k] = torch.tensor(gt_map[k], dtype=torch.float32)
    return gt_map

def list_val_images(img_dir: Path):
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP")
    items = []
    for p in img_dir.iterdir():
        if p.suffix in exts and p.is_file():
            stem = p.stem.lstrip("0")
            fid = int(stem) if stem != "" else 0
            items.append((fid, p))
    items.sort(key=lambda t: t[0])
    return items

def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    out = boxes.clone()
    out[:, 2] = boxes[:, 0] + boxes[:, 2]
    out[:, 3] = boxes[:, 1] + boxes[:, 3]
    return out

def compute_ap(rec, prec):
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])

def eval_split(det_records, gt_map_xywh, image_ids, iou_thr, max_dets=None, area_rng=None, return_curve=False):
    # 篩 det by image + top-k
    if max_dets is not None:
        per_img = defaultdict(list)
        for d in det_records:
            if d["image_id"] in image_ids:
                per_img[d["image_id"]].append(d)
        dets = []
        for fid, lst in per_img.items():
            lst = sorted(lst, key=lambda r: r["score"], reverse=True)[:max_dets]
            dets.extend(lst)
    else:
        dets = [d for d in det_records if d["image_id"] in image_ids]

    # 篩 GT by area
    if area_rng is not None:
        amin, amax = area_rng
        gt_sel = {}
        for fid in image_ids:
            g = gt_map_xywh.get(fid, torch.zeros((0, 4)))
            if g.numel() == 0:
                gt_sel[fid] = g
                continue
            areas = g[:, 2] * g[:, 3]
            keep = (areas >= amin) & (areas < amax)
            gt_sel[fid] = g[keep]
        gt_use = gt_sel
    else:
        gt_use = gt_map_xywh

    num_gt_total = sum(gt_use[fid].shape[0] for fid in image_ids)
    gt_matched = {fid: torch.zeros((gt_use[fid].shape[0],), dtype=torch.bool)
                  for fid in image_ids}

    det_sorted = sorted(dets, key=lambda r: r["score"], reverse=True)

    tp, fp = [], []
    scores_sorted = []
    for det in det_sorted:
        fid = det["image_id"]
        b = torch.tensor(det["box_xyxy"], dtype=torch.float32).unsqueeze(0)
        g = xywh_to_xyxy(gt_use.get(fid, torch.zeros((0, 4), dtype=torch.float32)))
        scores_sorted.append(det["score"])
        if g.shape[0] == 0:
            tp.append(0); fp.append(1); continue
        ious = torchvision.ops.box_iou(b, g)[0]
        if ious.numel() == 0:
            tp.append(0); fp.append(1); continue
        best_idx = int(ious.argmax().item())
        best_iou = float(ious[best_idx].item())
        if best_iou >= iou_thr and not gt_matched[fid][best_idx]:
            tp.append(1); fp.append(0); gt_matched[fid][best_idx] = True
        else:
            tp.append(0); fp.append(1)

    tp = np.array(tp); fp = np.array(fp)
    if tp.size == 0:
        rec = np.array([0.0]); prec = np.array([1.0]); ap = 0.0
    else:
        cum_tp = np.cumsum(tp); cum_fp = np.cumsum(fp)
        rec = cum_tp / max(1, num_gt_total)
        prec = cum_tp / np.maximum(cum_tp + cum_fp, 1e-12)
        ap = compute_ap(rec, prec)
    ar = float(rec.max()) if rec.size > 0 else 0.0

    if return_curve:
        return ap, ar, int(num_gt_total), rec.tolist(), prec.tolist(), scores_sorted
    return ap, ar, int(num_gt_total)


# ---------- 主程式 ----------
@torch.inference_mode()
def main():
    # 讀設定
    project_root = Path(__file__).resolve().parents[1]
    cfg = load_cfg(str(project_root / CFG_PATH), overrides=OVERRIDES)

    img_dir = Path(_cfg_get(cfg, "data.val_img_dir", "data/val/img"))
    gt_txt  = Path(_cfg_get(cfg, "data.val_gt", "data/val/gt_val.txt"))
    assert img_dir.exists(), f"找不到影像資料夾：{img_dir}"
    assert gt_txt.exists(), f"找不到標註檔：{gt_txt}"

    iou_list = _cfg_get(cfg, "eval.iou_thresholds",
                        [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95])
    max_side  = int(_cfg_get(cfg, "augment.max_side", 1280))
    score_thr = float(_cfg_get(cfg, "infer.score_thr", 0.05))
    max_dets_cfg = int(_cfg_get(cfg, "eval.max_det", 300))

    out_txt  = Path(_cfg_get(cfg, "eval.result_txt", "experiments/eval_results/eval_results.txt"))
    out_json = Path(_cfg_get(cfg, "eval.result_json", "experiments/eval_results/eval_details.json"))
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    # 清單與 GT
    val_images = list_val_images(img_dir)
    gt_map_xywh = load_gt(gt_txt)
    image_ids = [fid for fid, _ in val_images if fid in gt_map_xywh]

    # 建模：與訓練同 YAML（自動匹配 retinanet/fasterrcnn/ssdlite）
    device = torch.device("cuda" if torch.cuda.is_available() and _cfg_get(cfg, "device.cuda", True) else "cpu")
    model = build_detector_from_cfg(cfg).to(device)
    model.eval()

    # 權重路徑（優先用 save_full_path）
    ckpt_cfg  = cfg["checkpoint"]
    ckpt_path = Path(ckpt_cfg.get("save_full_path") or (Path(ckpt_cfg["dir"]) / ckpt_cfg["name"]))
    print(f"[Eval] Using checkpoint: {ckpt_path}")
    assert ckpt_path.exists(), f"找不到權重檔：{ckpt_path}"

    # 載入權重，容忍 fp16 → fp32
    def _state_to_fp32(state):
        for k, v in list(state.items()):
            if isinstance(v, torch.Tensor) and v.is_floating_point() and v.dtype == torch.float16:
                state[k] = v.float()
        return state

    try:
        state = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(str(ckpt_path), map_location="cpu")
    model.load_state_dict(_state_to_fp32(state), strict=True)
    model.to(device).eval()

    # 列印重要設定，檢查與訓練一致
    rpn_sizes   = _cfg_get(cfg, "model.rpn_anchor_sizes", None)
    rpn_ratios  = _cfg_get(cfg, "model.rpn_anchor_ratios", None)
    ret_sizes   = _cfg_get(cfg, "model.retinanet_anchor_sizes", None)
    ret_ratios  = _cfg_get(cfg, "model.retinanet_anchor_aspect_ratios", None)
    print(f"[Cfg] detector          = {_cfg_get(cfg, 'model.detector', 'unknown')}")
    print(f"[Cfg] min/max size      = {_cfg_get(cfg, 'model.min_size', '?')}/{_cfg_get(cfg, 'model.max_size', '?')}")
    print(f"[Cfg] score/nms/maxDet  = {score_thr}/{_cfg_get(cfg,'infer.nms_iou', 0.5)}/{max_dets_cfg}")
    if ret_sizes is not None:
        print(f"[Cfg] retinanet anchors = sizes:{ret_sizes} ratios:{ret_ratios}")
    elif rpn_sizes is not None:
        print(f"[Cfg] rpn anchors       = sizes:{rpn_sizes} ratios:{rpn_ratios}")

    print(f"[1/4] Inference on val images ...", flush=True)
    # 收集偵測（每張圖預先限制 Top-K，加速後續評分）
    det_records = []
    to_tensor = torchvision.transforms.ToTensor()
    for fid, fp in tqdm(val_images, ncols=100, desc="Eval on val"):
        if fid not in gt_map_xywh:
            continue
        img = Image.open(fp).convert("RGB")
        img = resize_keep_max_side(img, max_side=max_side)
        x = to_tensor(img).to(device).unsqueeze(0)
        out = model(x)[0]
        # torchvision 內部已做 NMS，這裡只做分數過濾與 Top-K
        boxes = out["boxes"].detach().cpu()
        scores = out["scores"].detach().cpu()
        keep = scores >= score_thr
        boxes = boxes[keep]; scores = scores[keep]
        if scores.numel() > max_dets_cfg:
            topk = min(max_dets_cfg, scores.numel())
            vals, idxs = torch.topk(scores, k=topk)
            boxes = boxes[idxs]; scores = vals
        for b, s in zip(boxes.tolist(), scores.tolist()):
            det_records.append({"image_id": fid, "score": float(s), "box_xyxy": b})

    # 面積區間（COCO）
    A_S, A_M, A_L = 32 ** 2, 96 ** 2, float("inf")
    area_ranges = {
        "all":    (0.0, float("inf")),
        "small":  (0.0, A_S),
        "medium": (A_S, A_M),
        "large":  (A_M, A_L),
    }

    print(f"[2/4] Computing AP (IoU=0.50:0.95, area=all, maxDets={max_dets_cfg}) ...", flush=True)
    aps_all = []
    for thr in iou_list:
        ap, _, _ = eval_split(det_records, gt_map_xywh, image_ids, thr, max_dets=max_dets_cfg, area_rng=area_ranges["all"])
        aps_all.append(ap)
    ap_50_95 = float(np.mean(aps_all) if aps_all else 0.0)

    print(f"[3/4] Computing AP by IoU (0.50/0.75) and by area (S/M/L) ...", flush=True)
    ap_50, _, _ = eval_split(det_records, gt_map_xywh, image_ids, 0.50, max_dets=max_dets_cfg, area_rng=area_ranges["all"])
    ap_75, _, _ = eval_split(det_records, gt_map_xywh, image_ids, 0.75, max_dets=max_dets_cfg, area_rng=area_ranges["all"])

    aps_s = [eval_split(det_records, gt_map_xywh, image_ids, t, max_dets=max_dets_cfg, area_rng=area_ranges["small"])[0]  for t in iou_list]
    aps_m = [eval_split(det_records, gt_map_xywh, image_ids, t, max_dets=max_dets_cfg, area_rng=area_ranges["medium"])[0] for t in iou_list]
    aps_l = [eval_split(det_records, gt_map_xywh, image_ids, t, max_dets=max_dets_cfg, area_rng=area_ranges["large"])[0]  for t in iou_list]
    ap_small  = float(np.mean(aps_s))
    ap_medium = float(np.mean(aps_m))
    ap_large  = float(np.mean(aps_l))

    print(f"[4/4] Computing AR (maxDets=1/10/100) ...", flush=True)
    def mean_ar(max_dets, area_key):
        ars = []
        for thr in iou_list:
            _, ar, _ = eval_split(det_records, gt_map_xywh, image_ids, thr, max_dets=max_dets, area_rng=area_ranges[area_key])
            ars.append(ar)
        return float(np.mean(ars) if ars else 0.0), ars

    ar_1_all,   ars_1_all   = mean_ar(1,   "all")
    ar_10_all,  ars_10_all  = mean_ar(10,  "all")
    ar_100_all, ars_100_all = mean_ar(100, "all")
    ar_100_s,   ars_100_s   = mean_ar(100, "small")
    ar_100_m,   ars_100_m   = mean_ar(100, "medium")
    ar_100_l,   ars_100_l   = mean_ar(100, "large")

    # 輸出（摘要）
    lines = []
    lines.append("===== COCO Evaluation Summary =====\n")
    lines.append(f"AP @[ IoU=0.50:0.95 | area=all | maxDets={max_dets_cfg} ]              = {ap_50_95:.6f}")
    lines.append(f"AP @[ IoU=0.50      | area=all | maxDets={max_dets_cfg} ]              = {ap_50:.6f}")
    lines.append(f"AP @[ IoU=0.75      | area=all | maxDets={max_dets_cfg} ]              = {ap_75:.6f}")
    lines.append(f"AP @[ IoU=0.50:0.95 | area=small | maxDets={max_dets_cfg} ]            = {ap_small:.6f}")
    lines.append(f"AP @[ IoU=0.50:0.95 | area=medium | maxDets={max_dets_cfg} ]           = {ap_medium:.6f}")
    lines.append(f"AP @[ IoU=0.50:0.95 | area=large | maxDets={max_dets_cfg} ]            = {ap_large:.6f}")
    lines.append(f"AR @[ IoU=0.50:0.95 | area=all | maxDets=1 ]                           = {ar_1_all:.6f}")
    lines.append(f"AR @[ IoU=0.50:0.95 | area=all | maxDets=10 ]                          = {ar_10_all:.6f}")
    lines.append(f"AR @[ IoU=0.50:0.95 | area=all | maxDets=100 ]                         = {ar_100_all:.6f}")
    lines.append(f"AR @[ IoU=0.50:0.95 | area=small | maxDets=100 ]                       = {ar_100_s:.6f}")
    lines.append(f"AR @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]                      = {ar_100_m:.6f}")
    lines.append(f"AR @[ IoU=0.50:0.95 | area=large | maxDets=100 ]                       = {ar_100_l:.6f}\n")
    lines.append("\n[說明]")
    lines.append("- AP: 平均精度；AR: 平均召回；主指標為 AP@[0.50:0.95]（COCO 標準）。")
    lines.append("- precision 的維度為 [IoU x recall x category x area x maxDets]。")
    lines.append("- recall    的維度為 [IoU x category x area x maxDets]（每個 IoU 的最大可達召回）。")
    lines.append("- 詳細陣列請見 *_details.json（可用來畫曲線與分析 S/M/L）。")

    out_txt.write_text("\n".join(lines), encoding="utf-8")

    # 輸出（細節 JSON：每個 IoU 的 AP/AR 列表 + 面積分段）
    details = {
        "AP_main": {
            "AP_50_95": ap_50_95,
            "AP_50": ap_50,
            "AP_75": ap_75,
        },
        "AP_by_IoU": {
            "IoUs": iou_list,
            "AP_all": aps_all,
            "AP_small": aps_s,
            "AP_medium": aps_m,
            "AP_large": aps_l,
        },
        "AR_by_IoU_and_maxDets": {
            "IoUs": iou_list,
            "AR@1_all":   ars_1_all,
            "AR@10_all":  ars_10_all,
            "AR@100_all": ars_100_all,
            "AR@100_S":   ars_100_s,
            "AR@100_M":   ars_100_m,
            "AR@100_L":   ars_100_l,
        },
        "cfg_refs": {
            "detector": _cfg_get(cfg, "model.detector"),
            "min_size": _cfg_get(cfg, "model.min_size"),
            "max_size": _cfg_get(cfg, "model.max_size"),
            "box_score_thresh": _cfg_get(cfg, "infer.score_thr"),
            "box_nms_thresh": _cfg_get(cfg, "infer.nms_iou"),
            "max_dets_eval": max_dets_cfg,
            # 兩種 anchors 欄位擇一存在
            "rpn_anchor_sizes":  _cfg_get(cfg, "model.rpn_anchor_sizes"),
            "rpn_anchor_ratios": _cfg_get(cfg, "model.rpn_anchor_ratios"),
            "retinanet_anchor_sizes": _cfg_get(cfg, "model.retinanet_anchor_sizes"),
            "retinanet_anchor_aspect_ratios": _cfg_get(cfg, "model.retinanet_anchor_aspect_ratios"),
        },
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(details, f, indent=2, ensure_ascii=False)

    # 同步印到 console
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
