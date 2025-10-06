#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/eval.py
---------------------------------
單類別（class=0=pig）COCO 風格指標（AP/AR）並以 COCO 官方摘要格式輸出。
只讀 data/val/img 與 data/val/gt_val.txt。
做法優化：
- 收集階段就先對每張影像只保留 Top-K（與 COCO maxDets 對齊）
- 每個主要階段印出進度（flush=True）
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
from modelv6 import get_fasterrcnn_r50_fpn  # 與 `python src/eval.py` 相容的匯入方式

# ========= 可在這裡快速覆寫設定（可留空） =========
CFG_PATH = "experiments/configs/v6.yaml"
OVERRIDES = [
    #"checkpoint.save_full_path=experiments/logs/fasterrcnn_v5.1/fasterrcnn_v5.1.pth",
    #"project.run_name=fasterrcnn_v5_50e",  # 只影響輸出檔名
]
# =================================================


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


def eval_split(det_records, gt_map_xywh, image_ids, iou_thr, max_dets=None, area_rng=None):
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
    for det in det_sorted:
        fid = det["image_id"]
        b = torch.tensor(det["box_xyxy"], dtype=torch.float32).unsqueeze(0)
        g = xywh_to_xyxy(gt_use.get(fid, torch.zeros((0, 4), dtype=torch.float32)))
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
    return ap, ar, int(num_gt_total)


@torch.inference_mode()
def main():
    # 讀設定
    project_root = Path(__file__).resolve().parents[1]
    cfg = load_cfg(str(project_root / CFG_PATH), overrides=OVERRIDES)

    img_dir = Path(cfg["data"].get("val_img_dir", "data/val/img"))
    gt_txt  = Path(cfg["data"].get("val_gt", "data/val/gt_val.txt"))
    assert img_dir.exists(), f"找不到影像資料夾：{img_dir}"
    assert gt_txt.exists(), f"找不到標註檔：{gt_txt}"

    iou_list = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    max_side = int(cfg["augment"]["max_side"])
    score_thr = float(cfg["infer"]["score_thr"])
    max_dets_cfg = int(cfg["eval"].get("max_det", 100))

    out_txt  = Path(cfg["eval"]["result_txt"])
    out_json = Path(cfg["eval"]["result_json"])
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    # 清單與 GT
    val_images = list_val_images(img_dir)
    gt_map_xywh = load_gt(gt_txt)
    image_ids = [fid for fid, _ in val_images if fid in gt_map_xywh]

    # 載模型
    device = torch.device("cuda" if torch.cuda.is_available() and cfg["device"]["cuda"] else "cpu")
    model = get_fasterrcnn_r50_fpn(
        num_classes=cfg["model"]["num_classes"],
        freeze_backbone=cfg["model"]["freeze_backbone"]
    ).to(device)

    # ✅ 優先使用 save_full_path；否則退回 dir/name
    ckpt_cfg  = cfg["checkpoint"]
    ckpt_path = Path(ckpt_cfg.get("save_full_path") or (Path(ckpt_cfg["dir"]) / ckpt_cfg["name"]))
    print(f"[Eval] Using checkpoint: {ckpt_path}")
    assert ckpt_path.exists(), f"找不到權重檔：{ckpt_path}"

    # 載入（相容舊版 torch，並自動把 fp16 權重轉回 fp32）
    def _state_to_fp32(state):
        for k, v in list(state.items()):
            if isinstance(v, torch.Tensor) and v.is_floating_point() and v.dtype == torch.float16:
                state[k] = v.float()
        return state

    try:
        state = torch.load(str(ckpt_path), map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(str(ckpt_path), map_location=device)
    state = _state_to_fp32(state)
    model.load_state_dict(state)
    model.eval()

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
        boxes = out["boxes"].detach().cpu()
        scores = out["scores"].detach().cpu()
        # 低分過濾
        keep = scores >= score_thr
        boxes = boxes[keep]; scores = scores[keep]
        # 只留 Top-K（與 COCO maxDets 對齊）
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
    aps = []
    for thr in iou_list:
        ap, _, _ = eval_split(det_records, gt_map_xywh, image_ids, thr, max_dets=max_dets_cfg, area_rng=area_ranges["all"])
        aps.append(ap)
    ap_50_95 = float(np.mean(aps) if aps else 0.0)

    print(f"[3/4] Computing AP by IoU (0.50/0.75) and by area (S/M/L) ...", flush=True)
    ap_50, _, _ = eval_split(det_records, gt_map_xywh, image_ids, 0.50, max_dets=max_dets_cfg, area_rng=area_ranges["all"])
    ap_75, _, _ = eval_split(det_records, gt_map_xywh, image_ids, 0.75, max_dets=max_dets_cfg, area_rng=area_ranges["all"])
    ap_small  = float(np.mean([eval_split(det_records, gt_map_xywh, image_ids, t, max_dets_cfg, area_ranges["small"])[0]  for t in iou_list]))
    ap_medium = float(np.mean([eval_split(det_records, gt_map_xywh, image_ids, t, max_dets_cfg, area_ranges["medium"])[0] for t in iou_list]))
    ap_large  = float(np.mean([eval_split(det_records, gt_map_xywh, image_ids, t, max_dets_cfg, area_ranges["large"])[0]  for t in iou_list]))

    print(f"[4/4] Computing AR (maxDets=1/10/100) ...", flush=True)
    def mean_ar(max_dets, area_key):
        ars = []
        for thr in iou_list:
            _, ar, _ = eval_split(det_records, gt_map_xywh, image_ids, thr, max_dets=max_dets, area_rng=area_ranges[area_key])
            ars.append(ar)
        return float(np.mean(ars) if ars else 0.0)

    ar_1_all   = mean_ar(1,   "all")
    ar_10_all  = mean_ar(10,  "all")
    ar_100_all = mean_ar(100, "all")
    ar_100_s   = mean_ar(100, "small")
    ar_100_m   = mean_ar(100, "medium")
    ar_100_l   = mean_ar(100, "large")

    # 輸出
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
    lines.append("- 詳細陣列請見 eval_details.json（可用來畫曲線）。")

    out_txt = Path(cfg["eval"]["result_txt"])
    out_json = Path(cfg["eval"]["result_json"])
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text("\n".join(lines), encoding="utf-8")

    details = {
        "AP_50_95": ap_50_95,
        "AP_50": ap_50,
        "AP_75": ap_75,
        "AP_small": ap_small,
        "AP_medium": ap_medium,
        "AP_large": ap_large,
        "AR_1_all": ar_1_all,
        "AR_10_all": ar_10_all,
        "AR_100_all": ar_100_all,
        "AR_100_small": ar_100_s,
        "AR_100_medium": ar_100_m,
        "AR_100_large": ar_100_l,
        "iou_list": iou_list,
        "max_dets_eval": max_dets_cfg,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(details, f, indent=2, ensure_ascii=False)

    # 同步印到 console
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
