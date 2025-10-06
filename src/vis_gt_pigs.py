#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
視覺化訓練集豬隻 BBox（from gt_train.txt）— 完整版
- 讀取每行標註: <frame_id>,<bb_left>,<bb_top>,<bb_width>,<bb_height>
- 將每張圖的所有框疊加並輸出到 VIS_OUT_DIR
- 產生 summary.csv（每張圖的框數、總面積、問題統計、IoU統計）
- 產生 index.html 方便快速瀏覽
- 產生 issues.csv（逐框與逐圖的問題彙整：缺圖、極小框、超界比例、過度重疊）

設計重點：
1) 全面解決「檔名位數不一致」：建立影像多鍵索引（原名、去前導零、各種補零寬度），基本不會 miss。
2) 子資料夾支援：RECURSIVE_SCAN=True 即會遞迴掃描。
3) 問題偵測：
   - Tiny box（寬或高 < MIN_W/MIN_H）
   - Out-of-bounds（超出邊界比例 > OOB_RATIO_THR）
   - Overlap（任兩框 IoU ≥ IOU_THR）
   - Missing image（某 frame_id 找不到對應影像）
4) 視覺化僅繪「裁邊後且通過最小尺寸門檻」的框，避免畫面雜訊；原始問題都在 issues.csv 詳列。

相依：
- Pillow、numpy、（可選）opencv-python（有的話畫框較快）
"""

from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import csv
import math
import sys
import traceback

try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

from PIL import Image, ImageDraw, ImageFont
import numpy as np


# ========= 參數（可改）=========
IMG_DIR = Path("data/train/img")           # 訓練影像資料夾
GT_FILE = Path("data/train/gt_train.txt")  # 訓練標註
VIS_OUT_DIR = Path("experiments/vis_gt")   # 可視化輸出資料夾
SUMMARY_CSV = VIS_OUT_DIR / "summary.csv"
ISSUES_CSV = VIS_OUT_DIR / "issues.csv"
INDEX_HTML = VIS_OUT_DIR / "index.html"
MISSING_LIST = VIS_OUT_DIR / "missing_images.txt"

# 掃描影像 & 檔名對齊
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP")
PAD_WIDTHS = (3, 4, 5, 6, 7, 8)   # 常見補零寬度
RECURSIVE_SCAN = False            # 若影像藏在子資料夾，改 True

# 視覺化外觀與濾除條件
LINE_THICKNESS = 3
FONT_SIZE = 20
COLOR_BOX = (255, 64, 64)     # 紅
COLOR_TEXT_BG = (0, 0, 0, 160)
COLOR_TEXT = (255, 255, 255)
MIN_W = 1                     # 過小 bbox 濾除（像素）
MIN_H = 1
MAX_SIDE = 1280               # 輸出影像最長邊（None=原尺寸）

# 問題偵測門檻（只影響 issues.csv 報告，不影響原始 GT）
OOB_RATIO_THR = 0.20          # 超出邊界比例 > 20% 視為超界問題
IOU_THR = 0.65                 # 任兩框 IoU ≥ 0.65 視為過度重疊
MAX_ISSUES_PER_IMAGE = 50      # 單圖 issues.csv 輸出上限，避免爆量
# =================================


@dataclass
class Box:
    left: float
    top: float
    width: float
    height: float

    @property
    def right(self) -> float:
        return self.left + self.width

    @property
    def bottom(self) -> float:
        return self.top + self.height

    def as_xyxy(self) -> Tuple[int, int, int, int]:
        return int(round(self.left)), int(round(self.top)), int(round(self.right)), int(round(self.bottom))

    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)


def read_gt(path: Path) -> Dict[str, List[Box]]:
    """
    讀取 gt_train.txt，回傳 {frame_id_str: [Box, ...]}
    """
    mapping: Dict[str, List[Box]] = {}
    invalid_cnt = 0
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.replace(" ", "").split(",")]
            if len(parts) < 5:
                parts = [p.strip() for p in line.split()]
            if len(parts) < 5:
                print(f"[Warn] Line {ln}: format not recognized -> {line}")
                continue
            fid, l, t, w, h = parts[:5]
            try:
                box = Box(float(l), float(t), float(w), float(h))
                if box.width <= 0 or box.height <= 0 or box.area() <= 0:
                    invalid_cnt += 1
                    continue
                mapping.setdefault(fid, []).append(box)
            except Exception:
                print(f"[Warn] Line {ln}: parse error -> {line}")
                continue
    if invalid_cnt:
        print(f"[Info] Skipped {invalid_cnt} invalid boxes (w<=0 or h<=0).")
    return mapping


def build_image_index(img_dir: Path) -> Dict[str, Path]:
    """
    掃描 IMG_DIR，建立多鍵索引：
    - key1: 原檔名（不含副檔名），如 "000807"
    - key2: 去前導零，如 "807"
    - key3+: 對純數字檔名建立各種補零鍵 "000807"(6位), "0807"(4位) 等
    """
    idx: Dict[str, Path] = {}
    files = []
    if RECURSIVE_SCAN:
        for ext in IMG_EXTS:
            files.extend(img_dir.rglob(f"*{ext}"))
    else:
        for ext in IMG_EXTS:
            files.extend(img_dir.glob(f"*{ext}"))

    for p in files:
        stem = p.stem
        keys = set()
        keys.add(stem)

        # 去前導零
        if all(ch == "0" for ch in stem):
            no0 = "0"
        else:
            no0 = stem.lstrip("0") or "0"
        keys.add(no0)

        # 純數字則補各種位數
        if no0.isdigit():
            num = int(no0)
            for w in PAD_WIDTHS:
                keys.add(str(num).zfill(w))

        for k in keys:
            if k not in idx:
                idx[k] = p
            else:
                # 以路徑較短者為優
                if len(str(p)) < len(str(idx[k])):
                    idx[k] = p

    print(f"[Index] indexed {len(files)} images with {len(idx)} unique keys.")
    # 顯示一些樣本鍵
    if idx:
        sample_keys = list(idx.keys())[:8]
        print(f"[Index] sample keys: {sample_keys}")
    return idx


def find_image_path_by_index(index: Dict[str, Path], frame_id: str) -> Optional[Path]:
    """依序嘗試原字串、去前導零、各寬度補零以命中索引。"""
    if frame_id in index:
        return index[frame_id]
    fid_no0 = frame_id.lstrip("0") or "0"
    if fid_no0 in index:
        return index[fid_no0]
    if fid_no0.isdigit():
        n = int(fid_no0)
        for w in PAD_WIDTHS:
            k = str(n).zfill(w)
            if k in index:
                return index[k]
    return None


def resize_keep_max_side(img: Image.Image, max_side: Optional[int]) -> Tuple[Image.Image, float]:
    if not max_side or max(img.size) <= max_side:
        return img, 1.0
    w, h = img.size
    scale = max_side / max(w, h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    return img.resize((nw, nh), Image.BILINEAR), scale


def clip_box_to_img(b: Box, W: int, H: int) -> Tuple[Box, float]:
    """
    將 b 裁到影像邊界內，回傳 (裁後 box, 超出比例)
    超出比例 = 1 - (clipped_area / original_area)；若原面積=0則回 1
    """
    x1, y1, x2, y2 = b.as_xyxy()
    x1c = max(0, min(W - 1, x1))
    y1c = max(0, min(H - 1, y1))
    x2c = max(0, min(W - 1, x2))
    y2c = max(0, min(H - 1, y2))
    w0, h0 = max(0, x2 - x1), max(0, y2 - y1)
    w1, h1 = max(0, x2c - x1c), max(0, y2c - y1c)
    area0 = w0 * h0
    area1 = w1 * h1
    oob_ratio = 1.0 if area0 == 0 else (1 - (area1 / area0))
    return Box(x1c, y1c, w1, h1), oob_ratio


def iou_xyxy(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    aw, ah = max(0, ax2 - ax1), max(0, ay2 - ay1)
    bw, bh = max(0, bx2 - bx1), max(0, by2 - by1)
    union = aw * ah + bw * bh - inter
    return 0.0 if union == 0 else inter / union


def draw_boxes_pil(img: Image.Image, boxes: List[Box], thickness=3, color=(255, 64, 64),
                   font_size=20) -> Image.Image:
    draw = ImageDraw.Draw(img, "RGBA")
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    for i, b in enumerate(boxes, 1):
        x1, y1, x2, y2 = b.as_xyxy()
        if x2 <= x1 or y2 <= y1:
            continue
        for t in range(thickness):
            draw.rectangle([x1 - t, y1 - t, x2 + t, y2 + t], outline=color)
        label = f"pig #{i}"
        tw = draw.textlength(label, font=font)
        th = font_size + 6
        draw.rectangle([x1, max(0, y1 - th), x1 + tw + 8, y1], fill=COLOR_TEXT_BG)
        draw.text((x1 + 4, y1 - th + 3), label, font=font, fill=COLOR_TEXT)
    return img


def save_index_html(items: List[Tuple[str, str, int]], out_path: Path):
    """
    items: list of (frame_id, rel_img_path, num_boxes)
    """
    html = [
        "<!DOCTYPE html>",
        "<meta charset='utf-8'>",
        "<title>GT Visualization</title>",
        "<style>body{font-family:ui-sans-serif,system-ui,Arial;margin:24px} .g{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:14px} figure{margin:0;border:1px solid #ddd;border-radius:8px;overflow:hidden} figcaption{padding:8px 10px;background:#fafafa;border-top:1px solid #eee} img{width:100%;display:block}</style>",
        "<h2>GT Visualization (train)</h2>",
        "<div class='g'>"
    ]
    for fid, relp, n in items:
        html += [
            "<figure>",
            f"<img loading='lazy' src='{relp}' alt='{fid}'>",
            f"<figcaption><b>{fid}</b> &nbsp; boxes: {n}</figcaption>",
            "</figure>"
        ]
    html += ["</div>"]
    out_path.write_text("\n".join(html), encoding="utf-8")


def main():
    print(f"[CFG] IMG_DIR={IMG_DIR.resolve()}")
    print(f"[CFG] GT_FILE={GT_FILE.resolve()}")
    VIS_OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 讀 GT
    gt = read_gt(GT_FILE)
    print(f"[Info] frames in gt: {len(gt)}")

    # 建影像索引
    img_index = build_image_index(IMG_DIR)

    missing_fids = []
    rows_summary = []
    rows_issues = []
    index_items = []

    for fid, boxes in gt.items():
        img_path = find_image_path_by_index(img_index, fid)
        if not img_path:
            missing_fids.append(fid)
            # issues: 缺圖（逐圖）
            rows_issues.append([fid, "missing_image", "no file matched", "", "", "", "", "", ""])
            continue

        # 讀圖
        try:
            pil = Image.open(img_path).convert("RGB")
        except Exception as e:
            missing_fids.append(fid)
            rows_issues.append([fid, "missing_image", f"open_failed: {e}", "", "", "", "", "", ""])
            continue

        W, H = pil.size

        # 問題偵測（逐框）
        kept_for_vis: List[Box] = []
        oob_count = 0
        tiny_count = 0
        iou_records: List[float] = []

        # 先裁邊 & 統計 OOB 與 tiny
        clipped_boxes = []
        original_xyxy = []
        for idx, b in enumerate(boxes):
            cb, oob_ratio = clip_box_to_img(b, W, H)
            x1, y1, x2, y2 = cb.as_xyxy()
            # 記錄原始（未裁前）盒與裁後盒
            original_xyxy.append(b.as_xyxy())
            clipped_boxes.append(cb)

            if oob_ratio > OOB_RATIO_THR:
                oob_count += 1
                if len(rows_issues) - len(missing_fids) < MAX_ISSUES_PER_IMAGE:
                    rows_issues.append([fid, "oob_ratio",
                                        f"{oob_ratio:.3f} > {OOB_RATIO_THR}",
                                        idx, f"{b.left:.1f}", f"{b.top:.1f}", f"{b.width:.1f}", f"{b.height:.1f}",
                                        f"WxH={W}x{H}"])
            # tiny 判斷以「裁後」為準（可視化也以裁後）
            if cb.width < MIN_W or cb.height < MIN_H:
                tiny_count += 1
                if len(rows_issues) - len(missing_fids) < MAX_ISSUES_PER_IMAGE:
                    rows_issues.append([fid, "tiny_box",
                                        f"w={cb.width:.1f}, h={cb.height:.1f} < ({MIN_W},{MIN_H})",
                                        idx, f"{b.left:.1f}", f"{b.top:.1f}", f"{b.width:.1f}", f"{b.height:.1f}",
                                        "after_clip"])
            else:
                kept_for_vis.append(cb)

        # 重疊檢測：對 kept_for_vis 計算 pairwise IoU
        K = len(kept_for_vis)
        if K >= 2:
            xyxy_list = [bb.as_xyxy() for bb in kept_for_vis]
            for i in range(K):
                for j in range(i + 1, K):
                    iou = iou_xyxy(xyxy_list[i], xyxy_list[j])
                    iou_records.append(iou)
                    if iou >= IOU_THR and (len(rows_issues) - len(missing_fids)) < MAX_ISSUES_PER_IMAGE:
                        rows_issues.append([fid, "overlap",
                                            f"IoU={iou:.3f} >= {IOU_THR}",
                                            f"{i}|{j}", "", "", "", "", "vis_clipped"])

        # 可視化輸出
        pil_vis, scale = resize_keep_max_side(pil, MAX_SIDE)
        scaled_boxes = [Box(b.left * scale, b.top * scale, b.width * scale, b.height * scale) for b in kept_for_vis]

        if _HAS_CV2:
            arr = np.array(pil_vis)[:, :, ::-1].copy()  # RGB->BGR
            for i, b in enumerate(scaled_boxes, 1):
                x1, y1, x2, y2 = b.as_xyxy()
                if x2 <= x1 or y2 <= y1:
                    continue
                cv2.rectangle(arr, (x1, y1), (x2, y2), (64, 64, 255), thickness=LINE_THICKNESS)
                label = f"pig #{i}"
                ((tw, th), _) = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                y0 = max(0, y1 - th - 6)
                cv2.rectangle(arr, (x1, y0), (x1 + tw + 8, y0 + th + 6), (0, 0, 0), thickness=-1)
                cv2.putText(arr, label, (x1 + 4, y0 + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            pil_out = Image.fromarray(arr[:, :, ::-1])
        else:
            pil_out = draw_boxes_pil(pil_vis, scaled_boxes, thickness=LINE_THICKNESS, color=COLOR_BOX, font_size=FONT_SIZE)

        out_name = f"{Path(img_path).stem}_gt.jpg"
        out_path = VIS_OUT_DIR / out_name
        pil_out.save(out_path, quality=92)

        # summary 統計
        tot_area = sum(b.width * b.height for b in kept_for_vis)
        iou_max = max(iou_records) if iou_records else 0.0
        iou_mean = (sum(iou_records) / len(iou_records)) if iou_records else 0.0

        rel_for_html = out_path.relative_to(VIS_OUT_DIR).as_posix()
        index_items.append((fid, rel_for_html, len(kept_for_vis)))

        rows_summary.append([
            fid,
            str(img_path.relative_to(IMG_DIR) if IMG_DIR in img_path.parents else img_path.name),
            len(boxes),                # boxes_in_gt
            len(kept_for_vis),         # boxes_kept(>=min_wh after clip)
            f"{tot_area:.1f}",         # sum_area_kept
            oob_count,
            tiny_count,
            f"{iou_max:.3f}",
            f"{iou_mean:.3f}",
            W, H
        ])

    # 輸出 CSV/HTML/TXT
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame_id", "img_path", "boxes_in_gt", "boxes_kept(>=min_wh,clipped)",
            "sum_area_kept", "oob_count", "tiny_count", "max_iou_kept", "mean_iou_kept", "img_W", "img_H"
        ])
        writer.writerows(rows_summary)

    with ISSUES_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame_id", "issue_type", "detail",
            "bbox_index_or_pair", "bb_left", "bb_top", "bb_width", "bb_height", "extra"
        ])
        writer.writerows(rows_issues)

    if missing_fids:
        MISSING_LIST.write_text("\n".join(missing_fids), encoding="utf-8")

    save_index_html(index_items, INDEX_HTML)

    # 報告
    print(f"[Done] Visualized: {len(index_items)} images -> {VIS_OUT_DIR.resolve()}")
    print(f"[Info] Summary CSV: {SUMMARY_CSV.resolve()}")
    print(f"[Info] Issues  CSV: {ISSUES_CSV.resolve()}")
    print(f"[Info] Index  HTML: {INDEX_HTML.resolve()}")
    if missing_fids:
        print(f"[Warn] Missing images ({len(missing_fids)}) -> {MISSING_LIST.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("[Fatal] Uncaught exception:", e)
        traceback.print_exc()
        sys.exit(1)
