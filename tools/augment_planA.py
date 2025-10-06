#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plan A: 加倍資料量（每張原圖新增 1 張）
- 標註格式：<frame_id>,<bb_left>,<bb_top>,<bb_width>,<bb_height>
- 只做水平翻轉會改 bbox；外觀增強不改 bbox
- 新檔名從現有最大數字 +1 開始連號（保留副檔名）
- 直接把新增樣本的標註 append 到 gt_train.txt（先備份 .bak）
- 原始影像不會被修改
"""

from pathlib import Path
import io, re, shutil, random
from collections import defaultdict
from typing import List, Tuple, Dict

from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import numpy as np

# ===== 參數（可改）=====
IMG_DIR = Path("data/train/img")
GT_FILE = Path("data/train/gt_train.txt")
BACKUP_GT = True
AUG_PER_IMAGE = 1              # 每張原圖產生幾張增強圖（=1 即加倍）
APPLY_HFLIP_PROB = 0.5         # 水平翻轉機率（唯一會更動 bbox 的增強）
APPLY_COLOR_JITTER = True      # 亮度/對比/飽和/色相輕微抖動
APPLY_GAUSSIAN_BLUR = True
APPLY_JPEG_ARTIFACT = True
APPLY_NOISE = True
JPEG_QUALITY_RANGE = (60, 95)

BRIGHTNESS_GAIN = 0.20
CONTRAST_GAIN   = 0.20
SATURATION_GAIN = 0.20
HUE_GAIN        = 0.02          # 0~0.5

ID_PAD = 8                      # ← 檔名位數：你的資料是 8 碼（例：00000001.jpg）
# =======================

_num_re = re.compile(r"(\d+)")

def _extract_number(stem: str):
    ms = _num_re.findall(stem)
    return int(ms[-1]) if ms else None

def _find_max_index(img_dir: Path) -> int:
    mx = 0
    for p in img_dir.iterdir():
        if p.is_file():
            n = _extract_number(p.stem)
            if n is not None:
                mx = max(mx, n)
    return mx

def _parse_gt_line(line: str) -> Tuple[int, float, float, float, float]:
    s = line.strip()
    if not s:
        raise ValueError("empty line")
    toks = [t.strip() for t in s.split(",") if t.strip() != ""]
    if len(toks) != 5:
        raise ValueError(f"bad gt format: {line}")
    fid, x, y, w, h = toks
    return int(fid), float(x), float(y), float(w), float(h)

def _color_jitter_pil(img: Image.Image) -> Image.Image:
    if not APPLY_COLOR_JITTER or random.random() > 0.8:
        return img
    img = ImageEnhance.Brightness(img).enhance(1.0 + random.uniform(-BRIGHTNESS_GAIN, BRIGHTNESS_GAIN))
    img = ImageEnhance.Contrast(img).enhance(1.0 + random.uniform(-CONTRAST_GAIN, CONTRAST_GAIN))
    img = ImageEnhance.Color(img).enhance(1.0 + random.uniform(-SATURATION_GAIN, SATURATION_GAIN))
    # hue shift
    try:
        arr = np.array(img.convert("HSV"))
        h = arr[..., 0].astype(np.int16)
        shift = int(255 * random.uniform(-HUE_GAIN, HUE_GAIN))
        arr[..., 0] = ((h + shift) % 255).astype(np.uint8)
        img = Image.fromarray(arr, "HSV").convert("RGB")
    except Exception:
        pass
    return img

def _gaussian_blur(img: Image.Image) -> Image.Image:
    if not APPLY_GAUSSIAN_BLUR or random.random() > 0.5:
        return img
    return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.2)))

def _jpeg_artifact(img: Image.Image) -> Image.Image:
    if not APPLY_JPEG_ARTIFACT or random.random() > 0.7:
        return img
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=random.randint(*JPEG_QUALITY_RANGE))
    buf.seek(0)
    return Image.open(buf).convert("RGB")

def _add_noise(img: Image.Image) -> Image.Image:
    if not APPLY_NOISE or random.random() > 0.7:
        return img
    arr = np.asarray(img).astype(np.float32)
    noise = np.random.normal(0, 5.0, size=arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def _apply_appearance_aug(img: Image.Image) -> Image.Image:
    img = _color_jitter_pil(img)
    img = _gaussian_blur(img)
    img = _jpeg_artifact(img)
    img = _add_noise(img)
    return img

def main():
    assert IMG_DIR.exists() and GT_FILE.exists(), f"paths not found: {IMG_DIR}, {GT_FILE}"

    # 讀標註並按 frame_id 彙整
    lines = GT_FILE.read_text(encoding="utf-8").splitlines()
    boxes_by_fid: Dict[int, List[Tuple[float,float,float,float]]] = defaultdict(list)
    for ln in lines:
        if not ln.strip():
            continue
        fid, x, y, w, h = _parse_gt_line(ln)
        boxes_by_fid[fid].append((x, y, w, h))

    # 推測影像副檔名：以現有最小 id 的檔案為準
    existing_files = sorted([p for p in IMG_DIR.iterdir() if p.is_file()],
                            key=lambda p: _extract_number(p.stem) or 0)
    assert existing_files, f"no images in {IMG_DIR}"
    ext = existing_files[0].suffix  # 如 .jpg/.png

    # 從現有最大編號 + 1 開始
    next_id = _find_max_index(IMG_DIR) + 1
    print(f"[INFO] numbering starts from {next_id}")

    # 備份 gt
    if BACKUP_GT:
        bak = GT_FILE.with_suffix(".txt.bak")
        shutil.copy2(GT_FILE, bak)
        print(f"[INFO] backup created: {bak}")

    new_gt_lines: List[str] = []
    new_images = 0

    # 逐張影像
    for fid in sorted(boxes_by_fid.keys()):
        img_path = IMG_DIR / f"{fid:0{ID_PAD}d}{ext}"
        if not img_path.exists():
            print(f"[WARN] image not found for frame {fid}: {img_path}")
            continue

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[WARN] cannot open {img_path}: {e}")
            continue

        w, h = img.size
        src_boxes = boxes_by_fid[fid]

        for _ in range(AUG_PER_IMAGE):
            aug = img.copy()

            # 幾何：水平翻轉（需要更新 bbox）
            do_flip = (random.random() < APPLY_HFLIP_PROB)
            if do_flip:
                aug = ImageOps.mirror(aug)
                # 新 left = W - (left + width)
                new_boxes = [(max(0.0, w - (x + bw)), y, bw, bh) for (x, y, bw, bh) in src_boxes]
            else:
                new_boxes = list(src_boxes)

            # 外觀增強（不改 bbox）
            aug = _apply_appearance_aug(aug)

            # 產生新檔名與 id
            while (IMG_DIR / f"{next_id:0{ID_PAD}d}{ext}").exists():
                next_id += 1
            out_path = IMG_DIR / f"{next_id:0{ID_PAD}d}{ext}"
            aug.save(out_path)

            # 同步寫入標註（frame_id 改為 next_id）
            for (x, y, bw, bh) in new_boxes:
                new_gt_lines.append(f"{next_id},{x:.1f},{y:.1f},{bw:.1f},{bh:.1f}")

            next_id += 1
            new_images += 1

    # 追加到 gt_train.txt
    with GT_FILE.open("a", encoding="utf-8") as f:
        for ln in new_gt_lines:
            f.write(ln + "\n")

    print(f"[DONE] added {new_images} new images and {len(new_gt_lines)} gt lines.")
    print(f"[NOTE] originals are untouched. Updated GT: {GT_FILE}")

if __name__ == "__main__":
    main()
