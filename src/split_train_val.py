#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
split_train_val.py
從 data/train 拆出一份 validation（預設 15%），生成：
  data/val/img/           # 驗證影像
  data/val/gt_val.txt     # 驗證標註
  data/train/gt_train.txt # 訓練標註（剩餘的）

預設「複製」影像到 val（安全）。若想把 val 影像「移走」，加 --move。
"""

import os
import argparse
import random
import shutil

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def read_gt(gt_path):
    """讀取 'frame,l,t,w,h' -> dict: { '00000001': [(l,t,w,h), ...], ... }"""
    by_id = {}
    with open(gt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 5:
                continue
            frame, l, t, w, h = parts
            fid = f"{int(float(frame)):08d}"  # 統一 8 位字串 id
            by_id.setdefault(fid, []).append((int(l), int(t), int(w), int(h)))
    return by_id


def list_image_ids(img_dir):
    """列出資料夾內所有影像的 id（以 8 位字串表示）"""
    ids = []
    for name in os.listdir(img_dir):
        base, ext = os.path.splitext(name)
        if ext.lower() not in IMG_EXTS:
            continue
        if not base.isdigit():
            continue
        ids.append(f"{int(base):08d}")
    ids.sort()
    return ids


def write_gt(path, gt_by_id, id_list):
    """將指定 id 的標註寫到 path，frame 寫回純數字（不補零）以符合原始 gt.txt 風格"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for fid in id_list:
            anns = gt_by_id.get(fid, [])
            for (l, t, w, h) in anns:
                f.write(f"{int(fid)},{l},{t},{w},{h}\n")


def main():
    ap = argparse.ArgumentParser("Split train/val from data/train")
    ap.add_argument("--train-img", default="data/train/img", help="train 影像資料夾")
    ap.add_argument("--gt", default="data/train/gt.txt", help="train 標註檔")
    ap.add_argument("--val-ratio", type=float, default=0.15, help="驗證比例 (0~1)")
    ap.add_argument("--val-img", default="data/val/img", help="val 影像輸出資料夾")
    ap.add_argument("--out-train-gt", default="data/train/gt_train.txt")
    ap.add_argument("--out-val-gt", default="data/val/gt_val.txt")
    ap.add_argument("--move", action="store_true", help="移動檔案而非複製（會把 val 圖片移出 train/img）")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    if not os.path.isdir(args.train_img):
        raise FileNotFoundError(f"train 影像資料夾不存在：{args.train_img}")
    if not os.path.isfile(args.gt):
        raise FileNotFoundError(f"找不到 gt.txt：{args.gt}")

    gt_by_id = read_gt(args.gt)
    all_ids = list_image_ids(args.train_img)
    if not all_ids:
        raise RuntimeError(
            f"在 {args.train_img} 找不到影像（副檔名限 {sorted(IMG_EXTS)}，檔名需為數字）"
        )

    n_total = len(all_ids)
    n_val = max(1, int(round(n_total * args.val_ratio)))
    val_ids = set(random.sample(all_ids, n_val))
    train_ids = [fid for fid in all_ids if fid not in val_ids]

    # 準備 val 影像資料夾
    os.makedirs(args.val_img, exist_ok=True)
    op = shutil.move if args.move else shutil.copy2

    moved = 0
    for fid in val_ids:
        for ext in IMG_EXTS:
            src1 = os.path.join(args.train_img, f"{int(fid)}{ext}")
            src2 = os.path.join(args.train_img, f"{fid}{ext}")  # 支援 00000001.jpg
            if os.path.exists(src1):
                dst = os.path.join(args.val_img, os.path.basename(src1))
                if not os.path.exists(dst):
                    op(src1, dst)
                    moved += 1
                break
            elif os.path.exists(src2):
                dst = os.path.join(args.val_img, os.path.basename(src2))
                if not os.path.exists(dst):
                    op(src2, dst)
                    moved += 1
                break

    write_gt(args.out_val_gt, gt_by_id, sorted(val_ids))
    write_gt(args.out_train_gt, gt_by_id, train_ids)

    print(f"[Done] total={n_total}, val={len(val_ids)}, train={len(train_ids)}")
    print(f"  val images -> {args.val_img}  ({'moved' if args.move else 'copied'} {moved})")
    print(f"  GT files: {args.out_val_gt}, {args.out_train_gt}")


if __name__ == "__main__":
    main()
