#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
split_train_val.py  (MOVE-ONLY, no CLI)
從 data/train 移出一份 validation（預設 15%），生成：
  - data/val/img/           # 驗證影像（從 train/img 「移動」過來）
  - data/val/gt_val.txt     # 驗證標註
  - data/train/gt_train.txt # 訓練標註（剩餘的）

設計重點：
- 固定使用「移動」(shutil.move)，不提供複製模式，也不使用 CLI 參數。
- 以檔頭常數設定路徑、比例與隨機種子，避免執行時指定參數。
- 嚴格安全檢查：每張被分派到 val 的影像，移動後來源需消失、目的地需存在。
- 原子寫檔：先寫 .tmp 再替換，避免中途中斷造成半張標註檔。
"""

import os
import random
import shutil
from typing import Dict, List, Tuple

# ========= 可在這裡快速調整的區域（無 CLI） =========
TRAIN_IMG_DIR = "data/train/img"
TRAIN_GT_PATH = "data/train/gt.txt"
VAL_RATIO = 0.15
VAL_IMG_DIR = "data/val/img"
OUT_TRAIN_GT = "data/train/gt_train.txt"
OUT_VAL_GT = "data/val/gt_val.txt"
SEED = 42
# ================================================

IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp"]  # 使用有序列表，避免 set 的不確定順序


def read_gt(gt_path: str) -> Dict[str, List[Tuple[int, int, int, int]]]:
    """讀取 'frame,l,t,w,h' -> dict: { '00000001': [(l,t,w,h), ...], ... }"""
    by_id: Dict[str, List[Tuple[int, int, int, int]]] = {}
    with open(gt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 5:
                continue
            frame, l, t, w, h = parts
            fid = f"{int(float(frame)):08d}"  # 統一 8 位 id
            by_id.setdefault(fid, []).append((int(l), int(t), int(w), int(h)))
    return by_id


def list_image_ids(img_dir: str) -> List[str]:
    """列出資料夾內所有影像的 id（以 8 位字串表示）"""
    ids: List[str] = []
    for name in os.listdir(img_dir):
        base, ext = os.path.splitext(name)
        if ext.lower() not in IMG_EXTS:
            continue
        if not base.isdigit():
            continue
        ids.append(f"{int(base):08d}")
    ids.sort()
    return ids


def write_gt_atomic(path: str, gt_by_id: Dict[str, List[Tuple[int, int, int, int]]], id_list: List[str]) -> None:
    """原子寫入：先輸出到 .tmp 再替換。frame 寫回純數字（不補零）以符合原始 gt.txt 風格"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        for fid in id_list:
            anns = gt_by_id.get(fid, [])
            for (l, t, w, h) in anns:
                f.write(f"{int(fid)},{l},{t},{w},{h}\n")
    os.replace(tmp_path, path)


def find_image_path(root: str, fid: str) -> str:
    """
    在 root 下嘗試兩種命名：'8位補零' 與 '不補零'，並依 IMG_EXTS 順序尋找。
    回傳存在的第一個完整路徑，找不到則回傳空字串。
    """
    int_name = str(int(fid))  # 不補零
    zero_name = fid          # 8位補零
    for name in (int_name, zero_name):
        for ext in IMG_EXTS:
            p = os.path.join(root, f"{name}{ext}")
            if os.path.exists(p):
                return p
    return ""


def ensure_moved(src_before: str, dst_after: str) -> None:
    """移動後的安全檢查：來源必須不存在、目的地必須存在。"""
    if os.path.exists(src_before):
        raise RuntimeError(f"移動驗證失敗：來源仍存在 {src_before}")
    if not os.path.exists(dst_after):
        raise RuntimeError(f"移動驗證失敗：目的地不存在 {dst_after}")


def main():
    random.seed(SEED)

    if not os.path.isdir(TRAIN_IMG_DIR):
        raise FileNotFoundError(f"train 影像資料夾不存在：{TRAIN_IMG_DIR}")
    if not os.path.isfile(TRAIN_GT_PATH):
        raise FileNotFoundError(f"找不到 gt.txt：{TRAIN_GT_PATH}")

    gt_by_id = read_gt(TRAIN_GT_PATH)
    all_ids = list_image_ids(TRAIN_IMG_DIR)
    if not all_ids:
        raise RuntimeError(
            f"在 {TRAIN_IMG_DIR} 找不到影像（副檔名限 {IMG_EXTS}，檔名需為數字）"
        )

    n_total = len(all_ids)
    n_val = max(1, int(round(n_total * VAL_RATIO)))
    val_ids = set(random.sample(all_ids, n_val))
    train_ids = [fid for fid in all_ids if fid not in val_ids]

    # 準備 val 影像資料夾
    os.makedirs(VAL_IMG_DIR, exist_ok=True)

    moved = 0
    missing = []
    for fid in sorted(val_ids):
        src = find_image_path(TRAIN_IMG_DIR, fid)
        if not src:
            missing.append(fid)
            continue
        dst = os.path.join(VAL_IMG_DIR, os.path.basename(src))
        # 若目的地已存在，先刪除以避免 shutil.move 變成 rename 失敗
        if os.path.exists(dst):
            os.remove(dst)
        # 執行「移動」
        shutil.move(src, dst)
        # 安全驗證
        ensure_moved(src, dst)
        moved += 1

    # 寫標註（原子操作）
    write_gt_atomic(OUT_VAL_GT, gt_by_id, sorted(val_ids))
    write_gt_atomic(OUT_TRAIN_GT, gt_by_id, train_ids)

    # 報告
    print(f"[Done] total={n_total}, val_target={len(val_ids)}, moved={moved}, train_left={len(train_ids)}")
    print(f"  val images dir : {VAL_IMG_DIR}")
    print(f"  GT files       : {OUT_VAL_GT}, {OUT_TRAIN_GT}")

    if missing:
        print(f"[Warn] 下列 id 在 {TRAIN_IMG_DIR} 未找到對應影像（略過）：{', '.join(missing[:10])}"
              + (" ..." if len(missing) > 10 else ""))

    # 額外一致性檢查：val 的每個 id，在 train/img 不應再有影像
    leftovers = []
    for fid in val_ids:
        if find_image_path(TRAIN_IMG_DIR, fid):
            leftovers.append(fid)
    if leftovers:
        raise RuntimeError(f"[Check] 仍在 {TRAIN_IMG_DIR} 找到下列 val 影像（應已移走）：{leftovers[:10]}")

    print("[OK] All selected val images have been MOVED out of train/img and annotated correctly.")


if __name__ == "__main__":
    main()
