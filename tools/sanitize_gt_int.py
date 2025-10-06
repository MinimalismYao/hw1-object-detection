#!/usr/bin/env python3
from pathlib import Path
import shutil

GT = Path("data/train/gt_train.txt")
OUT = GT.with_suffix(".clean.txt")
BAK = GT.with_suffix(".bak")

def to_int(x: str) -> int:
    return int(round(float(x.strip())))

bad, kept = 0, 0
with GT.open("r", encoding="utf-8") as fin, OUT.open("w", encoding="utf-8") as fout:
    for line in fin:
        line = line.strip()
        if not line or line.startswith("#"): 
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 5:
            bad += 1
            continue
        fid, l, t, w, h = parts
        try:
            l, t, w, h = map(to_int, (l, t, w, h))
        except Exception:
            bad += 1
            continue
        if w <= 0 or h <= 0:
            bad += 1
            continue
        fout.write(f"{int(fid)},{l},{t},{w},{h}\n")
        kept += 1

print(f"[sanitize] kept={kept}, dropped(bad)={bad}")
# 備份原檔，換成 new
shutil.copy2(GT, BAK)
OUT.replace(GT)
print(f"[sanitize] backup -> {BAK}")
