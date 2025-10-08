# check_ids.py
from pathlib import Path
import csv

TEST_DIR = Path("data/test/img")
SAMPLE_SUB = Path("submissions/sample_submission.csv")  # ← 檢查你的實際路徑

# 蒐集 test 檔名（去副檔名）
test_ids = sorted([p.stem for p in TEST_DIR.iterdir() if p.suffix.lower() in (".jpg",".jpeg",".png",".bmp")])

# 蒐集 sample_submission 的 Image_ID
with open(SAMPLE_SUB, "r") as f:
    reader = csv.DictReader(f)
    sub_ids = [row["Image_ID"].strip() for row in reader]

print(f"[Count] test images = {len(test_ids)} | sample rows = {len(sub_ids)}")

# 直接用字串集合比對
s_test = set(test_ids)
s_sub  = set(sub_ids)

only_in_test = sorted(list(s_test - s_sub))[:20]
only_in_sub  = sorted(list(s_sub  - s_test))[:20]

print(f"[Mismatch] only_in_test (show up to 20): {only_in_test}")
print(f"[Mismatch] only_in_sub  (show up to 20): {only_in_sub}")

# 額外：把 test 轉成「去前導零的整數字串」再比一次（某些競賽會要這種）
def strip_zero(s): 
    t = s.lstrip("0")
    return t if t != "" else "0"

s_test_int = set(strip_zero(s) for s in test_ids)
only_int_in_sub = sorted(list(set(sub_ids) - s_test_int))[:20]
print(f"[Alt check] sample - stripZero(test) (up to 20): {only_int_in_sub}")
