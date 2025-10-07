# src/dataset.py
import os, glob, cv2, torch
import numpy as np
from torch.utils.data import Dataset

_VALID_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP")

def _list_image_ids(img_dir):
    paths = []
    for ext in _VALID_EXTS:
        paths.extend(glob.glob(os.path.join(img_dir, f"*{ext}")))
    # 轉「數字 id」→ 零補 8 碼字串；非數字的檔名略過
    nums = []
    for p in paths:
        stem = os.path.splitext(os.path.basename(p))[0]
        s2 = stem.lstrip("0")
        if s2 == "":  # 全是 0
            n = 0
        else:
            try:
                n = int(s2)
            except Exception:
                continue  # 非數字檔名略過，避免後續 int(fid) 失敗
        nums.append(n)
    nums = sorted(set(nums))
    # 統一成 8 碼零補字串，跟 gt 的 key 對齊
    ids = [f"{n:08d}" for n in nums]
    return ids


class PigsDataset(Dataset):
    """
    讀取 data/*/img 與 gt.txt
    回傳：image (Tensor[C,H,W], 0~1), target(dict: boxes [N,4], labels [N])
    - boxes: xyxy (float32)
    - labels: int64，單類別 → 全部為 1（RetinaNet/FRCNN 的 0/1 基底差異由訓練端處理）
    """
    def __init__(self, img_dir, gt_txt=None, transforms=None):
        self.img_dir = img_dir
        self.transforms = transforms
        self.has_gt = gt_txt is not None

        # 蒐集影像 ID（不含副檔名）
        self.ids = _list_image_ids(img_dir)

        # 讀標註
        if self.has_gt:
            self.boxes_by_id = self._read_gt(gt_txt)

    def _read_gt(self, gt_path):
        boxes = {}
        bad = 0
        with open(gt_path, "r", encoding="utf-8", newline="") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                # 支援逗號或空白
                parts = [x for x in s.replace(",", " ").split() if x]
                if len(parts) < 5:
                    continue
                frame, l, t, w, h = parts[:5]
                # 允許小數 → 四捨五入後轉 int
                try:
                    fid_int = int(round(float(frame)))
                    l = int(round(float(l))); t = int(round(float(t)))
                    w = int(round(float(w))); h = int(round(float(h)))
                except Exception:
                    continue

                # ① 原始就不合法的框（寬或高 ≤ 0）丟棄
                if w <= 0 or h <= 0:
                    bad += 1
                    continue

                fid = f"{fid_int:08d}"
                x1, y1, x2, y2 = l, t, l + w, t + h
                boxes.setdefault(fid, []).append([x1, y1, x2, y2])

        if bad:
            print(f"[Dataset] Skipped {bad} invalid boxes from gt (w<=0 or h<=0).")
        # 轉成 float32 tensor
        for k, v in list(boxes.items()):
            boxes[k] = torch.as_tensor(v, dtype=torch.float32)
        return boxes

    def __len__(self):
        return len(self.ids)

    def _resolve_img_path(self, fid):
        # 嘗試多個副檔名
        for ext in _VALID_EXTS:
            p = os.path.join(self.img_dir, f"{fid}{ext}")
            if os.path.isfile(p):
                return p
        # 若找不到，退回 .jpg（讓錯誤訊息更一致）
        return os.path.join(self.img_dir, f"{fid}.jpg")

    def __getitem__(self, idx):
        fid = self.ids[idx]
        img_path = self._resolve_img_path(fid)
        img = cv2.imread(img_path)  # BGR
        if img is None:
            raise FileNotFoundError(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.has_gt:
            b = self.boxes_by_id.get(fid, torch.zeros((0, 4), dtype=torch.float32))
            if not isinstance(b, torch.Tensor):
                b = torch.as_tensor(b, dtype=torch.float32)
            labels = torch.ones((b.shape[0],), dtype=torch.int64)  # 單類別：豬=1
            target = {
                "boxes": b,
                "labels": labels,
                "image_id": torch.tensor(int(fid), dtype=torch.int64),
            }
        else:
            target = {"image_id": torch.tensor(int(fid), dtype=torch.int64)}

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

def collate_fn(batch):
    return tuple(zip(*batch))
