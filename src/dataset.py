# src/dataset.py
import os, glob, cv2, torch
import numpy as np
from torch.utils.data import Dataset

class PigsDataset(Dataset):
    """
    讀取 data/train/img 與 gt.txt
    回傳：image (Tensor[C,H,W], 0~1), target(dict: boxes [N,4], labels [N])
    """
    def __init__(self, img_dir, gt_txt=None, transforms=None):
        self.img_dir = img_dir
        self.transforms = transforms
        self.has_gt = gt_txt is not None

        # 蒐集 00000001.jpg 類型的檔名（不含副檔名）
        self.ids = sorted([os.path.splitext(os.path.basename(p))[0]
                           for p in glob.glob(os.path.join(img_dir, "*.jpg"))])

        # 讀標註
        if self.has_gt:
            self.boxes_by_id = self._read_gt(gt_txt)

    def _read_gt(self, gt_path):
        boxes = {}
        bad = 0
        with open(gt_path, "r", newline="") as f:
            for line in f:
                if not line.strip():
                    continue
                v = [x.strip() for x in line.split(",")]
                if len(v) != 5:
                    continue
                frame, l, t, w, h = v
                fid = f"{int(float(frame)):08d}"
                l, t, w, h = map(int, (l, t, w, h))

                # ① 先過濾原始就不合法的框（寬或高 ≤ 0）
                if w <= 0 or h <= 0:
                    bad += 1
                    continue

                x1, y1, x2, y2 = l, t, l + w, t + h
                boxes.setdefault(fid, []).append([x1, y1, x2, y2])
        if bad:
            print(f"[Dataset] Skipped {bad} invalid boxes from gt (w<=0 or h<=0).")
        return boxes


    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        fid = self.ids[idx]
        img_path = os.path.join(self.img_dir, f"{fid}.jpg")
        img = cv2.imread(img_path)  # BGR
        if img is None:
            raise FileNotFoundError(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.has_gt:
            b = np.array(self.boxes_by_id.get(fid, []), dtype=np.float32)
            labels = np.ones((len(b),), dtype=np.int64)  # 單類別：豬=1
            target = {
                "boxes": torch.as_tensor(b, dtype=torch.float32),
                "labels": torch.as_tensor(labels, dtype=torch.int64),
                "image_id": torch.tensor([int(fid)]),
            }
        else:
            target = {}

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

def collate_fn(batch):
    return tuple(zip(*batch))
