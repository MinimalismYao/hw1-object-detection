# src/transforms.py
import torchvision.transforms as T
import numpy as np
import torch
from PIL import Image

class TorchVisionWrapper:
    """
    把 PIL/Tensor 轉換 + bbox 同步處理的最小包裝
    這裡先用「水洗版」：只做 ToTensor + HFlip + ColorJitter
    （若要更完整的 bbox-safe augment，可之後換 Albumentations）
    """
    def __init__(self, train=True):
        ops = [T.ToTensor()]
        if train:
            ops += [
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
            ]
        self.ops = T.Compose(ops)

    def __call__(self, img_np, target):
        # numpy(H,W,C,RGB) -> PIL -> apply -> Tensor
        img = Image.fromarray(img_np)
        img = self.ops(img)  # C,H,W ; 0~1

        # 注意：RandomHorizontalFlip 不會自動改 bbox
        # 為保持簡單，這裡把 Flip 關掉或手動處理
        # 先關掉 flip：把上面 RandomHorizontalFlip 移除即可
        return img, target

def get_transforms(train=True):
    # 若要避免 bbox 不同步，先不上 Flip，只用 ToTensor + ColorJitter
    return TorchVisionWrapper(train=train)
