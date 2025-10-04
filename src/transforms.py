# src/transforms.py
import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image

def _clip_and_filter_boxes(boxes: torch.Tensor, w: int, h: int, min_size: float = 1.0):
    """裁邊到影像內，並過濾掉寬或高小於 min_size 的框"""
    boxes[:, 0::2] = boxes[:, 0::2].clamp(0, w - 1)
    boxes[:, 1::2] = boxes[:, 1::2].clamp(0, h - 1)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    ws = x2 - x1
    hs = y2 - y1
    keep = (ws >= min_size) & (hs >= min_size)
    return boxes[keep], keep

def _hflip_boxes(boxes, width: int):
    """水平翻轉 bboxes"""
    flipped = boxes.clone()
    flipped[:, 0] = width - boxes[:, 2]
    flipped[:, 2] = width - boxes[:, 0]
    return flipped

def _resize_img_and_boxes(img_pil, target, max_side=800):
    """等比縮放，讓最長邊 = max_side"""
    w, h = img_pil.size
    scale = max_side / max(h, w)
    if scale >= 1.0:  # 小圖就不放大
        return img_pil, target, 1.0

    new_w, new_h = int(w * scale), int(h * scale)
    img_resized = img_pil.resize((new_w, new_h), resample=Image.BILINEAR)

    if "boxes" in target and len(target["boxes"]) > 0:
        boxes = target["boxes"].clone()
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * (new_w / w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * (new_h / h)
        target["boxes"] = boxes

    return img_resized, target, scale


class TrainTransforms:
    def __init__(self, do_flip=True, do_color=True, do_resize=True, max_side=800):
        self.do_flip = do_flip
        self.do_color = do_color
        self.do_resize = do_resize
        self.max_side = max_side
        self.color = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)

    def __call__(self, img_np, target):
        img = Image.fromarray(img_np)  # HWC RGB -> PIL

        # resize（等比縮放）
        if self.do_resize:
            img, target, _ = _resize_img_and_boxes(img, target, max_side=self.max_side)

        # flip（50% 機率）
        if self.do_flip and "boxes" in target and len(target["boxes"]) > 0:
            if torch.rand(1).item() < 0.5:
                w, h = img.size
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                target["boxes"] = _hflip_boxes(target["boxes"], w)

        # color jitter
        if self.do_color:
            img = self.color(img)

        # ✨ bbox 檢查 + 過濾 ✨
        w_final, h_final = img.size
        if "boxes" in target and len(target["boxes"]) > 0:
            if not isinstance(target["boxes"], torch.Tensor):
                target["boxes"] = torch.as_tensor(target["boxes"], dtype=torch.float32)
            target["boxes"], keep = _clip_and_filter_boxes(target["boxes"], w_final, h_final)
            if "labels" in target:
                target["labels"] = target["labels"][keep]

        # ToTensor（0~1, C,H,W）
        img = T.ToTensor()(img)
        return img, target


class ValTransforms:
    def __init__(self, max_side=800):
        self.max_side = max_side

    def __call__(self, img_np, target):
        img = Image.fromarray(img_np)
        img, target, _ = _resize_img_and_boxes(img, target, max_side=self.max_side)

        # ✨ bbox 檢查 + 過濾 ✨
        w_final, h_final = img.size
        if "boxes" in target and len(target["boxes"]) > 0:
            if not isinstance(target["boxes"], torch.Tensor):
                target["boxes"] = torch.as_tensor(target["boxes"], dtype=torch.float32)
            target["boxes"], keep = _clip_and_filter_boxes(target["boxes"], w_final, h_final)
            if "labels" in target:
                target["labels"] = target["labels"][keep]

        img = T.ToTensor()(img)
        return img, target


def get_transforms(train=True, max_side=800):
    return TrainTransforms(max_side=max_side) if train else ValTransforms(max_side=max_side)
