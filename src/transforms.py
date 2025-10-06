# src/transforms.py
import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image
from typing import Dict, Tuple, Optional, List, Union

def _clip_and_filter_boxes(boxes: torch.Tensor, w: int, h: int, min_size: float = 1.0):
    boxes[:, 0::2] = boxes[:, 0::2].clamp(0, w - 1)
    boxes[:, 1::2] = boxes[:, 1::2].clamp(0, h - 1)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    ws = x2 - x1
    hs = y2 - y1
    keep = (ws >= min_size) & (hs >= min_size)
    return boxes[keep], keep

def _hflip_boxes(boxes: torch.Tensor, width: int):
    flipped = boxes.clone()
    flipped[:, 0] = width - boxes[:, 2]
    flipped[:, 2] = width - boxes[:, 0]
    return flipped

def _scale_boxes(target: Dict, sx: float, sy: float):
    if "boxes" in target and len(target["boxes"]) > 0:
        boxes = target["boxes"].clone()
        boxes[:, [0, 2]] *= sx
        boxes[:, [1, 3]] *= sy
        target["boxes"] = boxes
    return target

def _resize_keep_aspect(img_pil: Image.Image, target: Dict, new_w: int, new_h: int):
    w, h = img_pil.size
    if (new_w, new_h) == (w, h):
        return img_pil, target, 1.0, 1.0
    img_resized = img_pil.resize((new_w, new_h), resample=Image.BILINEAR)
    sx = new_w / w
    sy = new_h / h
    target = _scale_boxes(target, sx, sy)
    return img_resized, target, sx, sy

def _resize_long_to_max_side(img_pil: Image.Image, target: Dict, max_side: int):
    w, h = img_pil.size
    long_side = max(w, h)
    if long_side <= max_side:
        return img_pil, target, 1.0, 1.0
    scale = max_side / long_side
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    return _resize_keep_aspect(img_pil, target, new_w, new_h)

def _resize_short_to_value(img_pil: Image.Image, target: Dict, short_side_value: int):
    w, h = img_pil.size
    short_side = min(w, h)
    if short_side >= short_side_value:
        return img_pil, target, 1.0, 1.0
    scale = short_side_value / short_side
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    return _resize_keep_aspect(img_pil, target, new_w, new_h)

class TrainTransforms:
    def __init__(
        self,
        max_side: int = 800,
        flip_p: float = 0.5,
        hsv: bool = True,
        resize_list: Optional[List[int]] = None,
        mosaic: bool = False,
        min_box_size: float = 1.0,
    ):
        self.max_side = int(max_side)
        self.flip_p = float(flip_p)
        self.use_color = bool(hsv)
        self.resize_list = resize_list if (resize_list and len(resize_list) > 0) else None
        self.mosaic = bool(mosaic)
        self.min_box_size = float(min_box_size)
        self.color = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)

    def __call__(self, img_np: np.ndarray, target: Dict):
        img = Image.fromarray(img_np)

        # (預留) mosaic 未實作
        if self.resize_list is not None:
            short_choice = int(self.resize_list[torch.randint(low=0, high=len(self.resize_list), size=(1,)).item()])
            img, target, _, _ = _resize_short_to_value(img, target, short_choice)

        img, target, _, _ = _resize_long_to_max_side(img, target, self.max_side)

        if "boxes" in target and len(target["boxes"]) > 0 and self.flip_p > 0.0:
            if torch.rand(1).item() < self.flip_p:
                w, _ = img.size
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                target["boxes"] = _hflip_boxes(target["boxes"], w)

        if self.use_color:
            img = self.color(img)

        w_final, h_final = img.size
        if "boxes" in target and len(target["boxes"]) > 0:
            if not isinstance(target["boxes"], torch.Tensor):
                target["boxes"] = torch.as_tensor(target["boxes"], dtype=torch.float32)
            target["boxes"], keep = _clip_and_filter_boxes(target["boxes"], w_final, h_final, self.min_box_size)
            if "labels" in target:
                target["labels"] = target["labels"][keep]

        img = T.ToTensor()(img)
        return img, target

class ValTransforms:
    def __init__(self, max_side: int = 800, min_box_size: float = 1.0):
        self.max_side = int(max_side)
        self.min_box_size = float(min_box_size)

    def __call__(self, img_np: np.ndarray, target: Dict):
        img = Image.fromarray(img_np)
        img, target, _, _ = _resize_long_to_max_side(img, target, self.max_side)

        w_final, h_final = img.size
        if "boxes" in target and len(target["boxes"]) > 0:
            if not isinstance(target["boxes"], torch.Tensor):
                target["boxes"] = torch.as_tensor(target["boxes"], dtype=torch.float32)
            target["boxes"], keep = _clip_and_filter_boxes(target["boxes"], w_final, h_final, self.min_box_size)
            if "labels" in target:
                target["labels"] = target["labels"][keep]

        img = T.ToTensor()(img)
        return img, target

def get_transforms(
    train: bool = True,
    *,
    max_side: int = 800,
    flip_p: float = 0.5,
    hsv: bool = True,
    resize: Optional[Union[List[int], Tuple[int, ...]]] = None,
    mosaic: bool = False,
    min_box_size: float = 1.0,
):
    if train:
        resize_list = list(resize) if resize is not None else None
        return TrainTransforms(
            max_side=max_side,
            flip_p=flip_p,
            hsv=hsv,
            resize_list=resize_list,
            mosaic=mosaic,
            min_box_size=min_box_size,
        )
    else:
        return ValTransforms(max_side=max_side, min_box_size=min_box_size)
