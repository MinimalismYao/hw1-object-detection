# src/transforms.py
# -*- coding: utf-8 -*-
import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
from typing import Dict, Tuple, Optional, List, Union, Any

# 可選：OpenCV 用於 CLAHE（若環境沒裝，會自動關閉 CLAHE）
try:
    import cv2
    _CV2_OK = True
except Exception:
    cv2 = None
    _CV2_OK = False

# =========================================================
# Photometric（只改像素，不改 bbox）—— 一次補齊日/夜 domain gap
# =========================================================
PHOTOMETRIC_CFG: Dict[str, Any] = {
    "p_color_jitter": 0.8,   # 亮度/對比/飽和/色相
    "brightness": 0.25,
    "contrast":   0.25,
    "saturation": 0.25,
    "hue":        0.04,      # 0~0.5

    "p_gray": 0.05,          # 少量灰階

    "p_autocontrast": 0.30,
    "p_equalize":     0.20,

    "p_gamma": 0.60,         # Gamma 校正
    "gamma_range": (0.80, 1.30),

    "p_sharpen": 0.15,       # 輕度銳化／模糊（模擬壓縮/失焦）
    "sharpness": (0.95, 1.15),
    "p_blur": 0.10,
    "blur_radius": (0.4, 0.9),

    "p_clahe": 0.60,         # 夜視強化：自適應直方圖等化（需 cv2）
    "clahe_clip": 2.0,
    "clahe_grid": (8, 8),

    "p_gray_world_wb": 0.35, # Gray-World 白平衡
}

def _pil_to_np_bgr(img: Image.Image) -> np.ndarray:
    arr = np.array(img.convert("RGB"))
    if _CV2_OK:
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return arr[..., ::-1].copy()

def _np_bgr_to_pil(arr_bgr: np.ndarray) -> Image.Image:
    if _CV2_OK:
        arr_rgb = cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2RGB)
    else:
        arr_rgb = arr_bgr[..., ::-1].copy()
    return Image.fromarray(arr_rgb)

def _apply_clahe_pil(img: Image.Image, clip: float, grid: Tuple[int, int]) -> Image.Image:
    if not _CV2_OK:
        return img
    bgr = _pil_to_np_bgr(img)
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid)
    y2 = clahe.apply(y)
    out = cv2.merge([y2, cr, cb])
    out_bgr = cv2.cvtColor(out, cv2.COLOR_YCrCb2BGR)
    return _np_bgr_to_pil(out_bgr)

def _gray_world_wb_pil(img: Image.Image) -> Image.Image:
    """Gray-World：將 RGB channel 的平均值拉齊。"""
    arr = np.asarray(img.convert("RGB")).astype(np.float32)
    mean = arr.reshape(-1, 3).mean(axis=0) + 1e-6
    gray = float(mean.mean())
    scale = gray / mean
    arr = np.clip(arr * scale, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

class PhotometricAug:
    """不改 bbox 的光度增強；建議僅在 train 啟用。"""
    def __init__(self, cfg: Dict[str, Any] = PHOTOMETRIC_CFG):
        self.cfg = dict(cfg)

        # 若沒 cv2，關閉 CLAHE
        if not _CV2_OK:
            self.cfg["p_clahe"] = 0.0

        # 預先建 ColorJitter
        self.cj = T.ColorJitter(
            brightness=self.cfg["brightness"],
            contrast=self.cfg["contrast"],
            saturation=self.cfg["saturation"],
            hue=self.cfg["hue"],
        )

    def __call__(self, img: Image.Image, target: Dict) -> Tuple[Image.Image, Dict]:
        c = self.cfg

        # 1) ColorJitter
        if torch.rand(1).item() < c["p_color_jitter"]:
            img = self.cj(img)

        # 2) 少量灰階
        if torch.rand(1).item() < c["p_gray"]:
            img = ImageOps.grayscale(img).convert("RGB")

        # 3) 自動對比/等化
        if torch.rand(1).item() < c["p_autocontrast"]:
            img = ImageOps.autocontrast(img)
        if torch.rand(1).item() < c["p_equalize"]:
            img = ImageOps.equalize(img)

        # 4) Gamma
        if torch.rand(1).item() < c["p_gamma"]:
            g = float(np.random.uniform(*c["gamma_range"]))
            t = T.functional.adjust_gamma(T.functional.to_tensor(img), gamma=g, gain=1.0)
            img = T.functional.to_pil_image(t)

        # 5) 輕度銳化/模糊
        if torch.rand(1).item() < c["p_sharpen"]:
            k = float(np.random.uniform(*c["sharpness"]))
            img = ImageEnhance.Sharpness(img).enhance(k)
        if torch.rand(1).item() < c["p_blur"]:
            r = float(np.random.uniform(*c["blur_radius"]))
            img = img.filter(ImageFilter.GaussianBlur(radius=r))

        # 6) 夜視強化 CLAHE（Y 通道）
        if torch.rand(1).item() < c["p_clahe"]:
            img = _apply_clahe_pil(img, c["clahe_clip"], c["clahe_grid"])

        # 7) Gray-World 白平衡
        if torch.rand(1).item() < c["p_gray_world_wb"]:
            img = _gray_world_wb_pil(img)

        return img, target


# ===========================
# 幾何與 bbox 相關工具
# ===========================
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


# ===========================
# 主要 Transform 類別
# ===========================
class TrainTransforms:
    def __init__(
        self,
        max_side: int = 800,
        flip_p: float = 0.5,
        hsv: Optional[List[float]] = None,        # 期望 [h, s, v]；若為 bool/None 則忽略
        resize_list: Optional[List[int]] = None,
        mosaic: bool = False,
        min_box_size: float = 1.0,
        color_jitter_prob: float = 0.0,           # 0~1（若 color_jitter/hsv 有設，會自動啟用）
        color_jitter: Optional[List[float]] = None, # [b,c,s,h]
        enable_photometric: bool = True,          # ★ 新增：是否啟用 PhotometricAug
        photometric_cfg: Dict[str, Any] = PHOTOMETRIC_CFG,
    ):
        self.max_side = int(max_side)
        self.flip_p = float(flip_p)
        self.resize_list = resize_list if (resize_list and len(resize_list) > 0) else None
        self.mosaic = bool(mosaic)
        self.min_box_size = float(min_box_size)

        # Photometric（不動 bbox）
        self.photo = PhotometricAug(photometric_cfg) if enable_photometric else None

        # 顏色增強策略：優先 color_jitter，否則根據 hsv
        self.cj_prob = float(color_jitter_prob)
        if color_jitter and len(color_jitter) == 4:
            b, c, s, h = map(float, color_jitter)
            self.color = T.ColorJitter(brightness=b, contrast=c, saturation=s, hue=h)
            self.cj_prob = max(self.cj_prob, 0.8)
        elif isinstance(hsv, (list, tuple)) and len(hsv) == 3:
            # YOLO 式 hsv: [h, s, v] -> ColorJitter(hue, saturation, brightness)
            h, s, v = float(hsv[0]), float(hsv[1]), float(hsv[2])
            hue = max(0.0, min(abs(h), 0.5))
            self.color = T.ColorJitter(brightness=max(0.0, v), contrast=0.0, saturation=max(0.0, s), hue=hue)
            self.cj_prob = max(self.cj_prob, 0.8)
        else:
            self.color = None
            self.cj_prob = 0.0

    def __call__(self, img_np: np.ndarray, target: Dict):
        img = Image.fromarray(img_np)

        # Photometric（不改 bbox）—— 放最前，與幾何無相依
        if self.photo is not None:
            img, target = self.photo(img, target)

        # (預留) mosaic 未實作；若要實作請放在此處，並同步調整 bbox
        if self.resize_list is not None:
            short_choice = int(self.resize_list[torch.randint(low=0, high=len(self.resize_list), size=(1,)).item()])
            img, target, _, _ = _resize_short_to_value(img, target, short_choice)

        # 長邊對齊
        img, target, _, _ = _resize_long_to_max_side(img, target, self.max_side)

        # 水平翻轉
        if "boxes" in target and len(target["boxes"]) > 0 and self.flip_p > 0.0:
            if torch.rand(1).item() < self.flip_p:
                w, _ = img.size
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                target["boxes"] = _hflip_boxes(target["boxes"], w)

        # 顏色增強（可選——與 Photometric 相比幅度較小）
        if self.color is not None and torch.rand(1).item() < self.cj_prob:
            img = self.color(img)

        # clip 與過小框過濾
        w_final, h_final = img.size
        if "boxes" in target and len(target["boxes"]) > 0:
            if not isinstance(target["boxes"], torch.Tensor):
                target["boxes"] = torch.as_tensor(target["boxes"], dtype=torch.float32)
            target["boxes"], keep = _clip_and_filter_boxes(target["boxes"], w_final, h_final, self.min_box_size)
            if "labels" in target:
                target["labels"] = target["labels"][keep]

        # 只轉 tensor（Normalize 交給 model 端的 GeneralizedRCNNTransform）
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

        img = T.ToTensor()(img)  # 不做 Normalize
        return img, target


def get_transforms(
    train: bool = True,
    *,
    max_side: int = 800,
    flip_p: float = 0.5,
    hsv: Optional[Union[List[float], Tuple[float, float, float]]] = None,
    resize: Optional[Union[List[int], Tuple[int, ...]]] = None,
    mosaic: bool = False,
    min_box_size: float = 1.0,
    color_jitter_prob: float = 0.0,
    color_jitter: Optional[Union[List[float], Tuple[float, float, float, float]]] = None,
    enable_photometric: bool = True,            # ★ train 預設啟用 PhotometricAug
    photometric_cfg: Dict[str, Any] = PHOTOMETRIC_CFG,
):
    if train:
        resize_list = list(resize) if resize is not None else None
        return TrainTransforms(
            max_side=max_side,
            flip_p=flip_p,
            hsv=list(hsv) if hsv is not None else None,
            resize_list=resize_list,
            mosaic=mosaic,
            min_box_size=min_box_size,
            color_jitter_prob=color_jitter_prob,
            color_jitter=list(color_jitter) if color_jitter is not None else None,
            enable_photometric=enable_photometric,
            photometric_cfg=photometric_cfg,
        )
    else:
        return ValTransforms(max_side=max_side, min_box_size=min_box_size)
