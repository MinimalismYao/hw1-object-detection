# src/modelv7.py
# ============================================
# v7 — Solid & version-safe Faster R-CNN (ResNet50 + FPN, Dropout head + Label Smoothing)
# - 完全不使用預訓練權重（符合課程規範）
# - 加強小物件偵測 (8~96 anchors)
# - 四層 FC head with Dropout(0.2)
# - Label Smoothing (0.1)
# ============================================

from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign, misc as misc_ops


# ---------- helpers ----------
def _maybe_get(d: Dict[str, Any], path: str, default=None):
    cur = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def _sizes_per_level(cfg_model: Dict[str, Any], num_levels: int) -> Tuple[Tuple[int, ...], ...]:
    """將 YAML 的 anchors 轉為 AnchorGenerator 可用格式"""
    per_level = cfg_model.get("rpn_anchor_sizes_per_level", None)
    if isinstance(per_level, (list, tuple)) and len(per_level) > 0:
        out: List[Tuple[int, ...]] = []
        for s in per_level:
            if isinstance(s, (list, tuple)) and len(s) > 0:
                out.append(tuple(int(v) for v in s))
            elif isinstance(s, (int, float)):
                out.append((int(s),))
        out = out[:num_levels] + ([out[-1]] * max(0, num_levels - len(out)))
        return tuple(out)
    # default：小物件密集 anchor
    default = ((8,), (16,), (32,), (64,), (96,))
    return default[:num_levels]

def _ratios_per_level(cfg_model: Dict[str, Any], num_levels: int) -> Tuple[Tuple[float, ...], ...]:
    ratios = cfg_model.get("rpn_anchor_ratios", [0.5, 1.0, 2.0])
    if not (isinstance(ratios, (list, tuple)) and len(ratios) > 0):
        ratios = [0.5, 1.0, 2.0]
    rtuple = tuple(float(r) for r in ratios)
    return (rtuple,) * num_levels

def _norm_from_yaml(norm_str: str):
    ns = (norm_str or "frozenbn").lower()
    if ns in ("frozenbn", "frozen_batchnorm", "frozen"):
        return misc_ops.FrozenBatchNorm2d
    if ns in ("bn", "batchnorm"):
        return nn.BatchNorm2d
    if ns in ("gn", "groupnorm", "groupnorm32"):
        def _gn(ch: int):
            g = 32 if ch % 32 == 0 else (16 if ch % 16 == 0 else 8)
            return nn.GroupNorm(g, ch)
        return _gn
    return misc_ops.FrozenBatchNorm2d


# ---------- 自訂 Dropout Head ----------
class FourFCHead(nn.Module):
    """4層FC + Dropout head"""
    def __init__(self, in_channels: int, pool_res: int = 7, hidden: int = 1024, dropout_p: float = 0.2):
        super().__init__()
        input_dim = in_channels * pool_res * pool_res
        layers = []
        for i in range(4):
            layers += [
                nn.Linear(input_dim if i == 0 else hidden, hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_p),
            ]
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        if x.ndim == 4:
            x = torch.flatten(x, start_dim=1)
        return self.fc(x)


# ---------- 自訂 Label Smoothing Predictor ----------
class SmoothFastRCNNPredictor(FastRCNNPredictor):
    """支援 label smoothing 的 classifier"""
    def __init__(self, in_channels: int, num_classes: int, smoothing: float = 0.1):
        super().__init__(in_channels, num_classes)
        self.smoothing = smoothing
        self.num_classes = num_classes

    def forward(self, x, targets=None):
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        if self.training and targets is not None:
            labels = targets
            smooth = self.smoothing
            with torch.no_grad():
                soft_labels = torch.full(
                    (labels.size(0), self.num_classes),
                    smooth / (self.num_classes - 1),
                    device=labels.device,
                )
                soft_labels.scatter_(1, labels.unsqueeze(1), 1.0 - smooth)
            log_probs = F.log_softmax(scores, dim=1)
            loss_cls = -(soft_labels * log_probs).sum(dim=1).mean()
            return loss_cls, bbox_deltas
        else:
            return scores, bbox_deltas


# ---------- 核心組裝 ----------
def get_frcnn_r50_fpn_from_cfg(cfg: Dict[str, Any], *, num_classes: int) -> FasterRCNN:
    C = cfg or {}
    M = C.get("model", {}) or {}

    # I/O 與 Normalization
    min_size = int(_maybe_get(C, "model.min_size", 1024))
    max_size = int(_maybe_get(C, "model.max_size", 1024))
    image_mean = list(_maybe_get(C, "model.image_mean", (0.485, 0.456, 0.406)))
    image_std  = list(_maybe_get(C, "model.image_std",  (0.229, 0.224, 0.225)))
    norm_layer = _norm_from_yaml(str(_maybe_get(C, "model.norm", "frozenbn")))

    # backbone
    backbone = torchvision.models.detection.backbone_utils.resnet_fpn_backbone(
        "resnet50", weights=None, trainable_layers=3, norm_layer=norm_layer
    )

    fpn_out_ch = getattr(backbone, "out_channels", 256)
    num_levels = 5  # P2~P6

    # Anchors（小物件友好）
    sizes_per_level  = _sizes_per_level(M, num_levels)
    ratios_per_level = _ratios_per_level(M, num_levels)
    anchor_gen = AnchorGenerator(sizes=sizes_per_level, aspect_ratios=ratios_per_level)

    # Faster R-CNN 本體
    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_gen,
        box_roi_pool=MultiScaleRoIAlign(
            featmap_names=("0","1","2","3"), output_size=7, sampling_ratio=2),
        image_mean=image_mean,
        image_std=image_std,
        min_size=min_size,
        max_size=max_size,
    )

    # 替換掉 RoI head 為 Dropout 版
    in_ch = fpn_out_ch
    representation_size = 1024
    model.roi_heads.box_head = FourFCHead(in_ch, pool_res=7, hidden=representation_size, dropout_p=0.2)
    in_features = representation_size

    # Label smoothing 預測器
    model.roi_heads.box_predictor = SmoothFastRCNNPredictor(in_features, num_classes, smoothing=0.1)

    return model


# ---------- 工廠 ----------
def build_detector_from_cfg(cfg: Dict[str, Any]) -> FasterRCNN:
    m = cfg.get("model", {}) if isinstance(cfg, dict) else {}
    num_classes = int(m.get("num_classes", 2))
    return get_frcnn_r50_fpn_from_cfg(cfg, num_classes=num_classes)
