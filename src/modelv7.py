# src/modelv7.py
# ============================================
# v7 — Solid & version-safe Faster R-CNN (ResNet50 + FPN, Dropout head)
# - 完全不使用預訓練權重（符合課程規範）
# - 小物件友好 anchors（預設 8~96，亦可由 YAML 覆蓋）
# - 四層 FC head，含 Dropout（可由 YAML 設定 dropout_p）
# - 完整相容 YAML：anchors / rpn / roi / minmax / meanstd / norm
# ============================================

from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
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
    # default：小物件密集 anchor（P2..P6）
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
    def __init__(self, in_channels: int, pool_res: int = 7, hidden: int = 1024,
                 num_fc: int = 4, dropout_p: float = 0.2):
        super().__init__()
        input_dim = in_channels * pool_res * pool_res
        layers: List[nn.Module] = []
        for i in range(num_fc):
            layers += [
                nn.Linear(input_dim if i == 0 else hidden, hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_p),
            ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        if x.ndim == 4:
            x = torch.flatten(x, start_dim=1)
        return self.net(x)


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

    # backbone（禁止任何預訓練權重）
    backbone = torchvision.models.detection.backbone_utils.resnet_fpn_backbone(
        backbone_name="resnet50", weights=None, trainable_layers=3, norm_layer=norm_layer
    )


    fpn_out_ch = getattr(backbone, "out_channels", 256)
    num_levels = 5  # P2~P6

    # Anchors（小物件友好）
    sizes_per_level  = _sizes_per_level(M, num_levels)
    ratios_per_level = _ratios_per_level(M, num_levels)
    anchor_gen = AnchorGenerator(sizes=sizes_per_level, aspect_ratios=ratios_per_level)

    # RoI / RPN 其他參數（從 YAML 取，帶預設）
    roi_cfg = {
        "score_thresh": float(_maybe_get(C, "model.roi.score_thresh", 0.0)),
        "nms_thresh":   float(_maybe_get(C, "model.roi.nms_thresh",   0.5)),
        "detections":   int(_maybe_get(C, "model.roi.detections_per_img", 150)),
        "pool":         int(_maybe_get(C, "model.roi.roi_pool_size", 7)),
        "bbox_weights": tuple(_maybe_get(C, "model.roi.bbox_reg_weights", [1.0, 1.0, 1.0, 1.0])),
        "box_batch":    int(_maybe_get(C, "model.roi.box_batch_size_per_image", 512)),
        "box_pos":      float(_maybe_get(C, "model.roi.box_positive_fraction", 0.25)),
        "fg_iou":       float(_maybe_get(C, "model.roi.fg_iou_thresh", 0.5)),
        "bg_iou":       float(_maybe_get(C, "model.roi.bg_iou_thresh", 0.5)),
        "dropout_p":    float(_maybe_get(C, "model.roi.dropout_p", 0.30)),
    }
    rpn_cfg = {
        "batch_per_img": int(_maybe_get(C, "model.rpn.batch_size_per_image", 256)),
        "pos_frac":      float(_maybe_get(C, "model.rpn.positive_fraction", 0.5)),
        "fg_iou":        float(_maybe_get(C, "model.rpn.fg_iou_thresh", 0.5)),
        "bg_iou":        float(_maybe_get(C, "model.rpn.bg_iou_thresh", 0.0)),
        "pre_train":     int(_maybe_get(C, "model.rpn.pre_nms_top_n_train", 2000)),
        "post_train":    int(_maybe_get(C, "model.rpn.post_nms_top_n_train", 1000)),
        "pre_test":      int(_maybe_get(C, "model.rpn.pre_nms_top_n_test",  1000)),
        "post_test":     int(_maybe_get(C, "model.rpn.post_nms_top_n_test", 1000)),
        "nms":           float(_maybe_get(C, "model.rpn.nms_thresh", 0.7)),
    }

    roi_pool_size = roi_cfg["pool"]

    # Faster R-CNN 本體
    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_gen,
        box_roi_pool=MultiScaleRoIAlign(
            featmap_names=("0", "1", "2", "3"),
            output_size=roi_pool_size,
            sampling_ratio=2,
        ),
        box_score_thresh=roi_cfg["score_thresh"],
        box_nms_thresh=roi_cfg["nms_thresh"],
        box_detections_per_img=roi_cfg["detections"],
        image_mean=image_mean,
        image_std=image_std,
        min_size=min_size,
        max_size=max_size,
    )

    # 覆蓋 RPN 設定
    rpn = model.rpn
    rpn.batch_size_per_image        = rpn_cfg["batch_per_img"]
    rpn.positive_fraction           = rpn_cfg["pos_frac"]
    rpn.foreground_iou_threshold    = rpn_cfg["fg_iou"]
    rpn.background_iou_threshold    = rpn_cfg["bg_iou"]
    rpn.nms_thresh                  = rpn_cfg["nms"]
    if hasattr(rpn, "pre_nms_top_n_train"):  rpn.pre_nms_top_n_train  = rpn_cfg["pre_train"]
    if hasattr(rpn, "pre_nms_top_n_test"):   rpn.pre_nms_top_n_test   = rpn_cfg["pre_test"]
    if hasattr(rpn, "post_nms_top_n_train"): rpn.post_nms_top_n_train = rpn_cfg["post_train"]
    if hasattr(rpn, "post_nms_top_n_test"):  rpn.post_nms_top_n_test  = rpn_cfg["post_test"]

    # 覆蓋 RoI 設定（採樣、邊界框編碼權重）
    if hasattr(model.roi_heads, "box_coder") and hasattr(model.roi_heads.box_coder, "weights"):
        model.roi_heads.box_coder.weights = roi_cfg["bbox_weights"]
    if hasattr(model.roi_heads, "batch_size_per_image"):
        model.roi_heads.batch_size_per_image = roi_cfg["box_batch"]
    if hasattr(model.roi_heads, "positive_fraction"):
        model.roi_heads.positive_fraction = roi_cfg["box_pos"]
    if hasattr(model.roi_heads, "fg_iou_thresh"):
        model.roi_heads.fg_iou_thresh = roi_cfg["fg_iou"]
    if hasattr(model.roi_heads, "bg_iou_thresh"):
        model.roi_heads.bg_iou_thresh = roi_cfg["bg_iou"]

    # 4×FC Dropout Head
    model.roi_heads.box_head = FourFCHead(
        in_channels=fpn_out_ch,
        pool_res=roi_pool_size,
        hidden=1024,
        num_fc=4,
        dropout_p=roi_cfg["dropout_p"],
    )

    # 重新設定類別數（保險再做一次）
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


# ---------- 工廠 ----------
def build_detector_from_cfg(cfg: Dict[str, Any]) -> FasterRCNN:
    """
    任意 model.detector 名稱一律回到這個穩定組裝版（向後相容）。
    """
    m = cfg.get("model", {}) if isinstance(cfg, dict) else {}
    num_classes = int(m.get("num_classes", 2))
    return get_frcnn_r50_fpn_from_cfg(cfg, num_classes=num_classes)
