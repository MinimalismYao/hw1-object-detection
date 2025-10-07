# src/modelv7.py
# ============================================
# v7 — Minimal & stable Faster R-CNN builder (R101 + FPN P2-P6, GN/FrozenBN, Big RoI Head)
# ============================================

from typing import Any, Dict, Iterable, Optional, Tuple, List

import torch
import torch.nn as nn
import torchvision
from torchvision.ops import MultiScaleRoIAlign, misc as misc_ops
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import BackboneWithFPN

# ---------- helpers ----------
def _maybe_get(d: Dict[str, Any], path: str, default=None):
    cur = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def _make_norm(norm_type: str):
    nt = (norm_type or "groupnorm32").lower()
    if nt in ("bn", "batchnorm"):
        return nn.BatchNorm2d
    if nt in ("frozenbn", "frozen_batchnorm"):
        return misc_ops.FrozenBatchNorm2d
    if nt in ("groupnorm", "groupnorm32", "gn", "gn32"):
        def _gn(ch):
            g = 32 if ch % 32 == 0 else (16 if ch % 16 == 0 else 8)
            return nn.GroupNorm(g, ch)
        return _gn
    return nn.BatchNorm2d

def _normalize_sizes_per_level(cfg_model: Dict[str, Any], num_levels: int) -> Tuple[Tuple[int, ...], ...]:
    """
    產生 AnchorGenerator 需要的 sizes（每層一個 tuple），並保證長度 == num_levels。
    允許三種輸入：
      1) rpn_anchor_sizes_per_level: [[32],[64],[96],[128],[192]]
      2) rpn_anchor_sizes: [32,64,96,128,192]  # 視為每層主尺度
      3) rpn_anchor_sizes: [16,24,32]          # 共用同一組尺寸（不推薦，但支援）；會廣播到所有層
    """
    # ① 明確逐層
    per_level = cfg_model.get("rpn_anchor_sizes_per_level", None)
    if isinstance(per_level, (list, tuple)) and len(per_level) > 0:
        out: List[Tuple[int, ...]] = []
        for s in per_level:
            if isinstance(s, (list, tuple)) and len(s) > 0:
                out.append(tuple(int(v) for v in s))
            elif isinstance(s, (int, float)):
                out.append((int(s),))
        # 裁切/補齊為剛好 num_levels
        out = out[:num_levels] + ([out[-1]] * max(0, num_levels - len(out)))
        return tuple(out)

    # ② 同長度 list，視為每層主尺度
    sizes = cfg_model.get("rpn_anchor_sizes", None)
    if isinstance(sizes, (list, tuple)) and len(sizes) == num_levels and all(isinstance(x, (int, float)) for x in sizes):
        return tuple((int(x),) for x in sizes)

    # ③ 預設穩健主尺度
    if sizes is None:
        base = (32, 64, 96, 128, 192) if num_levels == 5 else (32, 64, 96, 128)
        return tuple((s,) for s in base)

    # ④ 共用一組尺寸（廣播到所有層）
    if isinstance(sizes, (int, float)):
        sizes = [int(sizes)]
    if isinstance(sizes, (list, tuple)) and all(isinstance(x, (int, float)) for x in sizes):
        t = tuple((int(x),) for x in sizes)
        # 廣播/補齊
        return (t * (num_levels // len(t))) + tuple([t[-1]] * (num_levels % len(t))) if len(t) < num_levels else t[:num_levels]

    # fallback
    return tuple((32,),) * num_levels

def _normalize_ratios_per_level(cfg_model: Dict[str, Any], num_levels: int) -> Tuple[Tuple[float, ...], ...]:
    ratios = cfg_model.get("rpn_anchor_ratios", [0.5, 1.0, 2.0])
    if isinstance(ratios, (list, tuple)) and len(ratios) > 0 and all(isinstance(r, (int, float)) for r in ratios):
        rtuple = tuple(float(r) for r in ratios)
    else:
        rtuple = (0.5, 1.0, 2.0)
    return (rtuple,) * num_levels

# ---------- Box head：4×FC(1024) ----------
class FourFCHead(nn.Module):
    def __init__(self, in_channels: int, pool_res: int = 14, hidden: int = 1024, num_fc: int = 4):
        super().__init__()
        input_dim = in_channels * pool_res * pool_res
        layers = []
        for i in range(num_fc):
            layers += [nn.Linear(input_dim if i == 0 else hidden, hidden), nn.ReLU(inplace=True)]
        self.fc = nn.Sequential(*layers)
    def forward(self, x):
        return self.fc(torch.flatten(x, 1))

# ---------- 核心建構 ----------
def get_fasterrcnn_r101_fpn(
    num_classes: int = 2,
    *,
    cfg: Optional[Dict[str, Any]] = None,
    min_size: Optional[int] = None,
    max_size: Optional[int] = None,
    image_mean: Iterable[float] = (0.485, 0.456, 0.406),
    image_std:  Iterable[float] = (0.229, 0.224, 0.225),
) -> FasterRCNN:
    C = cfg or {}
    # 尺度與正規化
    min_size = int(_maybe_get(C, "model.min_size", min_size) or 1280)
    max_size = int(_maybe_get(C, "model.max_size", max_size) or 1280)
    image_mean = list(_maybe_get(C, "model.image_mean", image_mean))
    image_std  = list(_maybe_get(C, "model.image_std",  image_std))

    # FPN / Norm / P6
    norm_type = str(_maybe_get(C, "model.norm", "groupnorm32"))
    fpn_out   = int(_maybe_get(C, "model.fpn_out_channels", 256))
    use_p6    = bool(_maybe_get(C, "model.use_p6", True))
    norm_layer = _make_norm(norm_type)

    # ----- backbone + FPN (R101, GN/FrozenBN, +P6) -----
    resnet = torchvision.models.resnet101(weights=None, norm_layer=norm_layer)
    return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
    in_channels_list = [256, 512, 1024, 2048]
    extra_blocks = torchvision.ops.feature_pyramid_network.LastLevelMaxPool() if use_p6 else None
    backbone = BackboneWithFPN(
        resnet, return_layers, in_channels_list, fpn_out, extra_blocks=extra_blocks
    )

    # FPN 層數（與 anchor 完全對齊）
    num_levels = 5 if use_p6 else 4
    sizes_per_level  = _normalize_sizes_per_level(_maybe_get(C, "model", {}), num_levels)
    ratios_per_level = _normalize_ratios_per_level(_maybe_get(C, "model", {}), num_levels)
    # 最終再防呆一次：確保長度完全一致
    if len(sizes_per_level) != num_levels:
        sizes_per_level = sizes_per_level[:num_levels] + tuple([sizes_per_level[-1]] * (num_levels - len(sizes_per_level)))
    if len(ratios_per_level) != num_levels:
        ratios_per_level = ratios_per_level[:num_levels] + tuple([ratios_per_level[-1]] * (num_levels - len(ratios_per_level)))

    # Anchors + RoIAlign
    anchor_gen = AnchorGenerator(sizes=sizes_per_level, aspect_ratios=ratios_per_level)
    feat_names: Tuple[str, ...] = ("0", "1", "2", "3", "pool") if use_p6 else ("0", "1", "2", "3")
    roi_pooler = MultiScaleRoIAlign(featmap_names=feat_names, output_size=int(_maybe_get(C, "model.roi.roi_pool_size", 14)), sampling_ratio=2)

    # RPN（穩健 baseline）
    RPN = {
        "batch_per_img": int(_maybe_get(C, "model.rpn.batch_size_per_image", 256)),
        "pos_frac":      float(_maybe_get(C, "model.rpn.positive_fraction", 0.5)),
        "fg_iou":        float(_maybe_get(C, "model.rpn.fg_iou_thresh", 0.7)),
        "bg_iou":        float(_maybe_get(C, "model.rpn.bg_iou_thresh", 0.3)),
        "pre_train":     int(_maybe_get(C, "model.rpn.pre_nms_top_n_train", 2000)),
        "post_train":    int(_maybe_get(C, "model.rpn.post_nms_top_n_train", 1000)),
        "pre_test":      int(_maybe_get(C, "model.rpn.pre_nms_top_n_test",  1000)),
        "post_test":     int(_maybe_get(C, "model.rpn.post_nms_top_n_test", 1000)),
        "nms":           float(_maybe_get(C, "model.rpn.nms_thresh", 0.7)),
    }

    # RoI / Head
    ROI = {
        "pool":            int(_maybe_get(C, "model.roi.roi_pool_size", 14)),
        "hidden":          int(_maybe_get(C, "model.roi.head_hidden_dim", 1024)),
        "num_fc":          int(_maybe_get(C, "model.roi.head_num_fc", 4)),
        "bbox_weights":    tuple(_maybe_get(C, "model.roi.bbox_reg_weights", [1.0, 1.0, 1.0, 1.0])),
        "nms":             float(_maybe_get(C, "model.roi.nms_thresh", 0.5)),
        "per_img":         int(_maybe_get(C, "model.roi.detections_per_img", 150)),
        "score_thresh":    float(_maybe_get(C, "model.roi.score_thresh", 0.0)),
        "box_batch_perimg":int(_maybe_get(C, "model.roi.box_batch_size_per_image", 512)),
        "box_pos_frac":    float(_maybe_get(C, "model.roi.box_positive_fraction", 0.25)),
        "fg_iou":          float(_maybe_get(C, "model.roi.fg_iou_thresh", 0.5)),
        "bg_iou":          float(_maybe_get(C, "model.roi.bg_iou_thresh", 0.5)),
    }

    # FasterRCNN
    model = FasterRCNN(
        backbone=backbone,
        num_classes=int(_maybe_get(C, "model.num_classes", 2)),
        rpn_anchor_generator=anchor_gen,
        box_roi_pool=roi_pooler,
        min_size=min_size, max_size=max_size,
        image_mean=image_mean, image_std=image_std,
        box_score_thresh=ROI["score_thresh"],
        box_nms_thresh=ROI["nms"],
        box_detections_per_img=ROI["per_img"],
    )

    # RPN 參數（相容不同 torchvision 版本）
    rpn = model.rpn
    rpn.batch_size_per_image = RPN["batch_per_img"]
    rpn.positive_fraction    = RPN["pos_frac"]
    rpn.foreground_iou_threshold = RPN["fg_iou"]
    rpn.background_iou_threshold = RPN["bg_iou"]
    rpn.nms_thresh               = RPN["nms"]
    if hasattr(rpn, "pre_nms_top_n_train"):  rpn.pre_nms_top_n_train  = RPN["pre_train"]
    if hasattr(rpn, "pre_nms_top_n_test"):   rpn.pre_nms_top_n_test   = RPN["pre_test"]
    if hasattr(rpn, "post_nms_top_n_train"): rpn.post_nms_top_n_train = RPN["post_train"]
    if hasattr(rpn, "post_nms_top_n_test"):  rpn.post_nms_top_n_test  = RPN["post_test"]

    # RoI head：4FC×1024 + bbox coder 權重 + 採樣策略
    out_ch = int(fpn_out)
    head = FourFCHead(out_ch, ROI["pool"], ROI["hidden"], ROI["num_fc"])
    model.roi_heads.box_head = head
    model.roi_heads.box_coder.weights = ROI["bbox_weights"]
    if hasattr(model.roi_heads, "batch_size_per_image"):
        model.roi_heads.batch_size_per_image = ROI["box_batch_perimg"]
    if hasattr(model.roi_heads, "positive_fraction"):
        model.roi_heads.positive_fraction = ROI["box_pos_frac"]
    if hasattr(model.roi_heads, "fg_iou_thresh"):
        model.roi_heads.fg_iou_thresh = ROI["fg_iou"]
    if hasattr(model.roi_heads, "bg_iou_thresh"):
        model.roi_heads.bg_iou_thresh = ROI["bg_iou"]

    # 最終 predictor
    model.roi_heads.box_predictor = FastRCNNPredictor(ROI["hidden"], int(_maybe_get(C, "model.num_classes", 2)))
    return model

# ---------- 工廠 ----------
def build_detector_from_cfg(cfg: Dict[str, Any]):
    m = cfg.get("model", {}) if isinstance(cfg, dict) else {}
    name = str(m.get("detector", "fasterrcnn_r101_fpn_v7")).lower()
    num_classes = int(m.get("num_classes", 2))
    min_size = _maybe_get(cfg, "model.min_size", None)
    max_size = _maybe_get(cfg, "model.max_size", None)
    image_mean = tuple(_maybe_get(cfg, "model.image_mean", (0.485, 0.456, 0.406)))
    image_std  = tuple(_maybe_get(cfg, "model.image_std",  (0.229, 0.224, 0.225)))

    return get_fasterrcnn_r101_fpn(
        num_classes=num_classes,
        min_size=min_size, max_size=max_size,
        image_mean=image_mean, image_std=image_std,
        cfg=cfg
    )
