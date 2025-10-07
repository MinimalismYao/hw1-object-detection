# src/modelv7.py
# ============================================
# v7 — Minimal & stable Faster R-CNN builder (R101 + FPN P2-P6, GN, Big RoI Head)
# 設計原則：簡潔、穩定可收斂、無多餘分支；參數優先讀 cfg，否則採用穩健預設。
# ============================================

from typing import Any, Dict, Iterable, Optional, Tuple

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

def _as_tuple_sizes(sizes_like):
    """
    轉成 AnchorGenerator 需要的格式：
      [8,16,32]           -> ((8,), (16,), (32,))
      [[8],[16],[32]]     -> ((8,), (16,), (32,))
      8                   -> ((8,),)
    """
    if sizes_like is None:
        return None
    if isinstance(sizes_like, (int, float)):
        return ((int(sizes_like),),)
    if isinstance(sizes_like, (list, tuple)) and all(isinstance(x, (int, float)) for x in sizes_like):
        return tuple((int(x),) for x in sizes_like)
    if isinstance(sizes_like, (list, tuple)):
        out = []
        for s in sizes_like:
            if isinstance(s, (list, tuple)):
                if len(s) == 0: continue
                out.append(tuple(int(v) for v in s))
            else:
                out.append((int(s),))
        return tuple(out)
    raise TypeError(f"Unsupported type for sizes: {type(sizes_like)}")

def _make_norm(norm_type: str):
    """回傳可被 ResNet/FPN 使用的 norm 層工廠。預設 GroupNorm32，從零訓練更穩。"""
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

# ---------- Box head：4×FC(1024) 大容量（小物件友善） ----------
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

# ---------- 核心建構：Faster R-CNN R101 + FPN(P2~P6) ----------
def get_fasterrcnn_r101_fpn(
    num_classes: int = 2,
    *,
    cfg: Optional[Dict[str, Any]] = None,
    min_size: Optional[int] = None,
    max_size: Optional[int] = None,
    image_mean: Iterable[float] = (0.485, 0.456, 0.406),
    image_std:  Iterable[float] = (0.229, 0.224, 0.225),
) -> FasterRCNN:
    """
    穩定可收斂預設（可被 v7.yaml 覆寫）：
      - Backbone: ResNet-101（無預訓練）+ FPN(P2~P5) + P6 (LastLevelMaxPool)
      - Norm    : GroupNorm32（從零訓練更穩）
      - Anchors : 預設小物件友善（可由 cfg.model.rpn_anchor_sizes / ratios 覆寫）
      - RoI     : RoIAlign=14、4×FC(1024) head、較保守 bbox_reg_weights
      - RPN     : 提案數合理、IoU 門檻穩健
    """
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

    # Anchors
    anc_sizes  = _maybe_get(C, "model.rpn_anchor_sizes",  [8, 12, 16, 24, 32, 48])
    anc_ratios = _maybe_get(C, "model.rpn_anchor_ratios", [0.5, 1.0, 2.0, 3.0])
    sizes_tuple = _as_tuple_sizes(anc_sizes)
    num_levels = 5 if use_p6 else 4  # P2~P5(+P6)
    if len(sizes_tuple) > num_levels: sizes_tuple = sizes_tuple[:num_levels]
    while len(sizes_tuple) < num_levels: sizes_tuple += (sizes_tuple[-1],)
    ratios_per_level = (tuple(float(r) for r in anc_ratios),) * num_levels

    # RPN
    RPN = {
        "batch_per_img": int(_maybe_get(C, "model.rpn.batch_size_per_image", 512)),
        "pos_frac":      float(_maybe_get(C, "model.rpn.positive_fraction", 0.5)),
        "fg_iou":        float(_maybe_get(C, "model.rpn.fg_iou_thresh", 0.6)),
        "bg_iou":        float(_maybe_get(C, "model.rpn.bg_iou_thresh", 0.2)),
        "pre_train":     int(_maybe_get(C, "model.rpn.pre_nms_top_n_train", 4000)),
        "post_train":    int(_maybe_get(C, "model.rpn.post_nms_top_n_train", 2000)),
        "pre_test":      int(_maybe_get(C, "model.rpn.pre_nms_top_n_test",  2000)),
        "post_test":     int(_maybe_get(C, "model.rpn.post_nms_top_n_test", 1000)),
        "nms":           float(_maybe_get(C, "model.rpn.nms_thresh", 0.7)),
    }

    # RoI / Head
    ROI = {
        "pool":            int(_maybe_get(C, "model.roi.roi_pool_size", 14)),
        "hidden":          int(_maybe_get(C, "model.roi.head_hidden_dim", 1024)),
        "num_fc":          int(_maybe_get(C, "model.roi.head_num_fc", 4)),
        "bbox_weights":    tuple(_maybe_get(C, "model.roi.bbox_reg_weights", [10.0, 10.0, 5.0, 5.0])),
        "nms":             float(_maybe_get(C, "model.roi.nms_thresh", 0.5)),
        "per_img":         int(_maybe_get(C, "model.roi.detections_per_img", 200)),
        "score_thresh":    float(_maybe_get(C, "model.roi.score_thresh", 0.0)),
        "box_batch_perimg":int(_maybe_get(C, "model.roi.box_batch_size_per_image", 1024)),
        "box_pos_frac":    float(_maybe_get(C, "model.roi.box_positive_fraction", 0.33)),
    }

    # ----- backbone + FPN (R101, GN, +P6) -----
    resnet = torchvision.models.resnet101(weights=None, norm_layer=norm_layer)
    return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
    in_channels_list = [256, 512, 1024, 2048]
    extra_blocks = torchvision.ops.feature_pyramid_network.LastLevelMaxPool() if use_p6 else None
    backbone = BackboneWithFPN(
        resnet, return_layers, in_channels_list, fpn_out, extra_blocks=extra_blocks
    )

    # Anchors + RoIAlign
    anchor_gen = AnchorGenerator(sizes=sizes_tuple, aspect_ratios=ratios_per_level)
    feat_names: Tuple[str, ...] = ("0", "1", "2", "3", "pool") if use_p6 else ("0", "1", "2", "3")
    roi_pooler = MultiScaleRoIAlign(featmap_names=feat_names, output_size=ROI["pool"], sampling_ratio=2)

    # FasterRCNN
    model = FasterRCNN(
        backbone=backbone,
        num_classes=int(num_classes),
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
    pre_tr, pre_te = RPN["pre_train"], RPN["pre_test"]
    post_tr, post_te = RPN["post_train"], RPN["post_test"]
    if callable(getattr(rpn, "pre_nms_top_n", None)) or callable(getattr(rpn, "post_nms_top_n", None)):
        rpn.pre_nms_top_n  = (lambda rpn=rpn, tr=pre_tr, te=pre_te:  tr if rpn.training else te)
        rpn.post_nms_top_n = (lambda rpn=rpn, tr=post_tr, te=post_te: tr if rpn.training else te)
    else:
        if hasattr(rpn, "pre_nms_top_n_train"):  rpn.pre_nms_top_n_train  = pre_tr
        if hasattr(rpn, "pre_nms_top_n_test"):   rpn.pre_nms_top_n_test   = pre_te
        if hasattr(rpn, "post_nms_top_n_train"): rpn.post_nms_top_n_train = post_tr
        if hasattr(rpn, "post_nms_top_n_test"):  rpn.post_nms_top_n_test  = post_te

    # RoI head：4FC×1024 + bbox coder 權重
    out_ch = int(fpn_out)
    head = FourFCHead(out_ch, ROI["pool"], ROI["hidden"], ROI["num_fc"])
    model.roi_heads.box_head = head
    model.roi_heads.box_coder.weights = ROI["bbox_weights"]
    # 訓練採樣配置（若 torchvision 版本支持）
    if hasattr(model.roi_heads, "batch_size_per_image"):
        model.roi_heads.batch_size_per_image = ROI["box_batch_perimg"]
    if hasattr(model.roi_heads, "positive_fraction"):
        model.roi_heads.positive_fraction = ROI["box_pos_frac"]

    # 最終 predictor
    model.roi_heads.box_predictor = FastRCNNPredictor(ROI["hidden"], int(num_classes))
    return model

# ---------- 工廠：由 YAML 的 model.detector 產生模型 ----------
def build_detector_from_cfg(cfg: Dict[str, Any]):
    """
    只支援 Faster R-CNN R101 強化版，名稱相容：
      - fasterrcnn_r101_fpn
      - fasterrcnn_r101_fpn_v7
      - （其他名稱一律回退至本實作）
    """
    m = cfg.get("model", {}) if isinstance(cfg, dict) else {}
    name = str(m.get("detector", "fasterrcnn_r101_fpn_v7")).lower()
    num_classes = int(m.get("num_classes", 2))
    min_size = _maybe_get(cfg, "model.min_size", None)
    max_size = _maybe_get(cfg, "model.max_size", None)
    image_mean = tuple(_maybe_get(cfg, "model.image_mean", (0.485, 0.456, 0.406)))
    image_std  = tuple(_maybe_get(cfg, "model.image_std",  (0.229, 0.224, 0.225)))

    # 統一走同一個建構器（名稱僅做相容）
    return get_fasterrcnn_r101_fpn(
        num_classes=num_classes,
        min_size=min_size, max_size=max_size,
        image_mean=image_mean, image_std=image_std,
        cfg=cfg
    )
