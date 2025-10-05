# src/modelv5.py
import torch
import torch.nn as nn
import torchvision
from typing import Any, Dict, Iterable, Optional

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


# ---------- utils ----------
def _as_tuple_sizes(sizes_like):
    """
    把多種寫法轉成 AnchorGenerator 需要的格式：
      [[8], [16], [32], [64], [128]] → ((8,), (16,), (32,), (64,), (128,))
      [8, 16, 32, 64, 128]           → ((8,), (16,), (32,), (64,), (128,))
      8                               → ((8,),)
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
                if len(s) == 0:
                    continue
                out.append(tuple(int(v) for v in s))
            else:
                out.append((int(s),))
        return tuple(out)
    raise TypeError(f"Unsupported type for rpn_anchor_sizes: {type(sizes_like)}")


def _maybe_get(d: Dict[str, Any], path: str, default=None):
    cur = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


# ---------- builder ----------
def get_fasterrcnn_r50_fpn(
    num_classes: int = 2,
    freeze_backbone: bool = False,
    *,
    # Transform / inference
    min_size: Optional[int] = None,
    max_size: Optional[int] = None,
    image_mean: Iterable[float] = (0.0, 0.0, 0.0),
    image_std: Iterable[float] = (1.0, 1.0, 1.0),
    # Anchors / RPN / ROI
    rpn_anchor_sizes: Optional[Iterable] = None,
    rpn_anchor_ratios: Iterable[float] = (0.5, 1.0, 2.0),
    rpn_pre_nms_top_n_train: int = 2000,
    rpn_pre_nms_top_n_test: int = 1000,
    rpn_post_nms_top_n_train: int = 1000,
    rpn_post_nms_top_n_test: int = 1000,
    rpn_nms_thresh: float = 0.7,
    box_score_thresh: float = 0.05,
    box_nms_thresh: float = 0.5,
    box_detections_per_img: int = 100,
    pretrained_backbone: bool = False,
    cfg: Optional[Dict[str, Any]] = None,
) -> FasterRCNN:
    """
    Faster R-CNN (ResNet50-FPN) using torchvision's resnet_fpn_backbone（相容性最佳）
    """
    # ---- 1) 從 cfg 覆蓋參數（若存在）----
    acfg = _maybe_get(cfg or {}, "augment", {}) if cfg else {}
    # Transform 尺寸：若沒指定，沿用 augment.max_side；否則 fallback 1024
    min_size = _maybe_get(cfg or {}, "model.min_size", min_size)
    max_size = _maybe_get(cfg or {}, "model.max_size", max_size)
    if min_size is None and acfg:
        min_size = int(acfg.get("max_side", 1024))
    if max_size is None and acfg:
        max_size = int(acfg.get("max_side", 1024))
    if min_size is None: min_size = 1024
    if max_size is None: max_size = 1024

    # Anchors
    rpn_anchor_sizes = _maybe_get(cfg or {}, "model.rpn_anchor_sizes", rpn_anchor_sizes)
    sizes_tuple = _as_tuple_sizes(rpn_anchor_sizes) if rpn_anchor_sizes is not None else ((32,), (64,), (128,), (256,), (512,))
    rpn_anchor_ratios = _maybe_get(cfg or {}, "model.rpn_anchor_ratios", list(rpn_anchor_ratios))

    # RPN / ROI / Inference
    rpn_pre_nms_top_n_train = int(_maybe_get(cfg or {}, "model.rpn_pre_nms_top_n_train", rpn_pre_nms_top_n_train))
    rpn_post_nms_top_n_train = int(_maybe_get(cfg or {}, "model.rpn_post_nms_top_n_train", rpn_post_nms_top_n_train))
    rpn_pre_nms_top_n_test  = int(_maybe_get(cfg or {}, "model.rpn_pre_nms_top_n_test",  rpn_pre_nms_top_n_test))
    rpn_post_nms_top_n_test = int(_maybe_get(cfg or {}, "model.rpn_post_nms_top_n_test", rpn_post_nms_top_n_test))
    rpn_nms_thresh = float(_maybe_get(cfg or {}, "model.rpn_nms_thresh", rpn_nms_thresh))

    box_score_thresh = float(_maybe_get(cfg or {}, "model.box_score_thresh", box_score_thresh))
    box_nms_thresh   = float(_maybe_get(cfg or {}, "model.box_nms_thresh",   box_nms_thresh))
    box_detections_per_img = int(_maybe_get(cfg or {}, "model.box_detections_per_img", box_detections_per_img))

    pretrained_backbone = bool(_maybe_get(cfg or {}, "model.pretrained_backbone", pretrained_backbone))
    freeze_backbone     = bool(_maybe_get(cfg or {}, "model.freeze_backbone", freeze_backbone))

    # ---- 2) Backbone with FPN（官方工具）----
    #   weights=None：from scratch；若要預訓練 backbone，會在新版用 ResNet50_Weights.DEFAULT
    if pretrained_backbone:
        try:
            weights = torchvision.models.ResNet50_Weights.DEFAULT
        except AttributeError:
            weights = "IMAGENET1K_V1"
    else:
        weights = None

    backbone = resnet_fpn_backbone(
        backbone_name="resnet50",
        weights=weights,
        trainable_layers=5,           # 先全開，等會根據 freeze_backbone 再凍結
        norm_layer=nn.BatchNorm2d,
    )

    # ---- 3) Anchors / RoIAlign ----
    anchor_generator = AnchorGenerator(
        sizes=sizes_tuple,
        aspect_ratios=(tuple(float(r) for r in rpn_anchor_ratios),) * len(sizes_tuple),
    )
    roi_pooler = MultiScaleRoIAlign(
        featmap_names=["0", "1", "2", "3"],
        output_size=7,
        sampling_ratio=2,
    )

    # ---- 4) FasterRCNN 主體 ----
    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        # Transform
        min_size=int(min_size),
        max_size=int(max_size),
        image_mean=list(image_mean),
        image_std=list(image_std),
        # RPN
        rpn_pre_nms_top_n_train=int(rpn_pre_nms_top_n_train),
        rpn_pre_nms_top_n_test=int(rpn_pre_nms_top_n_test),
        rpn_post_nms_top_n_train=int(rpn_post_nms_top_n_train),
        rpn_post_nms_top_n_test=int(rpn_post_nms_top_n_test),
        rpn_nms_thresh=float(rpn_nms_thresh),
        # ROI / Inference
        box_score_thresh=float(box_score_thresh),
        box_nms_thresh=float(box_nms_thresh),
        box_detections_per_img=int(box_detections_per_img),
    )

    # ---- 5) 凍結 backbone（如需）----
    if freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad_(False)

    # ---- 6) 保險：替換分類頭以確保 num_classes 正確 ----
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model
