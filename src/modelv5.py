# src/modelv5.py
import torch
import torch.nn as nn
import torchvision
from typing import Any, Dict, Iterable, Optional, Tuple

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.detection.backbone_utils import BackboneWithFPN


# -------------------------- utils --------------------------
def _as_tuple_sizes(sizes_like):
    """
    將多種寫法轉成 AnchorGenerator 需要的格式：
      A) [[8], [16], [32], [64], [128]]
      B) [8, 16, 32, 64, 128]
      C) 8
    皆會轉成：((8,), (16,), (32,), (64,), (128,))
    """
    if sizes_like is None:
        return None

    # C) 單一數值
    if isinstance(sizes_like, (int, float)):
        return ((int(sizes_like),),)

    # B) 扁平 list/tuple
    if isinstance(sizes_like, (list, tuple)) and all(isinstance(x, (int, float)) for x in sizes_like):
        return tuple((int(x),) for x in sizes_like)

    # A) 巢狀 list/tuple 或混合
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
    """安全取得巢狀 dict 值：_maybe_get(cfg, 'model.rpn_nms_thresh', 0.7)"""
    cur = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


# --------------------- backbone (ResNet50-FPN) ---------------------
def _build_resnet50_fpn(
    pretrained_backbone: bool = False,
    norm_layer=nn.BatchNorm2d,
) -> BackboneWithFPN:
    """
    建 ResNet50 + FPN：
    - 預設 from scratch（pretrained_backbone=False）
    - 使用可學習 BN（from scratch 收斂較穩）
    - 輸出 256 維通道的 FPN，並加上 P6（LastLevelMaxPool）
    """
    if pretrained_backbone:
        try:
            weights = torchvision.models.ResNet50_Weights.DEFAULT
        except AttributeError:
            # 舊版 torchvision 相容
            weights = "IMAGENET1K_V1"
        resnet = torchvision.models.resnet50(weights=weights, norm_layer=norm_layer)
    else:
        resnet = torchvision.models.resnet50(weights=None, norm_layer=norm_layer)

    # 取出 C2~C5
    return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
    in_channels_list = [256, 512, 1024, 2048]
    body = IntermediateLayerGetter(resnet, return_layers=return_layers)

    backbone = BackboneWithFPN(
        body=body,
        return_layers=return_layers,
        in_channels_list=in_channels_list,
        out_channels=256,
        extra_blocks=LastLevelMaxPool(),  # 產生 P6（鍵名通常為 "pool"）
    )
    return backbone  # .out_channels = 256


# --------------------- model builder ---------------------
def get_fasterrcnn_r50_fpn(
    num_classes: int = 2,
    freeze_backbone: bool = False,
    *,
    # 可直接由參數控制（若傳了 cfg，會以 cfg 為優先，參數為後備）
    min_size: Optional[int] = None,
    max_size: Optional[int] = None,
    image_mean: Iterable[float] = (0.0, 0.0, 0.0),
    image_std: Iterable[float] = (1.0, 1.0, 1.0),
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
    # 允許把完整 cfg 傳進來（優先使用）
    cfg: Optional[Dict[str, Any]] = None,
) -> FasterRCNN:
    """
    Faster R-CNN (ResNet50-FPN)
    - 預設 from scratch；可切為 pretrained_backbone=True
    - 支援從 cfg 或函式參數設定 Transform / RPN / ROI / anchors
    """

    # ---- 1) 從 cfg 取值（若存在） ----
    mcfg = _maybe_get(cfg or {}, "model", {}) if cfg else {}
    acfg = _maybe_get(cfg or {}, "augment", {}) if cfg else {}
    icfg = _maybe_get(cfg or {}, "infer", {}) if cfg else {}
    ecfg = _maybe_get(cfg or {}, "eval", {}) if cfg else {}

    # Transform 尺寸：若沒指定，嘗試沿用 augment.max_side；否則 fallback 1024
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

    # RPN / ROI
    rpn_pre_nms_top_n_train = int(_maybe_get(cfg or {}, "model.rpn_pre_nms_top_n_train", rpn_pre_nms_top_n_train))
    rpn_post_nms_top_n_train = int(_maybe_get(cfg or {}, "model.rpn_post_nms_top_n_train", rpn_post_nms_top_n_train))
    rpn_pre_nms_top_n_test  = int(_maybe_get(cfg or {}, "model.rpn_pre_nms_top_n_test",  rpn_pre_nms_top_n_test))
    rpn_post_nms_top_n_test = int(_maybe_get(cfg or {}, "model.rpn_post_nms_top_n_test", rpn_post_nms_top_n_test))
    rpn_nms_thresh = float(_maybe_get(cfg or {}, "model.rpn_nms_thresh", rpn_nms_thresh))

    # Inference（外部 eval/infer 仍可再做閾值/NMS，這裡是 RCNN 內部的）
    box_score_thresh = float(_maybe_get(cfg or {}, "model.box_score_thresh", _maybe_get(cfg or {}, "infer.score_thr", box_score_thresh)))
    box_nms_thresh   = float(_maybe_get(cfg or {}, "model.box_nms_thresh",   _maybe_get(cfg or {}, "infer.nms_iou",   box_nms_thresh)))
    box_detections_per_img = int(_maybe_get(cfg or {}, "model.box_detections_per_img", _maybe_get(cfg or {}, "eval.max_det", box_detections_per_img)))

    pretrained_backbone = bool(_maybe_get(cfg or {}, "model.pretrained_backbone", pretrained_backbone))
    freeze_backbone     = bool(_maybe_get(cfg or {}, "model.freeze_backbone", freeze_backbone))

    # ---- 2) Backbone + FPN ----
    backbone = _build_resnet50_fpn(pretrained_backbone=pretrained_backbone, norm_layer=nn.BatchNorm2d)

    # ---- 3) Anchors / RoIAlign ----
    # sizes_tuple 例如：((8,), (16,), (32,), (64,), (128,))
    anchor_generator = AnchorGenerator(
        sizes=sizes_tuple,
        aspect_ratios=(tuple(float(r) for r in rpn_anchor_ratios),) * len(sizes_tuple),
    )

    # ROI 常用設定：用 P2~P5；P6 交由 RPN 使用即可
    roi_pooler = MultiScaleRoIAlign(
        featmap_names=["0", "1", "2", "3"],
        output_size=7,
        sampling_ratio=2,
    )

    # ---- 4) FasterRCNN 主體（顯式指定 Transform 與 RPN/ROI 參數）----
    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_classes,  # 背景+1類 = 2
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        # GeneralizedRCNNTransform（與你的前處理協同）
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
