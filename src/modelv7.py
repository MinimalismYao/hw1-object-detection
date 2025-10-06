# src/modelv7.py
import torch
import torch.nn as nn
import torchvision
from typing import Any, Dict, Iterable, Optional, Sequence

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
    raise TypeError(f"Unsupported type for sizes: {type(sizes_like)}")


def _maybe_get(d: Dict[str, Any], path: str, default=None):
    cur = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


# ---------- Faster R-CNN R50-FPN（保留） ----------
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
    Faster R-CNN (ResNet50-FPN) using torchvision's resnet_fpn_backbone（相容性佳）

    合規保護：
    - 若使用預訓練骨幹（pretrained_backbone=True），將自動強制凍結骨幹（freeze_backbone=True），
      等同僅作為 feature extractor 使用。
    """
    # ---- 1) 從 cfg 覆蓋參數（若存在）----
    acfg = _maybe_get(cfg or {}, "augment", {}) if cfg else {}
    min_size = _maybe_get(cfg or {}, "model.min_size", min_size)
    max_size = _maybe_get(cfg or {}, "model.max_size", max_size)
    if min_size is None and acfg:
        min_size = int(acfg.get("max_side", 1024))
    if max_size is None and acfg:
        max_size = int(acfg.get("max_side", 1024))
    if min_size is None: min_size = 1024
    if max_size is None: max_size = 1024

    image_mean = list(_maybe_get(cfg or {}, "model.image_mean", image_mean))
    image_std  = list(_maybe_get(cfg or {}, "model.image_std",  image_std))

    rpn_anchor_sizes = _maybe_get(cfg or {}, "model.rpn_anchor_sizes", rpn_anchor_sizes)
    sizes_tuple = _as_tuple_sizes(rpn_anchor_sizes) if rpn_anchor_sizes is not None else ((32,), (64,), (128,), (256,), (512,))
    rpn_anchor_ratios = _maybe_get(cfg or {}, "model.rpn_anchor_ratios", list(rpn_anchor_ratios))

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

    # ---- 合規防呆：用到預訓練骨幹就強制凍結 ----
    if pretrained_backbone and not freeze_backbone:
        freeze_backbone = True  # 自動修正為合規設定

    # ---- 2) Backbone with FPN ----
    if pretrained_backbone:
        try:
            weights = torchvision.models.ResNet50_Weights.DEFAULT
        except AttributeError:
            weights = "IMAGENET1K_V1"
    else:
        weights = None

    trainable_layers = 0 if freeze_backbone else 5
    backbone = resnet_fpn_backbone(
        backbone_name="resnet50",
        weights=weights,
        trainable_layers=trainable_layers,
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

    if freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad_(False)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


# ---------- 其他可選路線（均不載入預訓練） ----------
def get_fasterrcnn_mbv3_fpn(num_classes=2, *, min_size=None, max_size=None,
                             image_mean=(0.0,0.0,0.0), image_std=(1.0,1.0,1.0),
                             cfg: Optional[Dict[str, Any]] = None):
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
        weights=None, weights_backbone=None,
        min_size=_maybe_get(cfg or {}, "model.min_size", min_size),
        max_size=_maybe_get(cfg or {}, "model.max_size", max_size),
        image_mean=list(_maybe_get(cfg or {}, "model.image_mean", image_mean)),
        image_std=list(_maybe_get(cfg or {}, "model.image_std", image_std)),
        num_classes=num_classes
    )
    return model


def get_fasterrcnn_r101_fpn(num_classes=2, *, min_size=None, max_size=None,
                             image_mean=(0.0,0.0,0.0), image_std=(1.0,1.0,1.0),
                             cfg: Optional[Dict[str, Any]] = None):
    min_size = _maybe_get(cfg or {}, "model.min_size", min_size)
    max_size = _maybe_get(cfg or {}, "model.max_size", max_size)
    image_mean = list(_maybe_get(cfg or {}, "model.image_mean", image_mean))
    image_std  = list(_maybe_get(cfg or {}, "model.image_std",  image_std))

    backbone = resnet_fpn_backbone('resnet101', weights=None, trainable_layers=5, norm_layer=nn.BatchNorm2d)
    rpn_anchor_sizes = _maybe_get(cfg or {}, "model.rpn_anchor_sizes", None)
    sizes_tuple = _as_tuple_sizes(rpn_anchor_sizes) if rpn_anchor_sizes is not None else ((32,), (64,), (128,), (256,), (512,))
    rpn_anchor_ratios = _maybe_get(cfg or {}, "model.rpn_anchor_ratios", [0.5, 1.0, 2.0])

    anchor_generator = AnchorGenerator(
        sizes=sizes_tuple,
        aspect_ratios=(tuple(float(r) for r in rpn_anchor_ratios),) * len(sizes_tuple),
    )
    roi_pooler = MultiScaleRoIAlign(featmap_names=["0","1","2","3"], output_size=7, sampling_ratio=2)

    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        min_size=int(min_size) if min_size else 1024,
        max_size=int(max_size) if max_size else 1024,
        image_mean=image_mean,
        image_std=image_std,
        box_score_thresh=float(_maybe_get(cfg or {}, "model.box_score_thresh", 0.05)),
        box_nms_thresh=float(_maybe_get(cfg or {}, "model.box_nms_thresh", 0.5)),
        box_detections_per_img=int(_maybe_get(cfg or {}, "model.box_detections_per_img", 100)),
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def get_retinanet_r50_fpn(num_classes=2, *, min_size=None, max_size=None,
                          image_mean=(0.0,0.0,0.0), image_std=(1.0,1.0,1.0),
                          cfg: Optional[Dict[str, Any]] = None):
    """
    RetinaNet-R50-FPN
    - 可自訂 anchors：model.retinanet_anchor_sizes（例：[16,32,64,128,256]）
                     model.retinanet_anchor_aspect_ratios（例：[1.0,2.0,3.0]）
    - 注意 torchvision 的 RetinaNet num_classes = 「實際類別數」（不含背景）。
      若 YAML 仍沿用「單類別+背景=2」的習慣，這裡會自動轉成 1。
    """
    # 1) 解析尺寸與正規化
    min_size = _maybe_get(cfg or {}, "model.min_size", min_size)
    max_size = _maybe_get(cfg or {}, "model.max_size", max_size)
    image_mean = list(_maybe_get(cfg or {}, "model.image_mean", image_mean))
    image_std  = list(_maybe_get(cfg or {}, "model.image_std",  image_std))

    # 2) 解析 anchors（優先用 retinanet_*，退回 rpn_* 或預設金字塔）
    anc_sizes = _maybe_get(cfg or {}, "model.retinanet_anchor_sizes", None)
    if anc_sizes is None:
        anc_sizes = _maybe_get(cfg or {}, "model.rpn_anchor_sizes", [16, 32, 64, 128, 256])
    sizes_tuple = _as_tuple_sizes(anc_sizes)
    while len(sizes_tuple) < 5:  # 補到 5 層
        sizes_tuple += (sizes_tuple[-1],)
    sizes_tuple = sizes_tuple[:5]

    anc_ratios = _maybe_get(cfg or {}, "model.retinanet_anchor_aspect_ratios",
                            _maybe_get(cfg or {}, "model.rpn_anchor_ratios", [1.0, 2.0, 3.0]))
    ratios_per_level = (tuple(float(r) for r in anc_ratios),) * len(sizes_tuple)
    anchor_generator = AnchorGenerator(sizes=sizes_tuple, aspect_ratios=ratios_per_level)

    # 3) num_classes 轉換（容忍「含背景」寫法）
    retina_num_classes = max(1, int(num_classes) - 1) if int(num_classes) > 1 else int(num_classes)

    # 4) 建模（不載入任何預訓練）
    model = torchvision.models.detection.retinanet_resnet50_fpn(
        weights=None, weights_backbone=None,
        num_classes=retina_num_classes,
        anchor_generator=anchor_generator,
        min_size=min_size, max_size=max_size,
        image_mean=image_mean, image_std=image_std,
    )
    return model


def get_ssdlite_mbv3(num_classes=2, *,
                     image_mean=(0.0,0.0,0.0), image_std=(1.0,1.0,1.0),
                     cfg: Optional[Dict[str, Any]] = None):
    model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(
        weights=None, weights_backbone=None,
        num_classes=max(1, int(num_classes) - 1) if int(num_classes) > 1 else int(num_classes),
        image_mean=list(_maybe_get(cfg or {}, "model.image_mean", image_mean)),
        image_std=list(_maybe_get(cfg or {}, "model.image_std", image_std)),
    )
    return model


# ---------- 工廠：由 YAML 的 model.detector 產生模型 ----------
_DET_BUILDERS = {
    "fasterrcnn_r50_fpn": get_fasterrcnn_r50_fpn,
    "fasterrcnn_mbv3_fpn": get_fasterrcnn_mbv3_fpn,
    "fasterrcnn_r101_fpn": get_fasterrcnn_r101_fpn,
    "retinanet_r50_fpn": get_retinanet_r50_fpn,
    "ssdlite_mbv3": get_ssdlite_mbv3,
}

def build_detector_from_cfg(cfg: Dict[str, Any]):
    mcfg = cfg.get("model", {})
    name = str(mcfg.get("detector", "fasterrcnn_r50_fpn")).lower()

    num_classes = int(mcfg.get("num_classes", 2))
    min_size = _maybe_get(cfg or {}, "model.min_size", None)
    max_size = _maybe_get(cfg or {}, "model.max_size", None)
    image_mean = tuple(_maybe_get(cfg or {}, "model.image_mean", (0.0,0.0,0.0)))
    image_std  = tuple(_maybe_get(cfg or {}, "model.image_std",  (1.0,1.0,1.0)))

    builder = _DET_BUILDERS.get(name, get_fasterrcnn_r50_fpn)

    # RetinaNet/SSDlite 允許 YAML 使用「含背景」習慣寫法 → 內部轉換
    if builder in (get_retinanet_r50_fpn, get_ssdlite_mbv3):
        return builder(
            num_classes=num_classes,  # 內部會自動轉換
            min_size=min_size, max_size=max_size,
            image_mean=image_mean, image_std=image_std,
            cfg=cfg
        )

    # 其它模型路線依各自 API 建構（盡量沿用 mean/std/size）
    return builder(
        num_classes=num_classes,
        min_size=min_size, max_size=max_size,
        image_mean=image_mean, image_std=image_std,
        cfg=cfg
    )
