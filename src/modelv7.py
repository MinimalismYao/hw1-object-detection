# src/modelv7.py
# ============================================
# v7 — Detector factory + strong Faster R-CNN (R101+FPN P2-P6、大容量 RoI Head)
#   - 保留：FRCNN-R50-FPN、RetinaNet-R50-FPN、SSDLite-MBV3
#   - 強化：新增 get_fasterrcnn_r101_fpn_v7 （可用 GroupNorm、P6、4FC×1024 box head、roi_pool=14）
#   - 參數來源：優先讀 cfg（v7.yaml），否則使用本檔頭的穩定預設
# ============================================

from typing import Any, Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torchvision
from torchvision.ops import MultiScaleRoIAlign, misc as misc_ops
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import BackboneWithFPN, resnet_fpn_backbone

# ---------- utils ----------
def _as_tuple_sizes(sizes_like):
    """
    轉成 AnchorGenerator 需要的格式：
      [[8], [16], [32], ...]  → ((8,), (16,), (32,), ...)
      [8, 16, 32, ...]        → ((8,), (16,), (32,), ...)
      8                        → ((8,),)
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

def _make_norm(norm_type: str):
    """回傳可被 ResNet/FPN 使用的 norm 層工廠。"""
    norm_type = (norm_type or "bn").lower()
    if norm_type == "bn":
        return nn.BatchNorm2d
    if norm_type == "frozenbn":
        return misc_ops.FrozenBatchNorm2d
    if norm_type in ("groupnorm", "groupnorm32", "gn", "gn32"):
        def _gn(ch):
            g = 32 if ch % 32 == 0 else (16 if ch % 16 == 0 else 8)
            return nn.GroupNorm(g, ch)
        return _gn
    return nn.BatchNorm2d

# ---------- 自訂 RoI Box Head（大容量 4FC×1024，可調） ----------
class FourFCHead(nn.Module):
    def __init__(self, in_channels: int, resolution: int = 14, hidden: int = 1024, num_fc: int = 4):
        super().__init__()
        input_size = in_channels * resolution * resolution
        layers = []
        for i in range(num_fc):
            layers += [nn.Linear(input_size if i == 0 else hidden, hidden), nn.ReLU(inplace=True)]
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        x = torch.flatten(x, 1)
        return self.fc(x)

# ---------- Faster R-CNN · 強化版（R101 + FPN P2-P6） ----------
def get_fasterrcnn_r101_fpn_v7(
    num_classes: int = 2,
    *,
    cfg: Optional[Dict[str, Any]] = None,
    # 通用 Transform / Normalize
    min_size: Optional[int] = None,
    max_size: Optional[int] = None,
    image_mean: Iterable[float] = (0.0, 0.0, 0.0),
    image_std: Iterable[float] = (1.0, 1.0, 1.0),
) -> FasterRCNN:
    """
    v7 強化路線：
      - Backbone: ResNet-101 + FPN(P2~P5) + 可選 P6（LastLevelMaxPool）
      - Anchors : min=8（8,16,32,64,128,256），多比例（含 3.0 長條）
      - RoI head: 4FC×1024，RoIAlign=14×14（可調）
      - Norm    : 可選 BN/FrozenBN/GroupNorm32（建議 GN，從零訓練更穩）
    """
    # ---- 基本尺寸與正規化 ----
    min_size = _maybe_get(cfg or {}, "model.min_size", min_size) or 1280
    max_size = _maybe_get(cfg or {}, "model.max_size", max_size) or 1280
    image_mean = list(_maybe_get(cfg or {}, "model.image_mean", image_mean))
    image_std  = list(_maybe_get(cfg or {}, "model.image_std",  image_std))

    # ---- FPN / Norm / P6 ----
    norm_type = str(_maybe_get(cfg or {}, "model.norm", "groupnorm32"))
    fpn_out_channels = int(_maybe_get(cfg or {}, "model.fpn_out_channels", 256))
    use_p6 = bool(_maybe_get(cfg or {}, "model.use_p6", True))

    # ---- Anchors（優先讀 rpn_anchor_sizes/ratios；否則使用小物件友善預設）----
    anc_sizes_cfg  = _maybe_get(cfg or {}, "model.rpn_anchor_sizes",  [8, 16, 32, 64, 128, 256])
    anc_ratios_cfg = _maybe_get(cfg or {}, "model.rpn_anchor_ratios", [0.5, 1.0, 2.0, 3.0])

    sizes_tuple = _as_tuple_sizes(anc_sizes_cfg)
    num_levels = 5 if use_p6 else 4  # FPN 輸出層數：P2~P5(+P6)
    # 裁/補到與 FPN 層數一致
    if len(sizes_tuple) > num_levels:
        sizes_tuple = sizes_tuple[:num_levels]
    while len(sizes_tuple) < num_levels:
        sizes_tuple += (sizes_tuple[-1],)
    ratios_per_level = (tuple(float(r) for r in anc_ratios_cfg),) * num_levels

    # ---- RPN 配置（多 proposals，密集場景更保險）----
    RPN = {
        "batch_size_per_image": int(_maybe_get(cfg or {}, "model.rpn.batch_size_per_image", 1024)),
        "positive_fraction":    float(_maybe_get(cfg or {}, "model.rpn.positive_fraction", 0.5)),
        "fg_iou_thresh":        float(_maybe_get(cfg or {}, "model.rpn.fg_iou_thresh", 0.6)),
        "bg_iou_thresh":        float(_maybe_get(cfg or {}, "model.rpn.bg_iou_thresh", 0.2)),
        "pre_nms_top_n_train":  int(_maybe_get(cfg or {}, "model.rpn.pre_nms_top_n_train", 6000)),
        "post_nms_top_n_train": int(_maybe_get(cfg or {}, "model.rpn.post_nms_top_n_train", 3000)),
        "pre_nms_top_n_test":   int(_maybe_get(cfg or {}, "model.rpn.pre_nms_top_n_test",  3000)),
        "post_nms_top_n_test":  int(_maybe_get(cfg or {}, "model.rpn.post_nms_top_n_test", 1500)),
        "nms_thresh":           float(_maybe_get(cfg or {}, "model.rpn.nms_thresh", 0.7)),
    }

    # ---- RoI / Box head（大容量）----
    ROI = {
        "roi_pool_size":      int(_maybe_get(cfg or {}, "model.roi.roi_pool_size", 14)),
        "head_hidden_dim":    int(_maybe_get(cfg or {}, "model.roi.head_hidden_dim", 1024)),
        "head_num_fc":        int(_maybe_get(cfg or {}, "model.roi.head_num_fc", 4)),
        "bbox_reg_weights":   tuple(_maybe_get(cfg or {}, "model.roi.bbox_reg_weights", [10.0, 10.0, 5.0, 5.0])),
        "nms_thresh":         float(_maybe_get(cfg or {}, "model.roi.nms_thresh", 0.5)),
        "detections_per_img": int(_maybe_get(cfg or {}, "model.roi.detections_per_img", 300)),
        "score_thresh":       float(_maybe_get(cfg or {}, "model.roi.score_thresh", 0.0)),
    }

    # ---- 建 backbone+FPN（R101、GN、P6）----
    norm_layer = _make_norm(norm_type)
    resnet = torchvision.models.resnet101(weights=None, norm_layer=norm_layer)

    return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
    in_channels_list = [256, 512, 1024, 2048]
    extra_blocks = torchvision.ops.feature_pyramid_network.LastLevelMaxPool() if use_p6 else None
    backbone = BackboneWithFPN(
        resnet,
        return_layers,
        in_channels_list,
        fpn_out_channels,       # out_channels（必填）
        extra_blocks=extra_blocks
    )

    # ---- Anchors / RoIAlign ----
    anchor_generator = AnchorGenerator(sizes=sizes_tuple, aspect_ratios=ratios_per_level)
    feat_names: Tuple[str, ...] = ("0", "1", "2", "3", "pool") if use_p6 else ("0", "1", "2", "3")
    roi_pooler = MultiScaleRoIAlign(featmap_names=feat_names, output_size=ROI["roi_pool_size"], sampling_ratio=2)

    # ---- FasterRCNN 主體 ----
    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_classes,  # 直接提供，後面仍會自訂 head/predictor
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        min_size=int(min_size),
        max_size=int(max_size),
        image_mean=list(image_mean),
        image_std=list(image_std),
        box_score_thresh=ROI["score_thresh"],
        box_nms_thresh=ROI["nms_thresh"],
        box_detections_per_img=ROI["detections_per_img"],
    )

    # ===== RPN 參數（版本相容：callable 或 *_train/_test）=====
    rpn = model.rpn
    # 基本門檻/比例/IoU
    rpn.batch_size_per_image = RPN["batch_size_per_image"]
    rpn.positive_fraction    = RPN["positive_fraction"]
    rpn.foreground_iou_threshold = RPN["fg_iou_thresh"]
    rpn.background_iou_threshold = RPN["bg_iou_thresh"]
    rpn.nms_thresh               = RPN["nms_thresh"]

    pre_train, pre_test   = RPN["pre_nms_top_n_train"],  RPN["pre_nms_top_n_test"]
    post_train, post_test = RPN["post_nms_top_n_train"], RPN["post_nms_top_n_test"]

    if callable(getattr(rpn, "pre_nms_top_n", None)) or callable(getattr(rpn, "post_nms_top_n", None)):
        # 你的 torchvision 會呼叫 rpn.pre_nms_top_n() / post_nms_top_n()
        rpn.pre_nms_top_n  = (lambda rpn=rpn, tr=pre_train, te=pre_test:  tr if rpn.training else te)
        rpn.post_nms_top_n = (lambda rpn=rpn, tr=post_train, te=post_test: tr if rpn.training else te)
    elif hasattr(rpn, "pre_nms_top_n_train"):
        # 舊版 API：獨立屬性
        rpn.pre_nms_top_n_train  = pre_train
        rpn.pre_nms_top_n_test   = pre_test
        rpn.post_nms_top_n_train = post_train
        rpn.post_nms_top_n_test  = post_test
    # 千萬不要把 pre_nms_top_n/post_nms_top_n 設成 dict（否則會被當成 callable 再次崩潰）

    # ---- RoI box head（替換成 4FC×1024）----
    out_channels = fpn_out_channels
    rep_dim = ROI["head_hidden_dim"]
    model.roi_heads.box_head = FourFCHead(out_channels, ROI["roi_pool_size"], rep_dim, ROI["head_num_fc"])
    model.roi_heads.box_coder.weights = ROI["bbox_reg_weights"]

    # ---- 最終分類器 ----
    in_features = rep_dim
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

# ---------- Faster R-CNN · R50-FPN（保留，較輕量） ----------
def get_fasterrcnn_r50_fpn(
    num_classes: int = 2,
    freeze_backbone: bool = False,
    *,
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
    cfg: Optional[Dict[str, Any]] = None,
) -> FasterRCNN:
    # 尺寸
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

    # anchors
    rpn_anchor_sizes = _maybe_get(cfg or {}, "model.rpn_anchor_sizes", rpn_anchor_sizes)
    sizes_tuple = _as_tuple_sizes(rpn_anchor_sizes) if rpn_anchor_sizes is not None else ((32,), (64,), (128,), (256,), (512,))
    rpn_anchor_ratios = _maybe_get(cfg or {}, "model.rpn_anchor_ratios", list(rpn_anchor_ratios))

    # torchvision R50-FPN
    weights = torchvision.models.ResNet50_Weights.DEFAULT if (pretrained_backbone) else None
    trainable_layers = 0 if (pretrained_backbone or freeze_backbone) else 5
    backbone = resnet_fpn_backbone(
        backbone_name="resnet50",
        weights=weights,
        trainable_layers=trainable_layers,
        norm_layer=nn.BatchNorm2d,
    )

    anchor_generator = AnchorGenerator(
        sizes=sizes_tuple,
        aspect_ratios=(tuple(float(r) for r in rpn_anchor_ratios),) * len(sizes_tuple),
    )
    roi_pooler = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2)

    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        min_size=int(min_size),
        max_size=int(max_size),
        image_mean=list(image_mean),
        image_std=list(image_std),
        rpn_pre_nms_top_n_train=int(_maybe_get(cfg or {}, "model.rpn_pre_nms_top_n_train", rpn_pre_nms_top_n_train)),
        rpn_pre_nms_top_n_test=int(_maybe_get(cfg or {}, "model.rpn_pre_nms_top_n_test", rpn_pre_nms_top_n_test)),
        rpn_post_nms_top_n_train=int(_maybe_get(cfg or {}, "model.rpn_post_nms_top_n_train", rpn_post_nms_top_n_train)),
        rpn_post_nms_top_n_test=int(_maybe_get(cfg or {}, "model.rpn_post_nms_top_n_test", rpn_post_nms_top_n_test)),
        rpn_nms_thresh=float(_maybe_get(cfg or {}, "model.rpn_nms_thresh", rpn_nms_thresh)),
        box_score_thresh=float(_maybe_get(cfg or {}, "model.box_score_thresh", box_score_thresh)),
        box_nms_thresh=float(_maybe_get(cfg or {}, "model.box_nms_thresh", box_nms_thresh)),
        box_detections_per_img=int(_maybe_get(cfg or {}, "model.box_detections_per_img", box_detections_per_img)),
    )

    if freeze_backbone or pretrained_backbone:
        for p in model.backbone.parameters():
            p.requires_grad_(False)

    in_feat = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, num_classes)
    return model

# ---------- Faster R-CNN · R101-FPN（輕改版，保留向下相容） ----------
def get_fasterrcnn_r101_fpn(
    num_classes=2, *, min_size=None, max_size=None,
    image_mean=(0.0,0.0,0.0), image_std=(1.0,1.0,1.0),
    cfg: Optional[Dict[str, Any]] = None
):
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
        box_detections_per_img=float(_maybe_get(cfg or {}, "model.box_detections_per_img", 100)),
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# ---------- RetinaNet-R50-FPN（保留；支援自訂 anchors） ----------
def get_retinanet_r50_fpn(
    num_classes=2, *, min_size=None, max_size=None,
    image_mean=(0.0,0.0,0.0), image_std=(1.0,1.0,1.0),
    cfg: Optional[Dict[str, Any]] = None
):
    min_size = _maybe_get(cfg or {}, "model.min_size", min_size)
    max_size = _maybe_get(cfg or {}, "model.max_size", max_size)
    image_mean = list(_maybe_get(cfg or {}, "model.image_mean", image_mean))
    image_std  = list(_maybe_get(cfg or {}, "model.image_std",  image_std))

    anc_sizes = _maybe_get(cfg or {}, "model.retinanet_anchor_sizes",
                           _maybe_get(cfg or {}, "model.rpn_anchor_sizes", [16, 32, 64, 128, 256]))
    sizes_tuple = _as_tuple_sizes(anc_sizes)
    while len(sizes_tuple) < 5:
        sizes_tuple += (sizes_tuple[-1],)
    sizes_tuple = sizes_tuple[:5]

    anc_ratios = _maybe_get(cfg or {}, "model.retinanet_anchor_aspect_ratios",
                            _maybe_get(cfg or {}, "model.rpn_anchor_ratios", [1.0, 2.0, 3.0]))
    ratios_per_level = (tuple(float(r) for r in anc_ratios),) * len(sizes_tuple)

    retina_num_classes = max(1, int(num_classes) - 1) if int(num_classes) > 1 else int(num_classes)

    model = torchvision.models.detection.retinanet_resnet50_fpn(
        weights=None, weights_backbone=None,
        num_classes=retina_num_classes,
        anchor_generator=AnchorGenerator(sizes=sizes_tuple, aspect_ratios=ratios_per_level),
        min_size=min_size, max_size=max_size,
        image_mean=image_mean, image_std=image_std,
    )
    return model

# ---------- SSDLite-MBV3（保留） ----------
def get_ssdlite_mbv3(
    num_classes=2, *, image_mean=(0.0,0.0,0.0), image_std=(1.0,1.0,1.0),
    cfg: Optional[Dict[str, Any]] = None
):
    model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(
        weights=None, weights_backbone=None,
        num_classes=max(1, int(num_classes) - 1) if int(num_classes) > 1 else int(num_classes),
        image_mean=list(_maybe_get(cfg or {}, "model.image_mean", image_mean)),
        image_std=list(_maybe_get(cfg or {}, "model.image_std", image_std)),
    )
    return model

# ---------- 工廠：由 YAML 的 model.detector 產生模型 ----------
_DET_BUILDERS = {
    "fasterrcnn_r50_fpn":   get_fasterrcnn_r50_fpn,
    "fasterrcnn_r101_fpn":  get_fasterrcnn_r101_fpn,      # 輕改版（相容）
    "fasterrcnn_r101_fpn_v7": get_fasterrcnn_r101_fpn_v7, # ★ 推薦 · 強化版
    "retinanet_r50_fpn":    get_retinanet_r50_fpn,
    "ssdlite_mbv3":         get_ssdlite_mbv3,
}

def build_detector_from_cfg(cfg: Dict[str, Any]):
    mcfg = cfg.get("model", {})
    name = str(mcfg.get("detector", "fasterrcnn_r101_fpn_v7")).lower()

    num_classes = int(mcfg.get("num_classes", 2))
    min_size = _maybe_get(cfg or {}, "model.min_size", None)
    max_size = _maybe_get(cfg or {}, "model.max_size", None)
    image_mean = tuple(_maybe_get(cfg or {}, "model.image_mean", (0.0, 0.0, 0.0)))
    image_std  = tuple(_maybe_get(cfg or {}, "model.image_std",  (1.0, 1.0, 1.0)))

    builder = _DET_BUILDERS.get(name, get_fasterrcnn_r101_fpn_v7)

    # RetinaNet/SSD：num_classes 轉「不含背景」的習慣在各自 builder 內處理
    if builder in (get_retinanet_r50_fpn, get_ssdlite_mbv3):
        return builder(
            num_classes=num_classes,
            min_size=min_size, max_size=max_size,
            image_mean=image_mean, image_std=image_std,
            cfg=cfg
        )

    # Faster R-CNN 路線
    return builder(
        num_classes=num_classes,
        min_size=min_size, max_size=max_size,
        image_mean=image_mean, image_std=image_std,
        cfg=cfg
    )
