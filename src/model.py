# src/model.py
import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.detection.backbone_utils import BackboneWithFPN


def _build_resnet50_fpn(norm_layer=nn.BatchNorm2d):
    """
    從零開始構建 ResNet50 + FPN（不載任何預訓練權重）。
    - 使用可學習 BatchNorm2d（from scratch 更好收斂）
    - 輸出 FPN，通道數固定 256
    """
    resnet = torchvision.models.resnet50(weights=None, norm_layer=norm_layer)

    # ResNet 結構：layer1→C2, layer2→C3, layer3→C4, layer4→C5
    return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
    in_channels_list = [256, 512, 1024, 2048]

    body = IntermediateLayerGetter(resnet, return_layers=return_layers)
    backbone = BackboneWithFPN(
        body,
        return_layers=return_layers,
        in_channels_list=in_channels_list,
        out_channels=256,
        extra_blocks=LastLevelMaxPool(),  # 產生 P6
    )
    return backbone  # .out_channels = 256


def _as_tuple_sizes(sizes_like):
    """
    將 YAML 可能的 [[8],[16],[32],[64],[128]] 轉為 ((8,), (16,), ...)，
    若給的是 [32,64,128,...] 也會轉成 ((32,), (64,), ...).
    """
    if sizes_like is None:
        return ((32,), (64,), (128,), (256,), (512,))
    if isinstance(sizes_like, (list, tuple)) and len(sizes_like) > 0:
        # [[8],[16],...] 或 [8,16,...]
        if isinstance(sizes_like[0], (list, tuple)):
            return tuple(tuple(int(s[0])) if isinstance(s, (list, tuple)) else (int(s),) for s in sizes_like)
        else:
            return tuple((int(s),) for s in sizes_like)
    return ((32,), (64,), (128,), (256,), (512,))


def get_fasterrcnn_r50_fpn(
    num_classes: int = 2,
    freeze_backbone: bool = False,
    # 你可以忽略下列參數，改由 cfg 來覆蓋
    min_size: int = 1024,
    max_size: int = 1024,
    image_mean=(0.0, 0.0, 0.0),
    image_std=(1.0, 1.0, 1.0),
    rpn_pre_nms_top_n_train: int = 2000,
    rpn_pre_nms_top_n_test: int = 1000,
    rpn_post_nms_top_n_train: int = 1000,
    rpn_post_nms_top_n_test: int = 1000,
    rpn_nms_thresh: float = 0.7,
    box_score_thresh: float = 0.0,
    box_nms_thresh: float = 0.5,
    box_detections_per_img: int = 100,
    cfg: dict | None = None,
):
    """
    Faster R-CNN (ResNet50-FPN) - from scratch 版，支援從 cfg 讀 anchors/RPN 參數。
    建議把外部前處理的縮放策略（max_side）與這裡的 min/max_size 對齊以避免雙重縮放。
    """
    # ===== 讀取 YAML 覆蓋（若提供 cfg） =====
    mcfg = (cfg or {}).get("model", {})
    acfg = (cfg or {}).get("augment", {})

    # 解析度（與你的 transforms 一致）：以長邊上限為 max_size，為避免雙重縮放，min/max 取同值
    ms = int(acfg.get("max_side", max_size))
    min_size = ms
    max_size = ms

    # Anchors：支援更小尺寸
    sizes  = _as_tuple_sizes(mcfg.get("rpn_anchor_sizes", None))
    ratios = mcfg.get("rpn_anchor_ratios", [0.5, 1.0, 2.0])

    # RPN 提案數（若 YAML 有就覆蓋）
    rpn_pre_nms_top_n_train = int(mcfg.get("rpn_pre_nms_top_n_train", rpn_pre_nms_top_n_train))
    rpn_post_nms_top_n_train = int(mcfg.get("rpn_post_nms_top_n_train", rpn_post_nms_top_n_train))
    rpn_pre_nms_top_n_test  = int(mcfg.get("rpn_pre_nms_top_n_test",  rpn_pre_nms_top_n_test))
    rpn_post_nms_top_n_test = int(mcfg.get("rpn_post_nms_top_n_test", rpn_post_nms_top_n_test))
    rpn_nms_thresh = float(mcfg.get("rpn_nms_thresh", rpn_nms_thresh))

    # ===== Backbone + FPN =====
    backbone = _build_resnet50_fpn(norm_layer=nn.BatchNorm2d)

    # ===== Anchor generator（用 YAML 覆蓋） =====
    anchor_generator = AnchorGenerator(
        sizes=sizes,                          # 例如 ((8,), (16,), (32,), (64,), (128,))
        aspect_ratios=(tuple(ratios),) * len(sizes),
    )

    # ===== RoIAlign =====
    roi_pooler = MultiScaleRoIAlign(
        featmap_names=["0", "1", "2", "3"],
        output_size=7,
        sampling_ratio=2,
    )

    # ===== FasterRCNN 主體 =====
    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        # ---- GeneralizedRCNNTransform ----
        min_size=min_size,
        max_size=max_size,
        image_mean=list(image_mean),
        image_std=list(image_std),
        # ---- RPN/ROI 參數（覆蓋）----
        rpn_pre_nms_top_n_train=rpn_pre_nms_top_n_train,
        rpn_post_nms_top_n_train=rpn_post_nms_top_n_train,
        rpn_pre_nms_top_n_test=rpn_pre_nms_top_n_test,
        rpn_post_nms_top_n_test=rpn_post_nms_top_n_test,
        rpn_nms_thresh=rpn_nms_thresh,
        box_score_thresh=box_score_thresh,    # 訓練保持 0，推論由 infer.score_thr 控制
        box_nms_thresh=box_nms_thresh,
        box_detections_per_img=box_detections_per_img,
    )

    # 凍結 backbone（如需要）
    for p in model.backbone.parameters():
        p.requires_grad = (not freeze_backbone)

    # 保險：替換分類頭以確保 num_classes 正確
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model
