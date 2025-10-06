# src/modelv4.py
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
    從零開始構建 ResNet50 + FPN（不載入任何預訓練權重）。
    - 使用可學習的 BatchNorm2d（from scratch 更好收斂）
    - 輸出 FPN，通道數固定為 256
    """
    # 1) ResNet50 backbone（無預訓練、可學習 BN）
    resnet = torchvision.models.resnet50(weights=None, norm_layer=norm_layer)

    # 2) 指定要抽出的層（C2~C5）
    #    ResNet 結構：layer1→C2, layer2→C3, layer3→C4, layer4→C5
    return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
    in_channels_list = [256, 512, 1024, 2048]

    # 3) 用 IntermediateLayerGetter 擷取中間特徵後，包成 FPN
    body = IntermediateLayerGetter(resnet, return_layers=return_layers)
    backbone = BackboneWithFPN(
        body,
        return_layers=return_layers,
        in_channels_list=in_channels_list,
        out_channels=256,
        extra_blocks=LastLevelMaxPool(),  # 產生 P6
    )
    return backbone  # 具備 .out_channels = 256


def get_fasterrcnn_r50_fpn(
    num_classes: int = 2,
    freeze_backbone: bool = False,
    # ---- GeneralizedRCNNTransform 參數（建議和你的資料前處理對齊）----
    min_size: int = 1024,   # shorter side
    max_size: int = 1024,   # longer side 上限（避免雙重縮放）
    image_mean=(0.0, 0.0, 0.0),
    image_std=(1.0, 1.0, 1.0),
    # ---- RPN / ROI 常用可調參（維持預設就好，需要再改）----
    rpn_pre_nms_top_n_train: int = 2000,
    rpn_pre_nms_top_n_test: int = 1000,
    rpn_post_nms_top_n_train: int = 1000,
    rpn_post_nms_top_n_test: int = 1000,
    rpn_nms_thresh: float = 0.7,
    box_score_thresh: float = 0.05,
    box_nms_thresh: float = 0.5,
    box_detections_per_img: int = 100,
):
    """
    Faster R-CNN (ResNet50-FPN) - from scratch 自組版：
    - 絕不載入預訓練（weights=None）
    - BatchNorm2d 可學（非 FrozenBN）
    - 完整可控的 Transform/RPN/ROI
    """
    # 1) Backbone with FPN（from scratch, BN 可學）
    backbone = _build_resnet50_fpn(norm_layer=nn.BatchNorm2d)

    # 2) Anchor generator（可依 bbox 分布調整）
    #    預設 5 個 FPN level，各用 (32, 64, 128, 256, 512) 與三種長寬比
    anchor_generator = AnchorGenerator(
        sizes=((32,), (64,), (128,), (256,), (512,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5,
    )

    # 3) RoIAlign（對應 FPN 的多尺度特徵）
    roi_pooler = MultiScaleRoIAlign(
        featmap_names=["0", "1", "2", "3"],
        output_size=7,
        sampling_ratio=2,
    )

    # 4) 建 FasterRCNN，顯式指定 Transform 參數以避免雙重縮放/不一致的 normalize
    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_classes,                 # 2：背景+豬
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        # ---- GeneralizedRCNNTransform ----
        min_size=min_size,
        max_size=max_size,
        image_mean=list(image_mean),
        image_std=list(image_std),
        # ---- RPN/ROI inference 參數（必要時可微調）----
        rpn_pre_nms_top_n_train=rpn_pre_nms_top_n_train,
        rpn_pre_nms_top_n_test=rpn_pre_nms_top_n_test,
        rpn_post_nms_top_n_train=rpn_post_nms_top_n_train,
        rpn_post_nms_top_n_test=rpn_post_nms_top_n_test,
        rpn_nms_thresh=rpn_nms_thresh,
        box_score_thresh=box_score_thresh,
        box_nms_thresh=box_nms_thresh,
        box_detections_per_img=box_detections_per_img,
    )

    # 5) 可選：凍結或解凍 backbone（from scratch 建議 False）
    for p in model.backbone.parameters():
        p.requires_grad = (not freeze_backbone)

    # 6) 保險：替換分類頭以確保 num_classes 正確（有些版本需要）
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model
