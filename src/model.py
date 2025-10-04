# src/model.py
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.rpn import AnchorGenerator

def get_fasterrcnn_r50_fpn(num_classes=2, freeze_backbone=True):
    # 不載任何 ImageNet 權重
    backbone = resnet_fpn_backbone('resnet50', weights=None,
                                   trainable_layers=0 if freeze_backbone else 5)

    # 依你的豬尺寸做一個合理 anchor（可再微調）
    # 假設多數框在 40~160 寬高，FPN P3~P6
    anchor_sizes = ((32, 64), (64, 128), (128, 256), (256, 512))
    aspect_ratios = ((0.75, 1.0, 1.5),) * len(anchor_sizes)
    rpn_anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)

    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=rpn_anchor_generator,
        box_score_thresh=0.05,   # baseline
        box_nms_thresh=0.5,      # baseline
        box_detections_per_img=300
    )

    # 保險起見，若 freeze_backbone，把 backbone 參數設為不更新
    if freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad = False

    return model
