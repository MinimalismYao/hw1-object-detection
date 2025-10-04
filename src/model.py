# src/model.py
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.rpn import AnchorGenerator

def get_fasterrcnn_r50_fpn(num_classes=2, freeze_backbone=True):
    backbone = resnet_fpn_backbone(backbone_name='resnet50', weights=None,
                                   trainable_layers=0 if freeze_backbone else 5)

    # 5 個 feature maps → 必須給 5 組 sizes / aspect_ratios
    anchor_sizes   = ((32,), (64,), (128,), (256,), (512,))
    aspect_ratios  = ((0.75, 1.0, 1.5),) * 5
    rpn_anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)

    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=rpn_anchor_generator,
        box_score_thresh=0.05,
        box_nms_thresh=0.5,
        box_detections_per_img=300
    )
    if freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad = False
    return model
