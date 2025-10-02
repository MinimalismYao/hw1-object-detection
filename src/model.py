# src/model.py
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

def get_fasterrcnn_r50_fpn(num_classes=2, freeze_backbone=True):
    # 不載入任何 ImageNet 權重
    backbone = resnet_fpn_backbone('resnet50', weights=None, trainable_layers=0 if freeze_backbone else 5)
    model = FasterRCNN(backbone, num_classes=num_classes)
    return model
