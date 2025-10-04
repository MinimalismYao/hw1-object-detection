# 放哪都行，跑一次即可
import torch
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

bb = resnet_fpn_backbone(backbone_name='resnet50', weights=None)
x = torch.randn(1, 3, 800, 800)  # 假輸入
feats = bb(x)
print([f.shape[-2:] for f in feats.values()])  # 看有幾層
# 你會看到類似：[(100,100), (50,50), (25,25), (13,13), (7,7)] → 共 5 層
