# src/train.py
import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from dataset import PigsDataset   # 你要自己寫的 Dataset
from transforms import get_transforms

def get_model(num_classes=2):
    # 建立 ResNet50 + FPN backbone，不載入任何 ImageNet 權重
    backbone = resnet_fpn_backbone('resnet50', weights=None, trainable_layers=0)
    model = FasterRCNN(backbone, num_classes=num_classes)
    return model

def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for images, targets in dataloader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()
    return total_loss / len(dataloader)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset & DataLoader
    train_dataset = PigsDataset("data/train/img", "data/train/gt.txt", transforms=get_transforms(train=True))
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

    # Model
    model = get_model(num_classes=2)
    model.to(device)

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # Training Loop
    for epoch in range(10):
        loss = train_one_epoch(model, train_loader, optimizer, device)
        print(f"[Epoch {epoch}] Loss: {loss:.4f}")
        scheduler.step()

        # 儲存 checkpoint
        torch.save(model.state_dict(), f"experiments/logs/model_epoch{epoch}.pth")

if __name__ == "__main__":
    main()
