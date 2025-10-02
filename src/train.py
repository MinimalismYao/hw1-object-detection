# src/train.py
import os, torch
from torch.utils.data import DataLoader
from dataset import PigsDataset, collate_fn
from transforms import get_transforms
from model import get_fasterrcnn_r50_fpn

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    avg_loss = 0.0
    for images, targets in loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss += loss.item()
    return avg_loss / max(1, len(loader))

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = PigsDataset(img_dir="data/train/img",
                           gt_txt="data/train/gt.txt",
                           transforms=get_transforms(train=True))
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True,
                              num_workers=2, collate_fn=collate_fn)

    model = get_fasterrcnn_r50_fpn(num_classes=2, freeze_backbone=True).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    os.makedirs("experiments/logs", exist_ok=True)

    for epoch in range(2):  # 先跑短一點驗證流程
        loss = train_one_epoch(model, train_loader, optimizer, device)
        print(f"[Epoch {epoch}] loss={loss:.4f}")
        scheduler.step()
        torch.save(model.state_dict(), f"experiments/logs/fasterrcnn_r50fpn_e{epoch}.pth")

if __name__ == "__main__":
    main()
