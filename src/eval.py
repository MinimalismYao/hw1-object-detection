# src/eval.py
import torch
from dataset import PigsDataset, collate_fn
from transforms import get_transforms
from model import get_fasterrcnn_r50_fpn

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = PigsDataset(img_dir="data/train/img", gt_txt="data/train/gt.txt",
                     transforms=get_transforms(train=False))
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False,
                                         num_workers=2, collate_fn=collate_fn)

    model = get_fasterrcnn_r50_fpn(num_classes=2, freeze_backbone=True).to(device)
    ckpt = "experiments/logs/fasterrcnn_r50fpn_e1.pth"
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    # TODO: 接 COCO evaluator 算 mAP50:95
    # 這裡先示範跑一次 forward，檢查無錯
    with torch.no_grad():
        for images, _ in loader:
            images = [img.to(device) for img in images]
            outputs = model(images)  # list of dict
            break
    print("[Eval] forward OK. 後續接 COCO mAP 計算。")

if __name__ == "__main__":
    main()
