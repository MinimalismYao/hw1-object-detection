# src/infer_to_csv.py
import os, csv, torch
from dataset import PigsDataset, collate_fn
from transforms import get_transforms
from model import get_fasterrcnn_r50_fpn

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_ds = PigsDataset(img_dir="data/test/img", gt_txt=None,
                          transforms=get_transforms(train=False))
    loader = torch.utils.data.DataLoader(test_ds, batch_size=1, shuffle=False,
                                         num_workers=2, collate_fn=collate_fn)

    model = get_fasterrcnn_r50_fpn(num_classes=2, freeze_backbone=True).to(device)
    ckpt = "experiments/logs/fasterrcnn_r50fpn_e1.pth"
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    os.makedirs("experiments", exist_ok=True)
    out_csv = "experiments/submission.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Image_ID", "PredictionString"])
        # TODO: 依競賽格式組 PredictionString（score x y w h 以空白分隔）
        for i, (img, _) in enumerate(loader, 1):
            preds = model([img[0].to(device)])[0]
            # 這裡先寫空字串，後續你把 preds['boxes'], preds['scores'] 組起來
            writer.writerow([i, ""])

    print(f"[Infer] wrote to {out_csv}")

if __name__ == "__main__":
    main()
