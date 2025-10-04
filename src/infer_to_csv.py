# ==== 1) import 區補強 ====
import os, csv, glob
import time
import torch
from PIL import Image
import torchvision
from tqdm import tqdm

from model import get_fasterrcnn_r50_fpn

# ==== 2) 參數（保持你原本邏輯；先把 MAX_SIDE 不用外部縮放，交給 model 限制）====
IMG_DIR     = "data/test/img"
CKPT_PATH   = "experiments/logs/fasterrcnn_r50fpn_final_v2.pth"
OUT_CSV     = "submission_v2.csv"
MAX_SIDE    = 1024           # 不外部縮放；用它來限制「模型內部 transform」的 max_size
SCORE_THR   = 0.05           # 建議 0.05
MAX_IMAGES  = None
MAX_DETS    = 100            # 新增：每張最多保留 100

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== 3) 載模型：確保 eval、GPU、限制 transform 尺寸、AMP 更快 ====
def load_model(ckpt_path, device):
    model = get_fasterrcnn_r50_fpn(num_classes=2, freeze_backbone=False).to(device)
    # weights_only=True 更安全也更快一點（若你是舊版 torch，沒這參數就拿掉）
    try:
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()

    # ★ 關鍵：把模型內部 GeneralizedRCNNTransform 限制在我們想要的大小
    #   這樣就算你餵「原圖」，模型也只會把短邊拉到 MAX_SIDE、長邊最多 MAX_SIDE
    if hasattr(model, "transform"):
        if MAX_SIDE is not None:
            model.transform.min_size = (MAX_SIDE,)   # shorter side
            model.transform.max_size = MAX_SIDE      # longer side cap
        # 你訓練時若沒用 ImageNet normalize，推論也改成 0/1（和你自組版一致）
        model.transform.image_mean = [0.0, 0.0, 0.0]
        model.transform.image_std  = [1.0, 1.0, 1.0]

    return model

# ==== 4) 讀圖：不要外部 resize ====
def load_image(fp):
    img = Image.open(fp).convert("RGB")
    w, h = img.size
    return img, (w, h)  # 不縮放

# ==== 5) 推論主程：加 tqdm、AMP、自動排序取 Top-100 ====
@torch.inference_mode()
def run_infer_to_strings(model, img_dir, score_thr=0.05, device=None):
    device = device or DEVICE
    tfm = torchvision.transforms.ToTensor()

    img_files = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        img_files.extend(glob.glob(os.path.join(img_dir, ext)))

    def _key(fp):
        base = os.path.splitext(os.path.basename(fp))[0]
        try: return int(base)
        except ValueError: return base

    img_files = sorted(img_files, key=_key)
    if MAX_IMAGES is not None:
        img_files = img_files[:MAX_IMAGES]

    rows = []
    pbar = tqdm(img_files, desc="Infer", ncols=100)
    torch.backends.cudnn.benchmark = True  # ★ 加速卷積
    # 可選：限制 CPU 執行緒，避免和其他程式打架
    # torch.set_num_threads(1)

    for fp in pbar:
        fname = os.path.basename(fp)
        fid_str = os.path.splitext(fname)[0]
        try:
            fid = int(fid_str)
        except ValueError:
            continue

        img, (orig_w, orig_h) = load_image(fp)
        x = tfm(img).to(device).unsqueeze(0)

        t0 = time.perf_counter()
        # ★ AMP 自動混合精度（GPU 上會更快）
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            out = model(x)[0]
        torch.cuda.synchronize() if device.type == "cuda" else None
        pbar.set_postfix(sec=f"{(time.perf_counter()-t0):.3f}")

        boxes  = out["boxes"].detach().cpu().numpy()
        scores = out["scores"].detach().cpu().numpy()

        # 過濾 + 按分數排序 + 取前 MAX_DETS
        keep = scores >= score_thr
        boxes, scores = boxes[keep], scores[keep]
        order = scores.argsort()[::-1]
        if MAX_DETS is not None:
            order = order[:MAX_DETS]
        boxes, scores = boxes[order], scores[order]

        # xyxy -> xywh + 裁邊
        boxes[:, [0,2]] = boxes[:, [0,2]].clip(0, orig_w - 1)
        boxes[:, [1,3]] = boxes[:, [1,3]].clip(0, orig_h - 1)
        xywh = boxes.copy()
        xywh[:, 2] = (boxes[:, 2] - boxes[:, 0]).clip(min=0.0)
        xywh[:, 3] = (boxes[:, 3] - boxes[:, 1]).clip(min=0.0)

        parts = []
        for b, s in zip(xywh, scores):
            w = float(b[2]); h = float(b[3])
            if w <= 0.0 or h <= 0.0:
                continue
            parts.extend([
                f"{float(s):.6f}",
                f"{float(b[0]):.2f}",
                f"{float(b[1]):.2f}",
                f"{w:.2f}",
                f"{h:.2f}",
                "0"
            ])
        pred_str = " ".join(parts)
        rows.append((fid, pred_str))

    rows.sort(key=lambda x: x[0])
    return rows

def write_submission(rows, out_csv):
    d = os.path.dirname(out_csv)
    if d: os.makedirs(d, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Image_ID", "PredictionString"])
        w.writerows(rows)

def main():
    assert os.path.isdir(IMG_DIR), f"找不到影像資料夾：{IMG_DIR}"
    assert os.path.isfile(CKPT_PATH), f"找不到權重：{CKPT_PATH}"
    print(f"[Config] device={DEVICE}  ckpt={CKPT_PATH}  img_dir={IMG_DIR}  out={OUT_CSV}  thr={SCORE_THR}  max_side={MAX_SIDE}")

    model = load_model(CKPT_PATH, DEVICE)
    rows = run_infer_to_strings(model, IMG_DIR, score_thr=SCORE_THR, device=DEVICE)
    write_submission(rows, OUT_CSV)
    print(f"[Done] Wrote submission to: {OUT_CSV}")

if __name__ == "__main__":
    main()
