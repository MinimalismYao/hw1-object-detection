<!-- cSpell:disable -->
# README.md (初稿)
13213213
# HW1 - Object Detection  
TAICA 課程作業專案  

## 專案說明
本專案為 **HW1 - Object Detection** 作業，目標是利用指定資料集 (豬隻偵測) 訓練物件偵測模型，並輸出符合 Kaggle 格式的 CSV 進行評估。  
本專案採用 **Faster R-CNN + ResNet-50 (ImageNet backbone 凍結)** 作為 baseline 模型，並嘗試透過調整 anchor、NMS、資料增強等方式改善 mAP50:95。

---

## 專案結構

```
hw1_<student-id>/
├── data/                        # 資料集 (train/test/gt.txt)，不會上傳 GitHub
│
├── src/                         # 主要程式碼
│   ├── dataset.py               # Dataset 定義，讀取影像與標註
│   ├── transforms.py            # Data augmentation 與前處理
│   ├── model.py                 # Faster R-CNN 模型 (ResNet50 backbone 凍結)
│   ├── train.py                 # 訓練流程
│   ├── eval.py                  # 驗證 mAP50:95
│   ├── infer_to_csv.py          # 推論並輸出 Kaggle CSV
│   └── utils.py                 # 小工具 (IoU、可視化等)
│
├── experiments/                 # 實驗設定與紀錄
│   ├── configs/                 # YAML/JSON 設定檔
│   └── logs/                    # 訓練與驗證 log
│
├── report_<student-id>.pdf      # 報告 (3–5 頁)
├── readme.md                    # 專案說明文件 (本檔案)
├── requirements.txt             # 套件需求清單
└── code_<student-id>.zip        # 繳交壓縮檔 (助教指定格式)
```

---

## 環境安裝
建議使用 **Python 3.10+** 及 **conda/venv**。  

```bash
# 建立虛擬環境
conda create -n hw1 python=3.10 -y
conda activate hw1

# 安裝必要套件
pip install -r requirements.txt
```

requirements.txt 包含：

* torch, torchvision
* numpy, pandas
* matplotlib
* pycocotools

---

## 使用方法

### 1. 準備資料

將 Kaggle 提供的 dataset 放入 `data/`：

```
data/
├── train/         # 訓練影像
├── test/          # 測試影像
└── gt.txt         # 標註檔案
```

---

### 2. 訓練模型

```bash
python src/train.py --config experiments/configs/baseline.yaml
```

---

### 3. 驗證模型

```bash
python src/eval.py --checkpoint experiments/logs/best_model.pth
```

---

### 4. 產生 Kaggle 提交檔

```bash
python src/infer_to_csv.py --checkpoint experiments/logs/best_model.pth --out submission.csv
```

---

## 報告內容 (簡要)

* **Model Description**：Faster R-CNN + ResNet-50 (backbone frozen)
* **Implementation Details**：前處理、augmentation、訓練策略
* **Result Analysis**：mAP50:95 分數表格、成功/失敗案例
* **Conclusion**：總結與反思

---

## 作者

* 學校：`NTUT`
* 學號：`114408051`
* 課程：TAICA - CVPDL

