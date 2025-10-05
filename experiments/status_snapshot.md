# Status Snapshot — fasterrcnn_v3 (2025-10-05)

## 實驗識別
- run_name：`fasterrcnn_v3`
- 輸出資料夾：`experiments/logs/fasterrcnn_v3/`
- 最終權重：`experiments/logs/fasterrcnn_v3/fasterrcnn_v3.pth`
- 評估結果檔：`experiments/eval_results/fasterrcnn_v3_results.txt`

## 設定摘要
- 模型：Faster R-CNN (ResNet50-FPN), num_classes=2  
- 預訓練：pretrained = **false**  
- 輸入：max_side=1024，增強：flip_p=0.5（無 mosaic/HSV）  
- 訓練：epochs=30, batch_size=8, grad_clip=10.0, amp=false（開 cudnn_benchmark）  
- 優化器/排程：SGD(lr=0.003, momentum=0.9, weight_decay=1e-4)；StepLR(step=5, gamma=0.1)  
- 推論：score_thr=0.05, nms_iou=0.5  
- 環境：CUDA 可用；dataloader: pin_memory, persistent_workers, prefetch_factor=4  
（以上來自 `config.final.json`）  
> 參考：`infer.submission_csv = submissions/fasterrcnn_v3_submission.csv`（輸出檔名）  
> 參考：`eval.result_txt = experiments/eval_results/fasterrcnn_v3_results.txt`（評估輸出）  
