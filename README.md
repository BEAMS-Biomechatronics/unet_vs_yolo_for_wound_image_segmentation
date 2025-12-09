This repository provides the full reproducible pipeline used in our study for the segmentation of chronic wounds using three model families:

U-Net (TensorFlow/Keras)

YOLOv8-seg (Ultralytics)
YOLO11-seg (Ultralytics)

The codebase includes: cross-validation, training, inference, per-threshold metric computation (IoU, Precision, Recall, Dice), aggregated results, and publication-ready plots.

This repository implements all steps required for transparency and reproducibility, as recommended for clinical machine-learning workflows.


Structure
.
├── unet/
│   ├── unet_model.py                 # U-Net architecture                 :contentReference[oaicite:0]{index=0}
│   ├── train.py                      # U-Net cross-validation training     :contentReference[oaicite:1]{index=1}
│   ├── predict.py                    # U-Net inference + metrics           :contentReference[oaicite:2]{index=2}
│   ├── average.py                    # Average metrics across folds        :contentReference[oaicite:3]{index=3}
│   ├── plot.py                       # IoU / ROC plot utilities            :contentReference[oaicite:4]{index=4}
│
├── yolo/
│   ├── training.py                   # YOLOv8/YOLO11 training (5-fold)     :contentReference[oaicite:5]{index=5}
│   ├── predict.py                    # YOLO inference + metrics            :contentReference[oaicite:6]{index=6}
│   ├── average_metric.py             # YOLO metric aggregation             :contentReference[oaicite:7]{index=7}
│   ├── plots.py                      # Simple metric plots                 :contentReference[oaicite:8]{index=8}
│   ├── plotarticle.py                # Publication-ready plots             :contentReference[oaicite:9]{index=9}
│   ├── yolo_training_info.txt        # Training parameters + notes         :contentReference[oaicite:10]{index=10}
│
├── notebooks/
│   └── Inference_with_Mask2Former.ipynb   # Baseline comparison             :contentReference[oaicite:11]{index=11}
│
├── requirements.txt
└── README.md


Dataset organisation
For U_Net
dataset/
├── fold_1/
│   ├── train/images
│   ├── train/masks
│   ├── val/images
│   ├── val/masks
│   └── test/images
│       └── test/masks
├── fold_2/
...

For YOLO
dataset_yolo/
├── fold_1.yaml
├── fold_2.yaml
...
path: /path/to/dataset
train: train/images
val: val/images
test: test/images
names:
  0: wound

U_Net pipeline          
unet_model.py

A classical encoder–decoder architecture:
  Downsampling blocks (conv–conv–maxpool)
  Bottleneck
  Upsampling blocks with skip-connections
  Sigmoid output mask

Train : python unet/train.py
Features:

Sequential training of folds 1–5
Data generators handling resizing & normalisation
Training metrics: Accuracy, MeanIoU
Saves weights:unet_weights_combined_AZHtest_fold_1.h5

Metrics : python unet/predict.py
Outputs for each fold:
Per-image metrics (IoU, Precision, Recall, Dice)
Per-threshold results (0.1 → 0.9)
CSV files saved per fold

average_metric__per_fold: python unet/average.py

Visualisation : python unet/plot.py

_______________________________________________________________________________
Pipeline (YOLO)
Supported models:

YOLOv8: n, s, m, l, x
YOLO11: n, s, m, l, x
All models use segmentation heads.

Train : python yolo/training.py
Features:
5-fold cross-validation
Training parameters recorded in yolo_training_info.txt

Metric: python yolo/predict.py
or each fold:
Loads YOLO best.pt
Runs inference on test images
Aggregates mask channels
Applies thresholds 0.1 → 0.9
Computes:IoU; Precision; Recall; Dice; TP / FP / TN / FN

average metrics : python yolo/average_metric.py

visualisation : python yolo/plotarticle.py


