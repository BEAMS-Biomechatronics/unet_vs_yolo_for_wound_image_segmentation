# Image segmentation with Unet and YOLOv8/11 with Python
***This code is for research purposes only.***

This repository contains the code associated with the publication: *paper to be submitted*.

## Overview

### ***Unet***
### `unet_model.py`
Definition of the U-Net architecture used for binary segmentation of chronic wound images.  
The model follows a classical encoder–decoder structure with skip connections and a sigmoid output layer.

### `train.py`
Main training script for the U-Net pipeline.  
This script performs 5-fold cross-validation and handles model training and validation.

This is where you need to specify:

(1) all parameters concerning the training procedure:
- *img_size*: input image resolution
- *batch_size*: batch size
- *epochs*: maximum number of epochs
- *learning_rate*: initial learning rate
- *optimizer*: optimization algorithm
- *patience*: early stopping patience

(2) the specifics regarding your dataset:
- *data_path*: path to the dataset
- *folds*: number of cross-validation folds
- *train/val split*: handled per fold
- *save_path*: folder in which trained weights are saved

The data for each fold should be organised into:
- *images*: RGB wound images
- *masks*: binary segmentation masks

---

### `predict.py`
Script handling inference with the trained U-Net models.  
It generates probability maps and computes segmentation metrics.

Outputs include:
- per-image metrics: IoU, Precision, Recall, Dice
- threshold-based evaluation (0.1 → 0.9)
- CSV files saved per fold

---

### `average.py`
Script aggregating segmentation metrics across the 5 folds.  
It produces global performance values for each metric.

---

### `plot.py`
Script generating visualisations of the segmentation performance, including metric evolution across thresholds.

---
---

### ***YOLO***
### `training.py`
Main training script for YOLO-based segmentation models (YOLOv8 and YOLO11).  
It performs 5-fold cross-validation using Ultralytics implementations.

This is where you need to specify:
- *model type*: YOLOv8 or YOLO11
- *model size*: n, s, m, l, or x
- *img_size*: input resolution
- *batch_size*
- *epochs*
- *dataset YAML files*

Training parameters and notes are saved in `yolo_training_info.txt`.

---

### `predict.py`
Inference script for YOLOv8 and YOLO11 segmentation models.  
For each fold, it:
- loads the best trained model
- runs inference on test images
- aggregates segmentation masks
- applies thresholds from 0.1 to 0.9
- computes IoU, Precision, Recall, Dice, TP, FP, TN, FN

---

### `average_unet_yolo.py`
Script aggregating YOLO segmentation metrics across folds to produce global performance results.

---

### `plotarticle.py`
Script generating publication-ready figures used in the associated manuscript.

---

## Dependencies

All dependencies required to run the code are listed in `requirements.txt`.

Main libraries include:
- Python
- TensorFlow / Keras
- PyTorch
- Ultralytics YOLO
- NumPy, Pandas, Matplotlib

---

## Usage

1. Organise the datasets according to the required U-Net or YOLO structure
2. Run the training scripts for the selected model family
3. Perform inference and metric computation
4. Aggregate results and generate figures

Each pipeline (U-Net and YOLO) can be executed independently.

---

## Contributors

Indrani Marchal

---

## License

If you use this algorithm for a publication (in a journal, in a conference, etc.), please cite the related publications (see below). The license attached to this toolbox is GPL v2, see https://www.gnu.org/licenses/gpl-2.0.txt. From https://www.gnu.org/licenses/gpl-2.0.html, it implies: This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 2 of the License, or (at your option) any later version.


---

## Citation

If you use this code or data in your research, please cite the following paper: *paper to be submitted*.


