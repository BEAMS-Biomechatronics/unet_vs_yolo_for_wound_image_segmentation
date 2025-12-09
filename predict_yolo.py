import os
from ultralytics import YOLO
import numpy as np
import csv
from pathlib import Path
import cv2

def calculate_metrics(mask1, mask2):
    assert mask1.shape == mask2.shape
    
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)
    
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou = np.sum(intersection) / np.sum(union)

    true_positive = np.sum(intersection)
    true_negative = np.sum(np.logical_and(np.logical_not(mask1), np.logical_not(mask2)))
    false_positive = np.sum(np.logical_and(np.logical_not(mask1), mask2))
    false_negative = np.sum(np.logical_and(mask1, np.logical_not(mask2)))
    
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive > 0 else 0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative > 0 else 0
    dice_coefficient = 2 * true_positive / (2 * true_positive + false_positive + false_negative) if true_positive + false_positive + false_negative > 0 else 0
    
    return {"tp": true_positive, "tn": true_negative, "fp": false_positive, "fn": false_negative, 
            "iou": iou, "precision": precision, "recall": recall, "dice_coefficient": dice_coefficient}

path = Path("yolo11s-seg/combined_AZHtest_yamls1") #chemin vers le modele qu'on a entraine et que l'on veut tester
base_path = Path("C:\\Users\\Haroun\\Desktop\\YOLO_CROSS_VALIDATION\\combined_AZHtest_folded1")  #chemin vers le dataset où ce trouve les images à tester 

output_dir = Path("combined_AZH_result/yolo11s")

# Créer les dossiers nécessaires
output_dir.mkdir(parents=True, exist_ok=True)

for i in range(1, 6):
    fold_path = path / f'fold_{i}/train/weights/best.pt'
    model = YOLO(str(fold_path))
    source_path = base_path / f'fold_{i}/test/images'
    elements = model.predict(source=source_path, split="test", device=0, save=True, project=f"predict/yolo11s-seg/combined_AZH/fold_{i}") #changer le nom du fichier output crée avec les résultats 
    
    thresholds = np.arange(0.1, 1.0, 0.1)
    metrics_sum = {th: {"iou": 0, "precision": 0, "recall": 0, "dice_coefficient": 0} for th in thresholds}
    elem = os.listdir(source_path)
    elem_count = len(elem)
    element_counts = {th: elem_count for th in thresholds} 
    
    result_file =  output_dir/f'result_1_yolo11s_combined_AZH_fold{i}.csv'
    avg_result_file = output_dir/f'avg_1_yolo11s_combined_AZH_fold{i}.csv'
    
    with open(result_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["image", "threshold", "tp", "tn", "fp", "fn", "iou", "precision", "recall", "dice_coefficient"])
        
        for element in elements:
            head, tail = os.path.split(element.path)
            new_head = head.replace('images', 'masks')
            file_name, ext = os.path.splitext(tail)
            new_tail = f'{file_name}.png'
            new_path = os.path.join(new_head, new_tail)
            grd_truth = cv2.imread(new_path, 0)
            
            if element.masks is not None and len(element.masks.data) > 0:
                mask = element.masks.data.cpu().numpy()
                for threshold in thresholds:
                    combined_mask = np.max(mask, axis=0)
                    thresholded_mask = (combined_mask > threshold).astype(int)
                    grd_truth_resized = cv2.resize(grd_truth, (thresholded_mask.shape[1], thresholded_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
                    results = calculate_metrics(thresholded_mask, grd_truth_resized)
                    row = [element.path, threshold] + [results[metric] for metric in ["tp", "tn", "fp", "fn", "iou", "precision", "recall", "dice_coefficient"]]
                    writer.writerow(row)

                    for metric in metrics_sum[threshold]:
                        metrics_sum[threshold][metric] += results[metric]
    
    with open(avg_result_file, mode='a', newline='') as avg_file:
        avg_writer = csv.writer(avg_file)
        for threshold in thresholds:
            if element_counts[threshold] > 0:
                mean_metrics = {metric: metrics_sum[threshold][metric] / element_counts[threshold] for metric in metrics_sum[threshold]}
                row = [threshold] + [mean_metrics[metric] for metric in mean_metrics]
                avg_writer.writerow(row)
