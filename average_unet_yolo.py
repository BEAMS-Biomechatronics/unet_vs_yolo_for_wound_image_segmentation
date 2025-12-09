import pandas as pd
import numpy as np
from pathlib import Path
def compute_average_metrics(fold_files):
    """
    Calcule les moyennes des métriques (IoU, Precision, Recall, Dice) sur plusieurs fichiers.
    
    Parameters:
        fold_files (list): Liste des chemins des fichiers CSV des folds.
        
    Returns:
        pd.DataFrame: DataFrame contenant les moyennes pour chaque threshold.
    """
    all_folds = []

    # Charger chaque fichier CSV et ajouter à une liste
    for file in fold_files:
        df = pd.read_csv(file, header=None, names=['threshold', 'iou', 'precision', 'recall', 'dice'])
        all_folds.append(df)

    # Concaténer tous les folds sur l'axe des lignes
    combined_df = pd.concat(all_folds)

    # Calculer les moyennes par threshold
    avg_metrics = combined_df.groupby('threshold').mean().reset_index()

    return avg_metrics

def save_average_metrics(avg_metrics, output_file):
    """
    Sauvegarde les moyennes des métriques dans un fichier CSV.
    
    Parameters:
        avg_metrics (pd.DataFrame): DataFrame contenant les moyennes calculées.
        output_file (str): Nom du fichier de sortie.
    """
    avg_metrics.to_csv(output_file, index=False, header=False)
    print(f"Les moyennes ont été sauvegardées dans le fichier : {output_file}")

# Liste des fichiers CSV (à adapter selon vos fichiers)
fold_files = [
    "unet_average_ipi_metrics_combined_AZHtest_fold1.csv",
    "unet_average_ipi_metrics_combined_AZHtest_fold2.csv",
    "unet_average_ipi_metrics_combined_AZHtest_fold3.csv",
    "unet_average_ipi_metrics_combined_AZHtest_fold4.csv",
    "unet_average_ipi_metrics_combined_AZHtest_fold5.csv"
]

# Calculer les moyennes
avg_metrics = compute_average_metrics(fold_files)

# Sauvegarder dans un fichier CSV
output_dir = Path("combined_AZH_result/unet")
output_file = f'average_unet_combined_AZH2025.csv'
save_average_metrics(avg_metrics, output_file)

# Afficher les moyennes
print(avg_metrics)
