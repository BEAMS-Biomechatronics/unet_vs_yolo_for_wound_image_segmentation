
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
def plot_metric_vs_threshold(csv_paths, metric, output_file, title, ylabel):
    """
    Trace un graphique pour une métrique donnée (IoU, Précision, Rappel, Dice) en fonction du Threshold,
    en superposant les courbes pour plusieurs fichiers CSV.

    Parameters:
        csv_paths (list): Liste des chemins des fichiers CSV.
        metric (str): La métrique à tracer ('iou', 'precision', 'recall', 'dice').
        output_file (str): Nom du fichier de sortie pour l'image.
        title (str): Titre du graphique.
        ylabel (str): Étiquette de l'axe Y.
    """
    plt.figure(figsize=(10, 6))
    
    for csv_path in csv_paths:
        # Extraire le nom du fichier pour la légende
        label = csv_path.split('/')[-1].split('.')[0]  # Nom du fichier sans extension
        
        # Charger le fichier CSV
        df = pd.read_csv(csv_path, header=None, names=['threshold', 'iou', 'precision', 'recall', 'dice'])
        df = df.sort_values(by='threshold')
        
        # Tracer la courbe pour la métrique donnée
        plt.plot(df['threshold'], df[metric], marker='o', label=label)

    # Ajouter les détails du graphique
    plt.title(title)
    plt.xlabel('Threshold')
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend(title="Fichiers CSV", loc='best')
    plt.savefig(output_file)
    plt.close()
    print(f"Graphique sauvegardé : {output_file}")

# Liste des chemins des fichiers CSV
csv_paths = [
    f"result/combined_kaggle/average_yolov8l_combined_kaggle.csv",
    f"result/combined_kaggle/average_yolov8m_combined_kaggle.csv",
    f"result/combined_kaggle/average_yolov8n_combined_kaggle.csv",
    f"result/combined_kaggle/average_yolov8s_combined_kagglex.csv",
    f"result/combined_kaggle/average_yolov8x_combined_kaggle.csv",
    
]
output_dir = Path("result")
# Appel des fonctions de tracé pour chaque métrique
plot_metric_vs_threshold(
    csv_paths, 
    metric='iou', 
    output_file=output_dir/ f'iou2_combined_kagglex.svg', 
    title='IoU en fonction du Threshold (Combinaison)', 
    ylabel='IoU'
)

plot_metric_vs_threshold(
    csv_paths, 
    metric='precision', 
    output_file=output_dir/ f'precision2_combined_kagglex.svg', 
    title='Precision vs Threshold (Combinaison)', 
    ylabel='Precision'
)

plot_metric_vs_threshold(
    csv_paths, 
    metric='recall', 
    output_file=output_dir/ f'recall2_combined_kagglex.svg', 
    title='Recall vs Threshold (Combinaison)', 
    ylabel='Recall'
)

plot_metric_vs_threshold(
    csv_paths, 
    metric='dice', 
    output_file=output_dir/ f'dice2_combined_kagglex.svg', 
    title='Dice Coefficient vs Threshold (Combinaison)', 
    ylabel='Dice coefficient'
)
