# Plant Diseases App (Streamlit)

Application Streamlit pour la classification des maladies des plantes (XGBoost).  
Fichiers principaux :
- `SN_Etape3.py` : application Streamlit (UI, visualisations, prédiction).
- `plant_disease_model_xgb.sav` : modèle XGBoost sérialisé (doit être présent ou téléchargeable).
- `plant_disease_dataset.csv`, `dataset_machine_learning.csv` : données utilisées.
- `Photos/` : images affichées dans l'app.

## Fonctionnalités
- Visualisation des données (histogrammes, boxplots, heatmap).
- Evaluation du modèle (accuracy, matrice de confusion, rapport de classification).
- Interface de prédiction interactive (saisie des caractéristiques de la plante).

## Prérequis
- Python 3.8+
- Recommandé : environnement virtuel (venv)
- Fichier modèle `plant_disease_model_xgb.sav` (placer à la racine du repo ou configurer un téléchargement automatique)
