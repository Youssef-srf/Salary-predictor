import pandas as pd
import os

# Chargement du jeu de données nettoyé (après le notebook 01_preprocess_data)
# On s'assure que le chemin dynamique est respecté
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(base_dir, "data", "processed", "salary_dataset_processed.csv")
out_path = os.path.join(base_dir, "data", "final", "train_ready_dataset.csv")

df = pd.read_csv(data_path)

# Renommage des colonnes en snake_case
df = df.rename(columns={'Age': 'age', 
                        'Gender': 'gender',
                        'Education Level': 'education_level',
                        'Job Title' : 'job_title',
                        'Years of Experience' : 'years_of_experience',
                        'Salary' : 'salary'})   

# Le traitement des valeurs aberrantes (Outliers) a été déplacé dans train.py
# pour éviter le Data Leakage (fuite de données) entre le jeu d'entraînement et de test.

os.makedirs(os.path.dirname(out_path), exist_ok=True)
df.to_csv(out_path, index=False)

print(f"Dataset renommé et sauvegardé : {len(df)} lignes")
print(f"Colonnes : {list(df.columns)}")
