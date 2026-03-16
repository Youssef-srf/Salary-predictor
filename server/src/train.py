import pandas as pd
import numpy as np
import os
import joblib
import json
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ============================================================
# Chemins d'accès absolus
# ============================================================
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(base_dir, 'data', 'final', 'train_ready_dataset.csv')
models_dir = os.path.join(base_dir, 'models')
os.makedirs(models_dir, exist_ok=True)

# ============================================================
# 1. Chargement du dataset nettoyé (sans valeurs manquantes)
# ============================================================
df = pd.read_csv(data_path)
print(f"Dataset chargé : {df.shape[0]} lignes, {df.shape[1]} colonnes")

# ============================================================
# 2. Séparation features / cible AVANT le nettoyage IQR
#    (Pour éviter le Data Leakage)
# ============================================================
X = df.drop('salary', axis=1)
y = df['salary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
print(f"\nTrain set (avant IQR) : {X_train.shape[0]} exemples")

# ============================================================
# 3. Traitement des valeurs aberrantes (IQR) UNIQUEMENT sur TRAIN
# ============================================================
train_df = X_train.copy()
train_df['salary'] = y_train

numerical_cols = ['age', 'years_of_experience', 'salary']
for col in numerical_cols:
    Q1 = train_df[col].quantile(0.25)
    Q3 = train_df[col].quantile(0.75)
    IQR = Q3 - Q1
    train_df = train_df[~((train_df[col] < (Q1 - 1.5 * IQR)) | (train_df[col] > (Q3 + 1.5 * IQR)))]

X_train = train_df.drop('salary', axis=1)
y_train = train_df['salary']

print(f"Train set (après IQR) : {X_train.shape[0]} exemples")
print(f"Test set (inchangé)   : {X_test.shape[0]} exemples")

# ============================================================
# 4. Sauvegarde des options pour l'API (Frontend)
# ============================================================
# On sauvegarde les options uniques pour que le frontend puisse les afficher,
# sans que le backend FastAPI ait besoin de décortiquer le pipeline scikit-learn.
options = {
    "education_levels": ["High School", "Bachelor's", "Master's", "PhD"],
    "job_titles": sorted(list(df['job_title'].dropna().unique())),
    "genders": sorted(list(df['gender'].dropna().unique()))
}
with open(os.path.join(models_dir, 'options.json'), 'w') as f:
    json.dump(options, f)

print(f"\nOptions pour l'interface web sauvegardées dans models/options.json")


# ============================================================
# 5. Pipeline Scikit-Learn
# ============================================================
# L'OrdinalEncoder assignera -1 s'il voit une nouvelle catégorie (ex: nouveau métier)
# sans faire planter l'API en production.
edu_order = [options["education_levels"]]

preprocessor = ColumnTransformer(
    transformers=[
        ('edu', OrdinalEncoder(categories=edu_order, handle_unknown='use_encoded_value', unknown_value=-1), ['education_level']),
        ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), ['job_title', 'gender'])
    ],
    remainder='passthrough'
)

# Configuration de la GridSearch
model_params = {
    'Linear_Regression': {
        'model': LinearRegression(),
        'params': {}
    },
    'Decision_Tree': {
        'model': DecisionTreeRegressor(),
        'params': {
            'regressor__max_depth': [2, 4, 6, 8, 10],
            'regressor__random_state': [0, 42],
            'regressor__min_samples_split': [2, 5, 10, 20]
        }
    },
    'Random_Forest': {
        'model': RandomForestRegressor(random_state=42),
        'params': {
            'regressor__n_estimators': [10, 20, 30, 50, 80]
        }
    }
}

print("\n" + "="*60)
print("Optimisation des hyperparamètres (GridSearchCV, cv=3)")
print("="*60)

score = []
best_models = {}

for model_name, m in model_params.items():
    # Création du pipeline Complet : Preprocessing + Modèle
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', m['model'])
    ])
    
    clf = GridSearchCV(pipeline, m['params'], cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    
    print(f"Entraînement de {model_name}...")
    clf.fit(X_train, y_train)
    
    score.append({
        'Model': model_name,
        'Params': str(clf.best_params_),
        'MSE(-ve)': clf.best_score_
    })
    best_models[model_name] = clf.best_estimator_
    print(f"  -> best_score : {clf.best_score_:.2f}")

# ============================================================
# 6. Évaluation sur le jeu de test
# ============================================================
s = pd.DataFrame(score)
sort = s.sort_values(by='MSE(-ve)', ascending=False)
print(f"\nClassement :\n{sort[['Model', 'MSE(-ve)']].to_string(index=False)}")

model_eval = {}
for model_name, best_pipeline in best_models.items():
    y_pred = best_pipeline.predict(X_test)
    model_eval[model_name] = {
        "R2 Score": r2_score(y_test, y_pred),
        "Mean Squared Error": mean_squared_error(y_test, y_pred),
        "Mean Absolute Error": mean_absolute_error(y_test, y_pred),
        "Root Mean Squared Error": np.sqrt(mean_squared_error(y_test, y_pred))
    }

print("\n" + "="*60)
print("RÉSULTATS SUR LE JEU DE TEST")
print("="*60)

for model_name, metrics in model_eval.items():
    print(f"\n--- {model_name} ---")
    for metric_name, value in metrics.items():
        if "Error" in metric_name:
            print(f"  {metric_name}: {value:,.2f} USD")
        else:
            print(f"  {metric_name}: {value:.4f}")

# ============================================================
# 7. Sauvegarde du meilleur Pipeline (Random Forest ici)
# ============================================================
os.makedirs(os.path.join(models_dir, 'models'), exist_ok=True)
models_summary = pd.DataFrame(model_eval)
models_summary.to_excel(os.path.join(models_dir, 'models', 'models_evaluation.xlsx'))

# ON NE SAUVEGARDE QU'UN SEUL OBJET: Le Pipeline entier (qui contient les data transformers + model).
joblib.dump(best_models['Random_Forest'], os.path.join(models_dir, 'rfr_pipeline.pkl'))
print(f"\n✅ Pipeline Random Forest sauvegardé dans models/rfr_pipeline.pkl")
