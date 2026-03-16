import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import joblib
import os

# Configuration du style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 11

# Dossier de sortie pour les figures
output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '')
print(f"Les figures seront sauvegardées dans : {output_dir}")

# ============================================================
# Chargement des données
# ============================================================
df_raw = pd.read_csv("data/processed/salary_dataset_processed.csv")
df_raw = df_raw.rename(columns={
    'Age': 'age', 'Gender': 'gender',
    'Education Level': 'education_level',
    'Job Title': 'job_title',
    'Years of Experience': 'years_of_experience',
    'Salary': 'salary'
})

# ============================================================
# Figure 1 : Distribution de la variable cible (Salary)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5.5))
sns.histplot(df_raw['salary'], bins=30, kde=True, color='#0046c8', edgecolor='white', alpha=0.7, ax=ax)
ax.axvline(df_raw['salary'].mean(), color='#00a0d2', linestyle='--', linewidth=2, label=f"Moyenne = {df_raw['salary'].mean():,.0f} USD")
ax.axvline(df_raw['salary'].median(), color='#e74c3c', linestyle='--', linewidth=2, label=f"Médiane = {df_raw['salary'].median():,.0f} USD")
ax.set_xlabel("Salaire annuel (USD)", fontsize=12)
ax.set_ylabel("Fréquence", fontsize=12)
ax.set_title("Distribution des salaires dans le jeu de données", fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "salary_distribution.png"), bbox_inches='tight')
plt.close()
print("✅ salary_distribution.png généré")

# ============================================================
# Figure 2 : Scatter plot Expérience vs Salaire
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5.5))
scatter = ax.scatter(
    df_raw['years_of_experience'], df_raw['salary'],
    c=df_raw['salary'], cmap='coolwarm', alpha=0.6, s=30, edgecolors='white', linewidth=0.3
)
# Droite de régression
z = np.polyfit(df_raw['years_of_experience'], df_raw['salary'], 1)
p = np.poly1d(z)
x_line = np.linspace(df_raw['years_of_experience'].min(), df_raw['years_of_experience'].max(), 100)
ax.plot(x_line, p(x_line), color='#e74c3c', linewidth=2.5, linestyle='-', label=f"Tendance linéaire")
ax.set_xlabel("Années d'expérience", fontsize=12)
ax.set_ylabel("Salaire annuel (USD)", fontsize=12)
ax.set_title("Relation entre l'expérience professionnelle et le salaire", fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
plt.colorbar(scatter, ax=ax, label="Salaire (USD)")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "scatter_exp_salary.png"), bbox_inches='tight')
plt.close()
print("✅ scatter_exp_salary.png généré")

# ============================================================
# Figure 3 : Matrice de corrélation
# ============================================================
fig, ax = plt.subplots(figsize=(8, 6))
numeric_df = df_raw[['age', 'years_of_experience', 'salary']]
corr = numeric_df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', mask=mask,
            square=True, linewidths=2, ax=ax,
            cbar_kws={"shrink": 0.8},
            annot_kws={"size": 14, "fontweight": "bold"})
ax.set_title("Matrice de corrélation des variables numériques", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "correlation_matrix.png"), bbox_inches='tight')
plt.close()
print("✅ correlation_matrix.png généré")

# ============================================================
# Figure 4 : Feature Importance (Random Forest Pipeline)
# ============================================================
pipeline = joblib.load("models/rfr_pipeline.pkl")
# Récupérer le modèle à l'intérieur du pipeline
rf_model = pipeline.named_steps['regressor']
preprocessor = pipeline.named_steps['preprocessor']

# Noms des features après ColumnTransformer (l'ordre des colonnes : 'education_level' puis 'job_title', 'gender' puis 'age', 'years_of_experience')
feature_names = ['education_level', 'job_title', 'gender', 'age', 'years_of_experience']
importances = rf_model.feature_importances_
indices = np.argsort(importances)

fig, ax = plt.subplots(figsize=(9, 5))
colors = sns.color_palette("coolwarm", len(feature_names))
bars = ax.barh(range(len(indices)), importances[indices], color=[colors[i] for i in range(len(indices))], edgecolor='white')
ax.set_yticks(range(len(indices)))
ax.set_yticklabels([feature_names[i] for i in indices], fontsize=11)
ax.set_xlabel("Importance", fontsize=12)
ax.set_title("Importance des variables — Random Forest (Pipeline)", fontsize=14, fontweight='bold')

# Ajouter les valeurs sur les barres
for bar, val in zip(bars, importances[indices]):
    ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2, 
            f'{val:.3f}', va='center', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "feature_importance.png"), bbox_inches='tight')
plt.close()
print("✅ feature_importance.png généré")

# ============================================================
# Figure 5 : Valeurs réelles vs prédites (Random Forest Pipeline)
# ============================================================
from sklearn.model_selection import train_test_split

df_clean = pd.read_csv("data/final/train_ready_dataset.csv")
X = df_clean.drop('salary', axis=1)
y = df_clean['salary']
# Même séparation que dans train.py sans modifier X_test
_, x_test, _, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Le pipeline s'occupe de l'encodage !
y_pred = pipeline.predict(x_test)

fig, ax = plt.subplots(figsize=(8, 7))
ax.scatter(y_test, y_pred, alpha=0.5, color='#0046c8', edgecolors='white', s=40, linewidth=0.5)

# Ligne diagonale parfaite
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
ax.plot(lims, lims, 'r--', linewidth=2, label='Prédiction parfaite')

ax.set_xlabel("Valeurs réelles (USD)", fontsize=12)
ax.set_ylabel("Valeurs prédites (USD)", fontsize=12)
ax.set_title("Valeurs réelles vs Valeurs prédites — Random Forest (Pipeline)", fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "pred_vs_actual.png"), bbox_inches='tight')
plt.close()
print("✅ pred_vs_actual.png généré")

print("\n🎉 Tous les graphiques ont été générés avec succès !")
print(f"Fichiers créés dans : {output_dir}")
