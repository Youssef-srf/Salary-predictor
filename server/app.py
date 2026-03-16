from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import json
import os

app = FastAPI(title="Salary Prediction API")

# ============================================================
# Chemins d'accès absolus sécurisés
# ============================================================
base_dir = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(base_dir, "static")
pipeline_path = os.path.join(base_dir, "models", "rfr_pipeline.pkl")
options_path = os.path.join(base_dir, "models", "options.json")

app.mount("/static", StaticFiles(directory=static_dir), name="static")

# ============================================================
# Chargement du Pipeline ML et des Options
# ============================================================
# Plus besoin de charger 4 encodeurs séparément, le Pipeline gère tout !
try:
    model_pipeline = joblib.load(pipeline_path)
    with open(options_path, "r", encoding='utf-8') as f:
        options_data = json.load(f)
except FileNotFoundError as e:
    print(f"ATTENTION: Fichier manquant. Veuillez exécuter 'src/train.py' d'abord. Erreur : {e}")
    model_pipeline = None
    options_data = {}

# ============================================================
# Modèles Pydantic pour la validation
# ============================================================
class PredictionInput(BaseModel):
    age: int
    gender: str
    education_level: str
    job_title: str
    years_of_experience: float

class PredictionOutput(BaseModel):
    predicted_salary: float

# ============================================================
# Routes API
# ============================================================
@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    """Prédit le salaire en passant les données brutes directement au pipeline."""
    if not model_pipeline:
        raise HTTPException(status_code=500, detail="Modèle non chargé sur le serveur.")

    try:
        # Construction du DataFrame avec les features brutes
        # L'encodage (OrdinalEncoder) est fait automatiquement par scikit-learn
        df = pd.DataFrame([{
            'age': input_data.age,
            'gender': input_data.gender,
            'education_level': input_data.education_level,
            'job_title': input_data.job_title,
            'years_of_experience': input_data.years_of_experience,
        }])
        
        # Prédiction (le pipeline s'occupe de tout le reste)
        prediction = model_pipeline.predict(df)[0]
        
        return PredictionOutput(predicted_salary=round(float(prediction), 2))
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur interne lors de la prédiction : {str(e)}")

@app.get("/options")
async def get_options():
    """Renvoie les listes des valeurs acceptées (chargées depuis le fichier json)."""
    return options_data

@app.get("/")
async def root():
    """Sert l'interface web frontend."""
    return FileResponse(os.path.join(static_dir, "index.html"))

# Pour lancer: uvicorn app:app --reload