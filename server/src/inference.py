import joblib
import pandas as pd
import numpy as np
from typing import Union, List, Dict
from pathlib import Path


import os

# Resolution absolue dynamique depuis server/src/inference.py -> server/models/...
DEFAULT_PIPELINE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "rfr_pipeline.pkl")

def predict_salaries(
    data: Union[str, List[Dict], Dict],
    output_csv_path: str = None,
    pipeline_path: str = DEFAULT_PIPELINE_PATH
) -> Dict:
    
    # Load pipeline
    print("Chargement du pipeline ML unifié...")
    pipeline = joblib.load(pipeline_path)
    
    # Load data based on type
    if isinstance(data, str):
        # If string, treat as file path
        print(f"Chargement des données depuis {data}...")
        df = pd.read_json(data)
    elif isinstance(data, dict):
        # Single prediction - convert to DataFrame
        print("Traitement d'une prédiction unique...")
        df = pd.DataFrame([data])
    elif isinstance(data, list):
        # Multiple predictions
        print(f"Traitement de {len(data)} prédictions...")
        df = pd.DataFrame(data)
    else:
        raise ValueError("data doit être un chemin JSON, un dict ou une liste de dicts")
    
    # Le prétraitement (ColumnTransformer) est intégré au pipeline
    print("Prétraitement et prédictions en cours via le pipeline...")
    predictions = pipeline.predict(df)
    
    # Add predictions to original dataframe
    df['predicted_salary'] = predictions
    
    # Save to CSV if requested
    if output_csv_path:
        df.to_csv(output_csv_path, index=False)
        print(f"Prédictions sauvegardées dans {output_csv_path}")
    
    # Prepare response
    response = {
        "status": "success",
        "count": len(df),
        "predictions": df.to_dict(orient='records')
    }
    
    print(f"\n✅ Prédictions complétées pour {len(df)} enregistrement(s)")
    print(f"\nAperçu des prédictions:")
    print(df.head())
    
    return response


# Utilisation standalone
if __name__ == "__main__":
    
    # Test avec un dict unique
    single_input = {
        "age": 46,
        "gender": "Male",
        "education_level": "Bachelor's",
        "job_title": "Software Engineer",
        "years_of_experience": 5
    }
    results = predict_salaries(single_input)
    print(f"\nRésultat: {results['predictions'][0]['predicted_salary']}")