from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np
from typing import Dict, Any
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI(title="Api Predict Accidents", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionRequest(BaseModel):
    Accident_Date: str
    Day_of_Week: str
    Junction_Control: str
    Junction_Detail: str
    Light_Conditions: str
    Number_of_Casualties: int
    Road_Surface_Conditions: str
    Road_Type: str
    Speed_limit: int
    Time: str
    Urban_or_Rural_Area: str
    Weather_Conditions: str

class PredictionResponse(BaseModel):
    prediction: str
    probability: Dict[str, float]
    confidence: float

class AccidentPredictor:
    def __init__(self, model_path='model/accident_model.pkl', scaler_path='model/accident_scaler.pkl'):
        self.model_loaded = False

        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            print('Fichiers modèle non trouvés. Mode démo activé.')
            return
        try:
            self.model = joblib.load(model_path)
            saved_data = joblib.load(scaler_path)
            self.scaler = saved_data['scaler']
            self.label_encoder = saved_data['label_encoder']
            self.model_loaded = True
            print("Modèle chargé avec succès")
        except Exception as e:
            print(f"Erreur chargement modèle: {e}")

    def extract_features(self, input_data: dict) -> dict:
        time_str = input_data['Time']
        try:
            hour = int(time_str.split(':')[0])
        except:
            hour = 12

        date_str = input_data['Accident_Date']
        try:
            month = int(date_str.split('/')[1])
        except:
            month = 6

        features = input_data.copy()
        features['Hour'] = hour
        features['Month'] = month
        features['Is_Weekend'] = 1 if features['Day_of_Week'] in ['Sunday', 'Saturday'] else 0

        return features

    def preprocess(self, input_data: dict) -> pd.DataFrame:
        df = pd.DataFrame([input_data])

        if self.model_loaded:
            for col, encoder in self.label_encoder.items():
                if col in df.columns:
                    if input_data[col] in encoder.classes_:
                        df[col] = encoder.transform([input_data[col]])[0]
                    else:
                        df[col] = 0

            numerical_columns = df.select_dtypes(include=[np.number]).columns
            if not numerical_columns.empty:
                df[numerical_columns] = self.scaler.transform(df[numerical_columns])

        return df

    def predict(self, input_data: dict) -> Dict[str, Any]:
        if not self.model_loaded:
            # Mode démo
            return {
                'prediction': 'serious',
                'probability': {'light': 0.3, 'serious': 0.6, 'fatal': 0.1},
                'confidence': 0.6
            }

        try:
            processed_data = self.preprocess(input_data)
            probabilities = self.model.predict_proba(processed_data)[0]
            classes = self.model.classes_

            return {
                'prediction': classes[np.argmax(probabilities)],
                'probability': dict(zip(classes, probabilities)),
                'confidence': float(np.max(probabilities))
            }
        except Exception as e:
            raise ValueError(f"Erreur lors de la prédiction: {str(e)}")

predictor = AccidentPredictor()

@app.get("/")
async def root():
    return {"message": "API de prédiction de gravité d'accidents"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/predict")
async def prediction_accident(request: PredictionRequest):
    try:
        input_data = request.dict()
        result = predictor.predict(input_data)

        return PredictionResponse(
            prediction=result['prediction'],
            probability=result['probability'],
            confidence=result['confidence']
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))