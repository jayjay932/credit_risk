"""
API FastAPI - Credit Risk Prediction
-------------------------------------
Endpoint principal : POST /predict
Charge le pipeline XGBoost complet (preprocessing + modele)
et retourne la probabilite de defaut pour un client.

Lancement :
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

Documentation interactive automatique :
    http://localhost:8000/docs
"""

import pickle
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import uvicorn


# --------------------------------------------------
# Chargement du modele au demarrage du serveur
# --------------------------------------------------
CHEMIN_MODELE = os.path.join(os.path.dirname(__file__), "model.pkl")

try:
    with open(CHEMIN_MODELE, "rb") as f:
        pipeline = pickle.load(f)
    print(f"Modele charge depuis : {CHEMIN_MODELE}")
except FileNotFoundError:
    raise RuntimeError(
        f"Fichier model.pkl introuvable : {CHEMIN_MODELE}\n"
        "Placez model.pkl dans le meme dossier que main.py."
    )


# --------------------------------------------------
# Schema de la requete (validation automatique par Pydantic)
# --------------------------------------------------
class ClientData(BaseModel):
    person_age: int = Field(..., ge=18, le=100, description="Age du client (18-100 ans)")
    person_income: float = Field(..., gt=0, description="Revenu annuel en dollars")
    person_home_ownership: str = Field(..., description="RENT | OWN | MORTGAGE | OTHER")
    person_emp_length: float = Field(..., ge=0, le=60, description="Anciennete emploi en annees (0-60)")
    loan_intent: str = Field(..., description="PERSONAL | EDUCATION | MEDICAL | VENTURE | HOMEIMPROVEMENT | DEBTCONSOLIDATION")
    loan_grade: str = Field(..., description="Grade du pret : A | B | C | D | E | F | G")
    loan_amnt: float = Field(..., gt=0, description="Montant du pret en dollars")
    loan_int_rate: float = Field(..., gt=0, description="Taux d'interet du pret (%)")
    loan_percent_income: float = Field(..., ge=0, le=1, description="Part du pret dans le revenu (0 a 1)")
    cb_person_default_on_file: str = Field(..., description="Antecedent de defaut : Y | N")
    cb_person_cred_hist_length: int = Field(..., ge=0, description="Duree historique credit (en annees)")

    class Config:
        json_schema_extra = {
            "example": {
                "person_age": 30,
                "person_income": 55000,
                "person_home_ownership": "RENT",
                "person_emp_length": 5.0,
                "loan_intent": "PERSONAL",
                "loan_grade": "B",
                "loan_amnt": 10000,
                "loan_int_rate": 11.5,
                "loan_percent_income": 0.18,
                "cb_person_default_on_file": "N",
                "cb_person_cred_hist_length": 4
            }
        }


# --------------------------------------------------
# Schema de la reponse
# --------------------------------------------------
class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="0 = Non-defaut | 1 = Defaut")
    prediction_label: str = Field(..., description="NON-DEFAUT ou DEFAUT")
    probabilite_defaut: float = Field(..., description="Probabilite estimee de defaut (0 a 1)")
    niveau_risque: str = Field(..., description="FAIBLE | MODERE | ELEVE | TRES ELEVE")


# --------------------------------------------------
# Application FastAPI
# --------------------------------------------------
app = FastAPI(
    title="Credit Risk Prediction API",
    description=(
        "API de prediction du risque de defaut sur un pret bancaire.\n\n"
        "Modele : XGBoost entraine sur le dataset Credit Risk (Kaggle).\n"
        "Metrique principale : ROC-AUC.\n\n"
        "Envoyer les donnees brutes d'un client via POST /predict."
    ),
    version="1.0.0",
)


# --------------------------------------------------
# Fonction utilitaire : niveau de risque metier
# --------------------------------------------------
def get_niveau_risque(proba: float) -> str:
    if proba < 0.20:
        return "FAIBLE"
    elif proba < 0.40:
        return "MODERE"
    elif proba < 0.65:
        return "ELEVE"
    else:
        return "TRES ELEVE"


# --------------------------------------------------
# Routes
# --------------------------------------------------
@app.get("/", summary="Statut de l'API")
def root():
    """Verification que l'API est en ligne."""
    return {
        "status": "en ligne",
        "modele": "XGBoost Credit Risk",
        "version": "1.0.0",
        "endpoints": {
            "prediction_unitaire": "POST /predict",
            "prediction_batch":    "POST /predict/batch",
            "documentation":       "GET /docs",
        }
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predire le risque de defaut pour un client",
    description=(
        "Envoyer les donnees d'un client et recevoir :\n"
        "- La prediction binaire (0 = non-defaut, 1 = defaut)\n"
        "- La probabilite de defaut entre 0 et 1\n"
        "- Le niveau de risque : FAIBLE / MODERE / ELEVE / TRES ELEVE"
    )
)
def predict(client: ClientData):
    """
    Prediction du risque de defaut pour un client.

    - **prediction** : 0 = non-defaut, 1 = defaut
    - **probabilite_defaut** : score entre 0 et 1
    - **niveau_risque** : interpretation metier du score
    """
    try:
        # Convertir la requete en DataFrame (format attendu par le pipeline sklearn)
        donnees = pd.DataFrame([client.model_dump()])

        # Prediction via le pipeline complet (preprocessing + XGBoost)
        prediction  = int(pipeline.predict(donnees)[0])
        probabilite = float(pipeline.predict_proba(donnees)[0][1])

        return PredictionResponse(
            prediction=prediction,
            prediction_label="DEFAUT" if prediction == 1 else "NON-DEFAUT",
            probabilite_defaut=round(probabilite, 4),
            niveau_risque=get_niveau_risque(probabilite)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de prediction : {str(e)}")


@app.post(
    "/predict/batch",
    summary="Prediction en lot (plusieurs clients)",
    description="Envoyer une liste de clients et recevoir toutes les predictions en une seule requete (max 1000)."
)
def predict_batch(clients: list[ClientData]):
    """Prediction pour une liste de clients (maximum 1000)."""
    if len(clients) == 0:
        raise HTTPException(status_code=400, detail="La liste de clients est vide.")
    if len(clients) > 1000:
        raise HTTPException(status_code=400, detail="Maximum 1000 clients par requete batch.")

    try:
        donnees      = pd.DataFrame([c.model_dump() for c in clients])
        predictions  = pipeline.predict(donnees).tolist()
        probabilites = pipeline.predict_proba(donnees)[:, 1].tolist()

        resultats = []
        for pred, proba in zip(predictions, probabilites):
            resultats.append({
                "prediction":         int(pred),
                "prediction_label":   "DEFAUT" if pred == 1 else "NON-DEFAUT",
                "probabilite_defaut": round(float(proba), 4),
                "niveau_risque":      get_niveau_risque(float(proba))
            })

        return {
            "nombre_clients": len(clients),
            "predictions":    resultats
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de prediction batch : {str(e)}")


# --------------------------------------------------
# Lancement direct
# --------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
