"""
API FastAPI - Credit Risk Prediction
--------------------------------------
Chaque modele a son propre endpoint versionne.

Structure des fichiers :


Lancement :
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

Documentation interactive :
    http://localhost:8000/docs
"""

import pickle
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import uvicorn


# --------------------------------------------------
# Dossier contenant les modeles
# --------------------------------------------------
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")


# --------------------------------------------------
# Chargement des modeles au demarrage
# --------------------------------------------------
def charger_modele(nom_fichier: str):
    chemin = os.path.join(MODELS_DIR, nom_fichier)
    if not os.path.exists(chemin):
        raise RuntimeError(
            f"Modele introuvable : {chemin}\n"
            f"Placez le fichier {nom_fichier} dans le dossier models/"
        )
    with open(chemin, "rb") as f:
        return pickle.load(f)


try:
    modele_xgboost = charger_modele("xgboost_v1.pkl")
    print("XGBoost v1 charge.")
except RuntimeError as e:
    print(f"ATTENTION : {e}")
    modele_xgboost = None

try:
    modele_rf = charger_modele("random_forest_v1.pkl")
    print("Random Forest v1 charge.")
except RuntimeError as e:
    print(f"ATTENTION : {e}")
    modele_rf = None


# --------------------------------------------------
# Registre des modeles disponibles
# --------------------------------------------------
MODELES = {
    "xgboost": {
        "version": "v1",
        "fichier": "xgboost_v1.pkl",
        "pipeline": modele_xgboost,
        "description": "XGBoost Classifier avec scale_pos_weight"
    },
    "random_forest": {
        "version": "v1",
        "fichier": "random_forest_v1.pkl",
        "pipeline": modele_rf,
        "description": "Random Forest Classifier avec class_weight balanced"
    }
}


# --------------------------------------------------
# Schemas Pydantic
# --------------------------------------------------
class ClientData(BaseModel):
    person_age: int = Field(..., ge=18, le=100, description="Age du client (18-100 ans)")
    person_income: float = Field(..., gt=0, description="Revenu annuel en dollars")
    person_home_ownership: str = Field(..., description="RENT | OWN | MORTGAGE | OTHER")
    person_emp_length: float = Field(..., ge=0, le=60, description="Anciennete emploi en annees")
    loan_intent: str = Field(..., description="PERSONAL | EDUCATION | MEDICAL | VENTURE | HOMEIMPROVEMENT | DEBTCONSOLIDATION")
    loan_grade: str = Field(..., description="A | B | C | D | E | F | G")
    loan_amnt: float = Field(..., gt=0, description="Montant du pret en dollars")
    loan_int_rate: float = Field(..., gt=0, description="Taux d'interet du pret (%)")
    loan_percent_income: float = Field(..., ge=0, le=1, description="Part du pret dans le revenu")
    cb_person_default_on_file: str = Field(..., description="Antecedent de defaut : Y | N")
    cb_person_cred_hist_length: int = Field(..., ge=0, description="Duree historique credit (annees)")

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


class PredictionResponse(BaseModel):
    modele: str
    version: str
    prediction: int
    prediction_label: str
    probabilite_defaut: float
    niveau_risque: str


# --------------------------------------------------
# Application FastAPI
# --------------------------------------------------
app = FastAPI(
    title="Credit Risk Prediction API",
    description=(
        "API de prediction du risque de defaut sur un pret.\n\n"
        "Deux modeles disponibles, chacun accessible via son propre endpoint :\n\n"
        "- `/predict/xgboost` : XGBoost v1\n"
        "- `/predict/random_forest` : Random Forest v1\n\n"
        "Chaque modele est versionne et stocke separement dans le dossier `models/`."
    ),
    version="1.0.0",
)


# --------------------------------------------------
# Fonction utilitaire
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


def faire_prediction(nom_modele: str, client: ClientData) -> PredictionResponse:
    """Logique commune de prediction pour tous les modeles."""
    if nom_modele not in MODELES:
        raise HTTPException(
            status_code=404,
            detail=f"Modele '{nom_modele}' inconnu. Modeles disponibles : {list(MODELES.keys())}"
        )

    info = MODELES[nom_modele]

    if info["pipeline"] is None:
        raise HTTPException(
            status_code=503,
            detail=f"Modele '{nom_modele}' non disponible. Verifiez que {info['fichier']} est present dans models/"
        )

    try:
        donnees = pd.DataFrame([client.model_dump()])
        prediction = int(info["pipeline"].predict(donnees)[0])
        probabilite = float(info["pipeline"].predict_proba(donnees)[0][1])

        return PredictionResponse(
            modele=nom_modele,
            version=info["version"],
            prediction=prediction,
            prediction_label="DEFAUT" if prediction == 1 else "NON-DEFAUT",
            probabilite_defaut=round(probabilite, 4),
            niveau_risque=get_niveau_risque(probabilite)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de prediction : {str(e)}")


# --------------------------------------------------
# Routes generales
# --------------------------------------------------
@app.get("/", summary="Statut de l'API")
def root():
    """Retourne le statut de l'API et la liste des modeles disponibles."""
    return {
        "status": "en ligne",
        "version_api": "1.0.0",
        "modeles_disponibles": {
            nom: {
                "version": info["version"],
                "description": info["description"],
                "disponible": info["pipeline"] is not None,
                "endpoint": f"/predict/{nom}"
            }
            for nom, info in MODELES.items()
        }
    }


@app.get("/models", summary="Liste des modeles")
def liste_modeles():
    """Retourne la liste de tous les modeles avec leur statut."""
    return {
        nom: {
            "version": info["version"],
            "fichier": info["fichier"],
            "description": info["description"],
            "disponible": info["pipeline"] is not None
        }
        for nom, info in MODELES.items()
    }


# --------------------------------------------------
# Routes de prediction par modele
# --------------------------------------------------
@app.post(
    "/predict/xgboost",
    response_model=PredictionResponse,
    summary="Prediction avec XGBoost v1",
    tags=["XGBoost"]
)
def predict_xgboost(client: ClientData):
    """
    Prediction du risque de defaut avec le modele XGBoost v1.

    XGBoost est un algorithme de boosting. Il construit des arbres de decision
    en sequence, chacun corrigeant les erreurs du precedent.
    """
    return faire_prediction("xgboost", client)


@app.post(
    "/predict/random_forest",
    response_model=PredictionResponse,
    summary="Prediction avec Random Forest v1",
    tags=["Random Forest"]
)
def predict_random_forest(client: ClientData):
    """
    Prediction du risque de defaut avec le modele Random Forest v1.

    Random Forest est un algorithme de bagging. Il entraine plusieurs arbres
    en parallele sur des sous-echantillons differents et fait la moyenne des resultats.
    """
    return faire_prediction("random_forest", client)


@app.post(
    "/predict/batch/{nom_modele}",
    summary="Prediction en lot pour un modele donne",
    tags=["Batch"]
)
def predict_batch(nom_modele: str, clients: list[ClientData]):
    """
    Prediction pour plusieurs clients avec le modele choisi.

    Remplacer {nom_modele} par : xgboost ou random_forest
    """
    if len(clients) == 0:
        raise HTTPException(status_code=400, detail="La liste de clients est vide.")
    if len(clients) > 1000:
        raise HTTPException(status_code=400, detail="Maximum 1000 clients par requete.")

    if nom_modele not in MODELES:
        raise HTTPException(
            status_code=404,
            detail=f"Modele '{nom_modele}' inconnu. Disponibles : {list(MODELES.keys())}"
        )

    info = MODELES[nom_modele]
    if info["pipeline"] is None:
        raise HTTPException(status_code=503, detail=f"Modele '{nom_modele}' non disponible.")

    try:
        donnees = pd.DataFrame([c.model_dump() for c in clients])
        predictions = info["pipeline"].predict(donnees).tolist()
        probabilites = info["pipeline"].predict_proba(donnees)[:, 1].tolist()

        resultats = []
        for pred, proba in zip(predictions, probabilites):
            resultats.append({
                "prediction": int(pred),
                "prediction_label": "DEFAUT" if pred == 1 else "NON-DEFAUT",
                "probabilite_defaut": round(float(proba), 4),
                "niveau_risque": get_niveau_risque(float(proba))
            })

        return {
            "modele": nom_modele,
            "version": info["version"],
            "nombre_clients": len(clients),
            "predictions": resultats
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur : {str(e)}")


# --------------------------------------------------
# Lancement direct
# --------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)