# Credit Risk Prediction API

API de prediction du risque de defaut sur un pret bancaire.
Modele : **XGBoost** entraine sur le dataset Credit Risk (Kaggle).

---

## Structure du dossier

```
api/
├── main.py            # Serveur FastAPI
├── model.pkl          # Pipeline XGBoost complet (genere par le notebook)
├── requirements.txt   # Dependances Python
└── README.md          # Ce fichier
```

---

## Installation et lancement

### 1. Installer les dependances

```bash
pip install -r requirements.txt
```

### 2. Placer model.pkl dans ce dossier

Le fichier `model.pkl` est genere a l'etape 42 du notebook `eda_credit_risk.ipynb`.
Il doit etre dans le meme dossier que `main.py`.

### 3. Lancer le serveur

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Acceder a la documentation interactive

```
http://localhost:8000/docs
```

---

## Endpoints

### GET /

Statut de l'API.

```bash
curl http://localhost:8000/
```

Reponse :
```json
{
  "status": "en ligne",
  "modele": "XGBoost Credit Risk",
  "version": "1.0.0"
}
```

---

### POST /predict

Prediction du risque de defaut pour un seul client.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

Reponse :
```json
{
  "prediction": 0,
  "prediction_label": "NON-DEFAUT",
  "probabilite_defaut": 0.0821,
  "niveau_risque": "FAIBLE"
}
```

---

### POST /predict/batch

Prediction pour plusieurs clients en une seule requete (max 1000).

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '[
    {
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
    },
    {
      "person_age": 22,
      "person_income": 15000,
      "person_home_ownership": "RENT",
      "person_emp_length": 1.0,
      "loan_intent": "VENTURE",
      "loan_grade": "F",
      "loan_amnt": 20000,
      "loan_int_rate": 19.5,
      "loan_percent_income": 0.65,
      "cb_person_default_on_file": "Y",
      "cb_person_cred_hist_length": 2
    }
  ]'
```

Reponse :
```json
{
  "nombre_clients": 2,
  "predictions": [
    {
      "prediction": 0,
      "prediction_label": "NON-DEFAUT",
      "probabilite_defaut": 0.0821,
      "niveau_risque": "FAIBLE"
    },
    {
      "prediction": 1,
      "prediction_label": "DEFAUT",
      "probabilite_defaut": 0.8943,
      "niveau_risque": "TRES ELEVE"
    }
  ]
}
```

---

## Niveaux de risque

| Probabilite de defaut | Niveau de risque |
|---|---|
| Inferieure a 20% | FAIBLE |
| Entre 20% et 40% | MODERE |
| Entre 40% et 65% | ELEVE |
| Superieure a 65% | TRES ELEVE |

---

## Variables attendues

| Variable | Type | Valeurs acceptees |
|---|---|---|
| person_age | int | 18 a 100 |
| person_income | float | superieur a 0 |
| person_home_ownership | str | RENT, OWN, MORTGAGE, OTHER |
| person_emp_length | float | 0 a 60 |
| loan_intent | str | PERSONAL, EDUCATION, MEDICAL, VENTURE, HOMEIMPROVEMENT, DEBTCONSOLIDATION |
| loan_grade | str | A, B, C, D, E, F, G |
| loan_amnt | float | superieur a 0 |
| loan_int_rate | float | superieur a 0 |
| loan_percent_income | float | 0 a 1 |
| cb_person_default_on_file | str | Y, N |
| cb_person_cred_hist_length | int | superieur ou egal a 0 |

---

## Modele

- **Algorithme** : XGBoost Classifier
- **Gestion du desequilibre** : scale_pos_weight = 3.58
- **Pipeline** : imputation mediane + encodage + StandardScaler + XGBoost
- **Metrique principale** : ROC-AUC
