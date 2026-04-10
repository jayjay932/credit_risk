# Credit Risk Prediction

Projet de data science appliqué à la prédiction du risque de défaut bancaire. L'objectif est de prédire si un client va rembourser son prêt ou non, à partir de son profil financier et personnel, et d'exposer ce modèle via une API versionnée.

---

## Contexte métier

Chaque prêt accordé par une banque représente un risque financier. Si un client ne rembourse pas, la banque enregistre une perte sèche. Un modèle de scoring crédit permet d'objectiver la décision d'octroi en attribuant à chaque demande une probabilité de défaut. Cette probabilité peut ensuite servir à accepter, refuser ou adapter les conditions du prêt.

Le dataset utilisé est le [Credit Risk Dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset) disponible sur Kaggle. Il contient 32 581 demandes de prêt avec 11 variables décrivant le profil du client et les caractéristiques du prêt. La variable cible est `loan_status` : 0 signifie que le client a remboursé, 1 signifie un défaut de paiement.

---

## Structure du projet

```
credit-risk/
├── eda_credit_risk.ipynb         # Notebook : EDA, preprocessing, modèles
├── api/
│   ├── main.py                   # Serveur FastAPI
│   ├── requirements.txt
│   ├── README.md
│   └── models/
│       ├── xgboost_v1.pkl
│       └── random_forest_v1.pkl
└── README.md
```

---

## Analyse exploratoire

La première étape a consisté à comprendre les données avant de toucher au moindre modèle.

La cible est déséquilibrée : 78.2% de non-défauts contre 21.8% de défauts, soit un ratio de 3.58. Ce déséquilibre est suffisant pour rendre l'accuracy inutile comme métrique — un modèle qui prédirait systématiquement 0 obtiendrait déjà 78% sans avoir rien appris. Cela oriente directement vers le ROC-AUC et le recall comme métriques principales.

L'analyse des distributions révèle deux problèmes structurels dans les données. `person_age` atteint un maximum de 144 ans et `person_emp_length` un maximum de 123 ans — des valeurs biologiquement impossibles qui trahissent des erreurs de saisie. `person_income` présente un skewness de 32.87, une asymétrie extrême liée à quelques revenus très élevés qui tirent la moyenne vers le haut.

Deux colonnes ont des valeurs manquantes : `person_emp_length` à 2.75% et `loan_int_rate` à 9.56%. Le reste du dataset est complet.

L'analyse des corrélations met en évidence deux relations fortes. `person_age` et `cb_person_cred_hist_length` sont corrélées à 0.86, ce qui est logique : l'historique de crédit s'accumule avec l'âge. `loan_amnt` et `loan_percent_income` sont corrélées à 0.57 car le montant du prêt représente mécaniquement une part du revenu. Ces corrélations sont surveillées mais ne justifient pas de supprimer des variables — XGBoost gère bien la colinéarité modérée.

L'analyse croisée avec la cible révèle que le grade du prêt est le signal le plus discriminant : le taux de défaut passe de 5% en grade A à plus de 50% en grade G, soit 45 points d'écart. Le statut de logement est également très informatif — les locataires défaillent à 31.6% contre 7.5% pour les propriétaires. Ces deux variables seront naturellement les plus importantes pour le modèle.

---

## Nettoyage et preprocessing

### Nettoyage des outliers

Les lignes avec `person_age > 100` et `person_emp_length > 60` sont supprimées. Ces valeurs sont biologiquement impossibles et constituent des erreurs de saisie, pas des observations rares mais légitimes. On passe de 32 581 à 31 679 lignes après nettoyage.

`person_income` est plafonné au 99e percentile via `clip()`. On choisit le capping plutôt que la suppression pour ne pas perdre les observations associées — seule la valeur extrême du revenu est corrigée, pas la ligne entière.

### Gestion des valeurs manquantes

Les deux colonnes avec des manquants sont imputées par la médiane. On choisit la médiane plutôt que la moyenne car les deux distributions sont asymétriques — la moyenne est tirée vers le haut par les valeurs extrêmes et ne représente pas bien le centre de la distribution. La médiane est insensible à ces extrêmes.

L'imputation est appliquée uniquement sur le jeu d'entraînement via un `SimpleImputer` intégré au pipeline. La valeur de médiane apprise sur le train est ensuite appliquée sur le test — on ne recalcule jamais la médiane sur le test pour éviter toute contamination.

### Encodage des variables catégorielles

`loan_grade` est encodé en ordinal avec l'ordre A=0, B=1, ..., G=6. Ce choix est justifié par le sens métier : les grades ont une progression logique de risque et un encodage ordinal permet au modèle de capturer cette relation d'ordre.

`person_home_ownership` et `loan_intent` sont encodés en one-hot. Ces variables n'ont pas d'ordre naturel — un encodage ordinal leur attribuerait une relation numérique arbitraire qui n'existe pas dans la réalité.

`cb_person_default_on_file` est encodé en binaire : N=0, Y=1.

### Normalisation

Un `StandardScaler` est appliqué sur toutes les variables numériques. XGBoost n'en a pas strictement besoin car il est invariant aux transformations monotones, mais le pipeline est conçu pour être réutilisable avec d'autres algorithmes comme la régression logistique ou le SVM qui, eux, y sont sensibles.

### Ordre du pipeline

Le split train/test est effectué avant toute transformation. Le pipeline sklearn encapsule imputation, encodage et normalisation dans un `ColumnTransformer` qui s'entraîne uniquement sur X_train et s'applique ensuite sur X_test. Cette architecture garantit qu'aucune statistique du jeu de test ne contamine l'entraînement.

---

## Modèles supervisés

### Choix des métriques

L'accuracy est exclue à cause du déséquilibre de classes. On retient le ROC-AUC comme métrique principale car il mesure la capacité de discrimination du modèle sur l'ensemble des seuils possibles, indépendamment du seuil de décision choisi. Le recall sur la classe défaut est la deuxième métrique clé : dans un contexte crédit, manquer un vrai défaut a un coût financier direct pour la banque, supérieur au coût d'un faux positif.

### XGBoost

XGBoost construit des arbres de décision en séquence, chaque arbre corrigeant les erreurs du précédent. Le paramètre `scale_pos_weight` est défini à 3.64 (ratio négatifs/positifs) pour pénaliser davantage les erreurs sur la classe minoritaire. 300 estimateurs avec un learning rate de 0.05 favorisent une convergence stable, le subsampling à 0.8 réduit l'overfitting.

Résultats sur le jeu de test :

- ROC-AUC : 0.9514
- F1-score : 0.8231
- Recall : 0.8029
- Précision : 0.8444

Sur les 1 365 vrais défauts du jeu de test, le modèle en détecte 1 096, soit 80%. Il génère 202 faux positifs. La validation croisée en 5 folds donne un ROC-AUC moyen de 0.9482 avec un écart-type de 0.0044, ce qui confirme que ces résultats sont stables et ne dépendent pas du split particulier.

### Random Forest

Random Forest entraîne 300 arbres en parallèle sur des sous-échantillons aléatoires et fait la moyenne de leurs prédictions. `class_weight='balanced'` ajuste automatiquement les poids inversement proportionnels aux fréquences de classe. `min_samples_leaf=5` évite les feuilles trop spécifiques qui mémorisent le bruit.

Résultats sur le jeu de test :

- ROC-AUC : 0.9383
- F1-score : 0.8272
- Recall : 0.7487
- Précision : 0.9241

### Comparaison et choix final

XGBoost obtient un ROC-AUC supérieur de 0.013 et un recall supérieur de 5.4 points. Random Forest a une meilleure précision de 8 points — il génère moins de faux positifs (84 contre 202) mais rate davantage de vrais défauts (343 contre 269).

Le choix dépend du coût relatif des deux types d'erreurs dans le contexte métier. En scoring crédit, un faux négatif (défaut non détecté) représente une perte directe pour la banque. Un faux positif (bon client refusé) représente un manque à gagner mais pas une perte. Ce raisonnement penche en faveur de XGBoost qui maximise le recall. XGBoost est donc retenu comme modèle de production.

La feature la plus importante est `loan_grade_encoded` avec un gain de 0.20, suivi de `loan_percent_income` et `loan_int_rate`. Ces résultats sont cohérents avec l'analyse exploratoire qui identifiait le grade comme le signal le plus discriminant.

---

## Modèles non supervisés

### KMeans

KMeans est appliqué sur les 7 variables numériques après normalisation. Le nombre de clusters optimal est déterminé en testant k de 2 à 8 et en retenant le k qui maximise le silhouette score. k=2 est optimal avec un silhouette de 0.2618.

Le cluster 0 regroupe des emprunteurs jeunes (25 ans en moyenne, 57 000$ de revenu, 3.9 ans d'ancienneté) avec un taux de défaut de 23%. Le cluster 1 regroupe des profils plus stables (35 ans, 87 000$, 7.3 ans d'ancienneté) avec un taux de défaut de 17%. L'écart de 6 points montre que la segmentation non supervisée capture une partie du signal de risque sans avoir accès à la variable cible.

### DBSCAN

DBSCAN est appliqué pour détecter les profils atypiques. Le paramètre eps est estimé via la courbe k-distance, qui montre le coude vers 1.5. Avec eps=1.5 et min_samples=10, DBSCAN isole 228 points de bruit soit 2.3% de l'échantillon.

Ces anomalies sont en moyenne plus âgées (45 ans contre 27), ont un revenu plus élevé (116 000$ contre 64 000$) et empruntent des montants plus importants. Leur taux de défaut est de 26% contre 22% pour les clients normaux. DBSCAN a donc capturé une légère surreprésentation du risque sans aucune supervision.

---

## API

### Lancement

```bash
pip install -r api/requirements.txt
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Documentation interactive : `http://localhost:8000/docs`

### Versioning des modèles

Chaque modèle est stocké dans `api/models/` avec son nom et sa version dans le nom de fichier. L'API charge tous les modèles au démarrage et les expose sur des endpoints distincts. Pour déployer une nouvelle version, il suffit de sauvegarder le pipeline sous `xgboost_v2.pkl` et d'ajouter une entrée dans le registre `MODELES` dans `main.py`. L'ancienne version reste disponible sans interruption.

### Endpoints

`POST /predict/xgboost` — prédiction avec XGBoost v1.

`POST /predict/random_forest` — prédiction avec Random Forest v1.

`POST /predict/batch/{nom_modele}` — prédictions en lot pour les deux modèles.

### Exemple

```bash
curl -X POST http://localhost:8000/predict/xgboost \
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

```json
{
  "modele": "xgboost",
  "version": "v1",
  "prediction": 0,
  "prediction_label": "NON-DEFAUT",
  "probabilite_defaut": 0.0821,
  "niveau_risque": "FAIBLE"
}
```

Le champ `niveau_risque` prend quatre valeurs selon la probabilité : FAIBLE sous 20%, MODERE entre 20% et 40%, ELEVE entre 40% et 65%, TRES ELEVE au-delà.

---

## Dépendances

Python 3.10+, pandas, numpy, scikit-learn, xgboost, fastapi, uvicorn, pydantic, matplotlib, seaborn.
