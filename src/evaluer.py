
import os
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    ConfusionMatrixDisplay
)

# Charger les données de test
test = pd.read_csv("data/prepare/test.csv")

X_test = test.drop(columns=["fraude"])
y_test = test["fraude"]

# Charger le modèle entraîné
model = joblib.load("models/model.pkl")

# Faire des prédictions
y_pred = model.predict(X_test)

# Calculer les métriques
metrics = {
    "f1": round(f1_score(y_test, y_pred), 4),
    "precision": round(precision_score(y_test, y_pred), 4),
    "recall": round(recall_score(y_test, y_pred), 4)
}

# Créer le dossier de résultats
os.makedirs("reports", exist_ok=True)

# Enregistrer les métriques
with open("reports/metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=4)

# Générer la matrice de confusion
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred
)

plt.title("Matrice de confusion")
plt.savefig("reports/confusion.png")
plt.close()

print("Evaluation :", metrics)
