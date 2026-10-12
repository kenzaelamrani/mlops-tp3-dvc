
import os
import yaml
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Charger les paramètres
with open("params.yaml", "r", encoding="utf-8") as f:
    params = yaml.safe_load(f)

config = params["entrainement"]

# Charger les données d'entraînement
train = pd.read_csv("data/prepare/train.csv")

# Séparer les variables explicatives et la cible
X_train = train.drop(columns=["fraude"])
y_train = train["fraude"]

# Créer le modèle Random Forest
model = RandomForestClassifier(
    n_estimators=config["n_estimators"],
    max_depth=config["max_depth"],
    random_state=config["random_state"]
)

# Entraîner le modèle
model.fit(X_train, y_train)

# Créer le dossier de sortie
os.makedirs("models", exist_ok=True)

# Enregistrer le modèle entraîné
joblib.dump(model, "models/model.pkl")

print("Entrainement termine : models/model.pkl cree")
