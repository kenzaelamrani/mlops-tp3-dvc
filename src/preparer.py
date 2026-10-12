import os
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split

# Charger les paramètres
with open("params.yaml", "r", encoding="utf-8") as f:
    params = yaml.safe_load(f)

test_size = params["preparation"]["test_size"]
random_state = params["preparation"]["random_state"]

# Charger les transactions
df = pd.read_csv("data/transactions.csv")

# Séparer les données en entraînement et test
train, test = train_test_split(
    df,
    test_size=test_size,
    random_state=random_state,
    stratify=df["fraude"]
)

# Créer le dossier de sortie
os.makedirs("data/prepare", exist_ok=True)

# Enregistrer les données préparées
train.to_csv("data/prepare/train.csv", index=False)
test.to_csv("data/prepare/test.csv", index=False)

print(f"Preparation : {len(train)} lignes train / {len(test)} lignes test")