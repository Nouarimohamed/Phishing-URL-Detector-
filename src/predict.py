from pathlib import Path

import joblib

from src.feature_extraction import extract_features

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
model = joblib.load(MODELS_DIR / "phishing_model.pkl")
meta = joblib.load(MODELS_DIR / "features.pkl")

# In the PhiUSIIL dataset: label 1 = legitimate, label 0 = phishing
PHISHING = "Phishing URL"
LEGITIMATE = "Legitimate URL"


def predict_url(url: str) -> str:
    X = extract_features(url, meta["tlds"], meta["columns"])
    return LEGITIMATE if model.predict(X)[0] == 1 else PHISHING
