from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from src.feature_extraction import features_dict, tld_of

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / "data" / "PhiUSIIL_Phishing_URL_Dataset.csv", usecols=["URL", "label"])
df = df.drop_duplicates(subset="URL").reset_index(drop=True)

train_idx, test_idx = train_test_split(
    df.index, test_size=0.2, random_state=42, stratify=df["label"]
)

# The 40 most frequent TLDs, chosen from the TRAINING part only
tlds = list(df.loc[train_idx, "URL"].map(tld_of).value_counts().head(40).index)

X = pd.DataFrame([features_dict(u, tlds) for u in df["URL"]])
y = df["label"]
columns = list(X.columns)

model = RandomForestClassifier(
    n_estimators=100, min_samples_leaf=10, class_weight="balanced", random_state=42, n_jobs=-1
)
model.fit(X.loc[train_idx], y.loc[train_idx])

y_pred = model.predict(X.loc[test_idx])
print("Accuracy:", accuracy_score(y.loc[test_idx], y_pred))
print("\nClassification report:\n", classification_report(y.loc[test_idx], y_pred))

joblib.dump(model, ROOT / "models" / "phishing_model.pkl")
joblib.dump({"columns": columns, "tlds": tlds}, ROOT / "models" / "features.pkl")
print("Model saved successfully!")
