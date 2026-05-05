import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "manual_deps"))
sys.path.insert(1, str(Path(__file__).parent / "python_deps"))
sys.path.insert(2, str(Path(__file__).parent / ".deps"))
sys.path.insert(3, str(Path(__file__).parent / ".python_packages"))

import pandas as pd
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib
import json

DATASET_PATH = "symptom_routing_expanded_dataset.csv"
METRICS_PATH = "metrics_phase2.json"
PREDICTIONS_PATH = "predictions_phase2.csv"
MODEL_PATH = "model_phase2.joblib"

df = pd.read_csv(DATASET_PATH)

train_df, test_df = train_test_split(
    df,
    test_size=0.25,
    random_state=712,
    stratify=df["label"]
)

model = SentenceTransformer("nicher92/saga-embed_v1")

X_train = model.encode(train_df["text"].tolist(), normalize_embeddings=True)
X_test = model.encode(test_df["text"].tolist(), normalize_embeddings=True)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, train_df["label"])

preds = clf.predict(X_test)

metrics = {
    "accuracy": accuracy_score(test_df["label"], preds),
    "macro_f1": f1_score(test_df["label"], preds, average="macro"),
    "weighted_f1": f1_score(test_df["label"], preds, average="weighted"),
    "classification_report": classification_report(
        test_df["label"], preds, output_dict=True
    )
}

with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f, indent=2)

predictions = test_df.copy()
predictions["predicted_label"] = preds
predictions.to_csv(PREDICTIONS_PATH, index=False)

joblib.dump(clf, MODEL_PATH)

print(f"Dataset: {DATASET_PATH}")
print(f"Rows: {len(df)}")
print(f"Accuracy: {metrics['accuracy']:.3f}")
print(f"Macro F1: {metrics['macro_f1']:.3f}")
print(f"Weighted F1: {metrics['weighted_f1']:.3f}")
print(f"Saved {METRICS_PATH}, {PREDICTIONS_PATH}, and {MODEL_PATH}")
