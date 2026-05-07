import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC


DEFAULT_DATASET_PATH = "symptom_routing_expanded_dataset.csv"
DEFAULT_EMBEDDING_MODEL = "nicher92/saga-embed_v1"
DEFAULT_TEST_SIZE = 0.25
DEFAULT_RANDOM_SEED = 712

DEPLOYED_MODEL_PATH = "model_phase2.joblib"
ALL_MODELS_PATH = "all_classifiers_phase2.joblib"
METRICS_PATH = "metrics.json"
PREDICTIONS_PATH = "predictions_phase2.csv"
TRAIN_SPLIT_PATH = "train.csv"
TEST_SPLIT_PATH = "test.csv"


def load_dataset(dataset_path):
    df = pd.read_csv(dataset_path)

    required_columns = {"text", "label"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataset is missing required column(s): {missing}")

    df = df[["text", "label"]].copy()
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip()
    df = df[(df["text"] != "") & (df["label"] != "")]

    if df.empty:
        raise ValueError("Dataset has no valid rows after cleaning.")

    return df


def build_classifiers(random_seed):
    return {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=random_seed,
        ),
        "linear_svm": LinearSVC(
            class_weight="balanced",
            random_state=random_seed,
        ),
        "knn_cosine": KNeighborsClassifier(
            n_neighbors=3,
            metric="cosine",
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=random_seed,
        ),
    }


def evaluate_classifier(classifier, X_test, y_test):
    predictions = classifier.predict(X_test)
    labels = sorted(y_test.unique())

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "macro_f1": f1_score(y_test, predictions, average="macro"),
        "weighted_f1": f1_score(y_test, predictions, average="weighted"),
        "classification_report": classification_report(
            y_test,
            predictions,
            labels=labels,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            y_test,
            predictions,
            labels=labels,
        ).tolist(),
        "labels": labels,
    }


def choose_best_classifier(results):
    return max(
        results,
        key=lambda name: (
            results[name]["macro_f1"],
            results[name]["accuracy"],
            results[name]["weighted_f1"],
        ),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate embedding-based symptom routing classifiers."
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET_PATH,
        help=f"Path to dataset CSV. Default: {DEFAULT_DATASET_PATH}",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help=f"SentenceTransformer embedding model. Default: {DEFAULT_EMBEDDING_MODEL}",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=DEFAULT_TEST_SIZE,
        help=f"Test split size. Default: {DEFAULT_TEST_SIZE}",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help=f"Random seed for train/test split. Default: {DEFAULT_RANDOM_SEED}",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    df = load_dataset(dataset_path)

    train_df, test_df = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.random_seed,
        stratify=df["label"],
    )

    train_df.to_csv(TRAIN_SPLIT_PATH, index=False)
    test_df.to_csv(TEST_SPLIT_PATH, index=False)

    embedding_model = SentenceTransformer(args.embedding_model)
    X_train = embedding_model.encode(
        train_df["text"].tolist(),
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    X_test = embedding_model.encode(
        test_df["text"].tolist(),
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    classifiers = build_classifiers(args.random_seed)
    trained_classifiers = {}
    results = {}

    for classifier_name, classifier in classifiers.items():
        classifier.fit(X_train, train_df["label"])
        trained_classifiers[classifier_name] = classifier
        results[classifier_name] = evaluate_classifier(
            classifier,
            X_test,
            test_df["label"],
        )

    best_classifier_name = choose_best_classifier(results)
    deployed_classifier = trained_classifiers[best_classifier_name]

    joblib.dump(deployed_classifier, DEPLOYED_MODEL_PATH)
    joblib.dump(trained_classifiers, ALL_MODELS_PATH)

    best_predictions = test_df.copy()
    best_predictions["predicted_label"] = deployed_classifier.predict(X_test)
    best_predictions["classifier"] = best_classifier_name
    best_predictions.to_csv(PREDICTIONS_PATH, index=False)

    metrics = {
        "dataset": str(dataset_path),
        "rows": len(df),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "test_size": args.test_size,
        "random_seed": args.random_seed,
        "embedding_model": args.embedding_model,
        "embedding_usage": "Frozen text encoder; only downstream classifiers were trained.",
        "best_classifier": best_classifier_name,
        "best_metrics": {
            "accuracy": results[best_classifier_name]["accuracy"],
            "macro_f1": results[best_classifier_name]["macro_f1"],
            "weighted_f1": results[best_classifier_name]["weighted_f1"],
        },
        "classifier_results": results,
        "label_distribution": df["label"].value_counts().sort_index().to_dict(),
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)

    print(f"Dataset: {dataset_path}")
    print(f"Rows: {len(df)}")
    print(f"Train rows: {len(train_df)}")
    print(f"Test rows: {len(test_df)}")
    print(f"Embedding model: {args.embedding_model}")
    print()
    print("Classifier comparison:")
    for classifier_name, classifier_metrics in results.items():
        print(
            f"- {classifier_name}: "
            f"accuracy={classifier_metrics['accuracy']:.3f}, "
            f"macro_f1={classifier_metrics['macro_f1']:.3f}, "
            f"weighted_f1={classifier_metrics['weighted_f1']:.3f}"
        )
    print()
    print(f"Best classifier: {best_classifier_name}")
    print(f"Saved deployed model to {DEPLOYED_MODEL_PATH}")
    print(f"Saved all classifiers to {ALL_MODELS_PATH}")
    print(f"Saved metrics to {METRICS_PATH}")
    print(f"Saved predictions to {PREDICTIONS_PATH}")
    print(f"Saved train/test splits to {TRAIN_SPLIT_PATH} and {TEST_SPLIT_PATH}")


if __name__ == "__main__":
    main()
