import os
import json
import joblib
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from train_classical_ml import build_feature_matrix

# Metrics
def binary_metrics(y_true, y_pred, y_prob):

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    return {
        "accuracy": accuracy_score(y_true, y_pred),

        "balanced_accuracy": balanced_accuracy_score(
            y_true,
            y_pred,
        ),

        "precision": precision_score(
            y_true,
            y_pred,
            zero_division=0,
        ),

        "recall": recall_score(
            y_true,
            y_pred,
            zero_division=0,
        ),

        "specificity": specificity,

        "f1": f1_score(
            y_true,
            y_pred,
            zero_division=0,
        ),

        "roc_auc": roc_auc_score(
            y_true,
            y_prob,
        ),

        "pr_auc": average_precision_score(
            y_true,
            y_prob,
        ),

        "mcc": matthews_corrcoef(
            y_true,
            y_pred,
        ),
    }

# Positive class probability
def get_positive_probabilities(model, X):

    if not hasattr(model, "predict_proba"):
        raise ValueError(
            "Classical model does not support predict_proba()."
        )

    probabilities = model.predict_proba(X)

    classes = model.named_steps["classifier"].classes_

    positive_indices = np.flatnonzero(classes == 1)

    if len(positive_indices) != 1:
        raise ValueError(
            f"Could not identify positive class 1. "
            f"Found classes: {classes}"
        )

    return probabilities[:, positive_indices[0]]

# Main comparison
def main():

    if not os.path.exists("models"):
        raise FileNotFoundError(
            "The models/ directory does not exist."
        )

    # Load the untouched test partition.
    X_test = np.load("X_test.npy")
    y_test = np.load("y_test.npy").reshape(-1)

    if len(X_test) != len(y_test):
        raise ValueError(
            "Test data and labels have different lengths."
        )

    if set(np.unique(y_test)) != {0, 1}:
        raise ValueError(
            "Test set must contain both binary classes. "
            f"Found: {np.unique(y_test)}"
        )

    print("Test samples:", len(y_test))

    print("\nTest class counts:")
    print(dict(zip(*np.unique(y_test, return_counts=True))))

    # CNN evaluation
    cnn_path = "models/ecg_cnn.keras"

    if not os.path.exists(cnn_path):
        raise FileNotFoundError(
            f"CNN model not found: {cnn_path}"
        )

    print("\nLoading CNN...")

    cnn = tf.keras.models.load_model(cnn_path)

    X_cnn = X_test

    if X_cnn.ndim == 2:
        X_cnn = X_cnn[..., np.newaxis]

    elif X_cnn.ndim == 3 and X_cnn.shape[-1] != 1:
        raise ValueError(
            f"Unexpected CNN input shape: {X_cnn.shape}"
        )

    cnn_prob = cnn.predict(
        X_cnn,
        verbose=0,
    ).reshape(-1)

    if len(cnn_prob) != len(y_test):
        raise ValueError(
            "CNN predictions and test labels have different lengths."
        )

    cnn_pred = (cnn_prob >= 0.5).astype(int)

    results = []

    cnn_metrics = binary_metrics(
        y_test,
        cnn_pred,
        cnn_prob,
    )

    results.append(
        {
            "model": "CNN",
            **cnn_metrics,
        }
    )

    # Load validation selected classical models
    selection_path = "models/selected_classical_models.json"

    if not os.path.exists(selection_path):
        raise FileNotFoundError(
            "Selected classical models file not found. "
            "Run train_classical_ml.py first."
        )

    with open(selection_path, "r") as file:
        selected_models = json.load(file)

    # Extract features from the same held out test beats.
    X_features = build_feature_matrix(X_test)

    for model_name, selection in selected_models.items():

        filename = selection["filename"]

        model_path = os.path.join(
            "models",
            filename,
        )

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Selected model not found: {model_path}"
            )

        print(f"\nEvaluating {model_name}...")

        model = joblib.load(model_path)

        y_pred = model.predict(X_features)

        y_prob = get_positive_probabilities(
            model,
            X_features,
        )

        if len(y_pred) != len(y_test):
            raise ValueError(
                f"{model_name}: prediction length mismatch."
            )

        if len(y_prob) != len(y_test):
            raise ValueError(
                f"{model_name}: probability length mismatch."
            )

        metrics = binary_metrics(
            y_test,
            y_pred,
            y_prob,
        )

        results.append(
            {
                "model": model_name,
                **metrics,
            }
        )

    # Display comparison
    results.sort(
        key=lambda row: row["pr_auc"],
        reverse=True,
    )

    columns = [
        "model",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
        "roc_auc",
        "pr_auc",
        "mcc",
    ]

    print("\n" + "=" * 110)
    print("FINAL TEST-SET MODEL COMPARISON")
    print("=" * 110)

    header = " | ".join(columns)

    print(header)
    print("-" * len(header))

    for row in results:

        values = []

        for column in columns:

            value = row[column]

            if isinstance(value, (int, float, np.number)):
                values.append(f"{value:.4f}")

            else:
                values.append(str(value))

        print(" | ".join(values))

    # Save final comparison
    with open(
        "models/final_test_comparison.json",
        "w",
    ) as file:
        json.dump(
            results,
            file,
            indent=4,
        )

    print("\nFinal test comparison saved to:")
    print("models/final_test_comparison.json")

    print(
        "\nHighest test PR-AUC:",
        results[0]["model"],
        f"({results[0]['pr_auc']:.4f})",
    )

if __name__ == "__main__":
    main()
