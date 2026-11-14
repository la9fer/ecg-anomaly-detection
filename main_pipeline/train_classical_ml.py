
import os
import json
import random
import joblib
import numpy as np

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# --------------------------------------------------
# Configuration
# --------------------------------------------------

SEED = 42

USE_SMOTE = False
APPLY_PCA = [False, True]
PCA_COMPONENTS = 5

SVM_MAX_SAMPLES = int(
    os.environ.get("SVM_MAX_SAMPLES", "10000")
)


# --------------------------------------------------
# Reproducibility
# --------------------------------------------------

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)


# --------------------------------------------------
# Feature extraction
# --------------------------------------------------

def extract_features(x_360):
    """
    Extract five simple statistical and peak-location
    features from a 360-sample ECG beat.

    Note:
    The peak-distance feature is a heuristic computed
    within an individual beat. It is not a true RR interval.
    """

    x = np.asarray(x_360, dtype=np.float64).reshape(-1)

    if x.shape[0] != 360:
        raise ValueError(
            f"Expected 360 samples, received {x.shape[0]}"
        )

    if not np.all(np.isfinite(x)):
        x = np.nan_to_num(x)

    mean = np.mean(x)
    std = np.std(x)
    peak_amp = np.max(np.abs(x))
    energy = np.sum(x ** 2) / len(x)

    first_peak = np.argmax(x[:180])
    second_peak = np.argmax(x[180:]) + 180

    peak_distance_proxy = abs(second_peak - first_peak)

    return np.array(
        [
            mean,
            std,
            peak_amp,
            energy,
            peak_distance_proxy,
        ],
        dtype=np.float32,
    )


def build_feature_matrix(X):
    """
    Convert a collection of ECG beats into a feature matrix.
    """

    X = np.asarray(X)

    if X.ndim < 2:
        raise ValueError(
            f"Expected a collection of beats, got shape {X.shape}"
        )

    return np.vstack(
        [extract_features(beat) for beat in X]
    )


# --------------------------------------------------
# Optional SMOTE
# --------------------------------------------------

def maybe_smote(X, y):
    """
    Apply SMOTE only to the training data, if enabled.
    """

    if not USE_SMOTE:
        return X, y

    try:
        from imblearn.over_sampling import SMOTE
    except ImportError as exc:
        raise ImportError(
            "Install imbalanced-learn to use SMOTE."
        ) from exc

    smote = SMOTE(random_state=SEED)

    return smote.fit_resample(X, y)


# --------------------------------------------------
# Validation metrics
# --------------------------------------------------

def get_positive_probabilities(model, X):
    """
    Return probabilities corresponding specifically to label 1.
    """

    if not hasattr(model, "predict_proba"):
        raise ValueError(
            "The model does not support predict_proba()."
        )

    probabilities = model.predict_proba(X)

    if probabilities.ndim != 2:
        raise ValueError("Unexpected probability array shape.")

    classes = model.named_steps["classifier"].classes_

    positive_indices = np.flatnonzero(classes == 1)

    if len(positive_indices) != 1:
        raise ValueError(
            f"Could not identify positive class 1. "
            f"Found classes: {classes}"
        )

    return probabilities[:, positive_indices[0]]


def calculate_metrics(y_true, y_pred, y_prob):
    """
    Calculate binary classification metrics.
    """

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, zero_division=0)
        ),
        "f1": float(
            f1_score(y_true, y_pred, zero_division=0)
        ),
        "roc_auc": float(
            roc_auc_score(y_true, y_prob)
        ),
        "pr_auc": float(
            average_precision_score(y_true, y_prob)
        ),
    }


# --------------------------------------------------
# Model construction
# --------------------------------------------------

def build_models(use_pca):
    """
    Build SVM, Random Forest and Logistic Regression pipelines.

    Scaling and PCA are fitted only on the training data
    because they are inside the sklearn Pipeline.
    """

    models = {}

    for_pca_steps = [
        ("scaler", StandardScaler())
    ]

    if use_pca:
        for_pca_steps.append(
            ("pca", PCA(n_components=PCA_COMPONENTS))
        )

    models["SVM"] = Pipeline(
        for_pca_steps + [
            (
                "classifier",
                SVC(
                    kernel="rbf",
                    C=2,
                    gamma="scale",
                    probability=True,
                    class_weight="balanced",
                    random_state=SEED,
                ),
            )
        ]
    )

    models["RandomForest"] = Pipeline(
        for_pca_steps + [
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=SEED,
                    class_weight="balanced",
                    n_jobs=-1,
                ),
            )
        ]
    )

    models["LogisticRegression"] = Pipeline(
        for_pca_steps + [
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=SEED,
                ),
            )
        ]
    )

    return models


# --------------------------------------------------
# Training and validation
# --------------------------------------------------

def main():

    set_seed()

    os.makedirs("models", exist_ok=True)

    # Load the already prepared partitions.
    X_train = np.load("X_train.npy")
    y_train = np.load("y_train.npy").reshape(-1)

    X_val = np.load("X_val.npy")
    y_val = np.load("y_val.npy").reshape(-1)

    # Basic validation checks.
    if len(X_train) != len(y_train):
        raise ValueError("Training data and labels have different lengths.")

    if len(X_val) != len(y_val):
        raise ValueError("Validation data and labels have different lengths.")

    if set(np.unique(y_train)) != {0, 1}:
        raise ValueError(
            f"Training set must contain both binary classes. "
            f"Found: {np.unique(y_train)}"
        )

    if set(np.unique(y_val)) != {0, 1}:
        raise ValueError(
            f"Validation set must contain both binary classes. "
            f"Found: {np.unique(y_val)}"
        )

    print("\nTraining samples:", len(y_train))
    print("Validation samples:", len(y_val))

    print("\nTraining class counts:")
    print(dict(zip(*np.unique(y_train, return_counts=True))))

    print("\nValidation class counts:")
    print(dict(zip(*np.unique(y_val, return_counts=True))))

    # Feature extraction.
    Xtr = build_feature_matrix(X_train)
    Xv = build_feature_matrix(X_val)

    print("\nFeature matrix shape:", Xtr.shape)

    if PCA_COMPONENTS > Xtr.shape[1]:
        raise ValueError(
            "PCA_COMPONENTS cannot exceed the number of features."
        )

    # Optional SMOTE: training data only.
    Xtr, ytr = maybe_smote(Xtr, y_train)

    print("\nTraining after optional resampling:", len(ytr))

    validation_results = []
    selected_models = {}

    # Train all configurations.
    for use_pca in APPLY_PCA:

        config_name = "with_PCA" if use_pca else "without_PCA"

        print("\n" + "=" * 60)
        print(f"CONFIGURATION: {config_name}")
        print("=" * 60)

        models = build_models(use_pca)

        for model_name, model in models.items():

            print(f"\nTraining {model_name} ({config_name})...")

            X_fit = Xtr
            y_fit = ytr

            # Limit SVM training size for computational efficiency.
            # This sampling is performed only on training data.
            if (
                model_name == "SVM"
                and len(Xtr) > SVM_MAX_SAMPLES
            ):

                X_fit, _, y_fit, _ = train_test_split(
                    Xtr,
                    ytr,
                    train_size=SVM_MAX_SAMPLES,
                    stratify=ytr,
                    random_state=SEED,
                )

                print(
                    f"Using {len(y_fit)} training samples for SVM."
                )

            model.fit(X_fit, y_fit)

            # Validation predictions.
            y_pred = model.predict(Xv)
            y_prob = get_positive_probabilities(model, Xv)

            metrics = calculate_metrics(
                y_val,
                y_pred,
                y_prob,
            )

            print("Validation metrics:")

            for metric_name, value in metrics.items():
                print(f"{metric_name}: {value:.4f}")

            # Save fitted candidate model.
            filename = f"{model_name}_{config_name}.joblib"

            model_path = os.path.join("models", filename)

            joblib.dump(model, model_path)

            result = {
                "model": model_name,
                "configuration": config_name,
                "filename": filename,
                "metrics": metrics,
            }

            validation_results.append(result)

            # Select the best configuration for each model family
            # using validation PR-AUC.
            previous = selected_models.get(model_name)

            if (
                previous is None
                or metrics["pr_auc"] > previous["metrics"]["pr_auc"]
            ):
                selected_models[model_name] = result

    # Save all validation results.
    with open(
        "models/classical_validation_results.json",
        "w",
    ) as file:
        json.dump(
            validation_results,
            file,
            indent=4,
        )

    # Save the selected configuration for each model family.
    with open(
        "models/selected_classical_models.json",
        "w",
    ) as file:
        json.dump(
            selected_models,
            file,
            indent=4,
        )

    print("\n" + "=" * 60)
    print("SELECTED CLASSICAL MODELS")
    print("=" * 60)

    for model_name, result in selected_models.items():

        print(
            f"{model_name}: {result['filename']} "
            f"(Validation PR-AUC: "
            f"{result['metrics']['pr_auc']:.4f})"
        )

    print("\nTraining and validation completed.")
    print("Models saved in the models/ directory.")


if __name__ == "__main__":
    main()
