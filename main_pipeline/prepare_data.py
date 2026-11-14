
import os
import random

import numpy as np
import wfdb
from scipy.signal import butter, filtfilt

# Configuration
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def resolve_dataset_path(raw: str) -> str:
    candidates = []

    if os.path.isabs(raw):
        candidates.append(raw)
    else:
        candidates.append(raw)
        candidates.append(os.path.join(_SCRIPT_DIR, raw))
        candidates.append(os.path.join(_SCRIPT_DIR, "..", raw))

    for path in candidates:
        absolute_path = os.path.abspath(path)

        if os.path.isdir(absolute_path):
            return absolute_path

    raise FileNotFoundError(
        "Dataset folder not found. Tried: "
        + ", ".join(os.path.abspath(c) for c in candidates)
    )

DATASET_PATH = resolve_dataset_path(
    os.environ.get("DATASET_PATH", "mitdb")
)

WINDOW_SIZE = 180
FS = 360
SEED = 42

NORMAL = {"N"}
ABNORMAL = {"V", "A", "L", "R", "F", "E"}

# Reproducibility
def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)

# Signal preprocessing
def bandpass(signal, fs=FS, low=0.5, high=40.0, order=3):
    nyq = fs / 2

    b, a = butter(
        order,
        [low / nyq, high / nyq],
        btype="band",
    )

    return filtfilt(b, a, signal)

def safe_normalize(x, mean, std):

    if not np.isfinite(std) or std == 0:
        raise ValueError(
            "Global standard deviation is invalid; cannot normalize."
        )

    x = (x - mean) / std

    x = np.nan_to_num(
        x,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return x

# Subject wise record grouping
def get_subject_group(record):
    """
    MIT-BIH has 48 records from 47 subjects.

    Records 201 and 202 belong to the same subject.
    All other records are treated as separate subjects.
    """

    if record in {"201", "202"}:
        return "subject_201_202"

    return f"subject_{record}"

def split_records_subject_wise(records):
    """
    Split records into training, validation and test sets
    without allowing records from the same subject to
    appear in different partitions.

    Approximate proportions:
        Training:   70%
        Validation: 15%
        Test:       15%
    """

    records = sorted(set(records))

    # Group recordings belonging to the same subject
    subject_records = {}

    for record in records:
        subject = get_subject_group(record)

        subject_records.setdefault(
            subject,
            [],
        ).append(record)

    subjects = sorted(subject_records)

    if len(subjects) < 3:
        raise ValueError(
            "At least 3 subject groups are required."
        )

    rng = np.random.default_rng(SEED)
    rng.shuffle(subjects)

    n = len(subjects)

    n_train = max(1, int(0.70 * n))
    n_val = max(1, int(0.15 * n))

    if n_train + n_val >= n:
        raise ValueError(
            "Not enough subject groups for train/validation/test."
        )

    train_subjects = subjects[:n_train]

    val_subjects = subjects[
        n_train:n_train + n_val
    ]

    test_subjects = subjects[
        n_train + n_val:
    ]

    train_records = {
        record
        for subject in train_subjects
        for record in subject_records[subject]
    }

    val_records = {
        record
        for subject in val_subjects
        for record in subject_records[subject]
    }

    test_records = {
        record
        for subject in test_subjects
        for record in subject_records[subject]
    }

    # Verify no subject appears in multiple partitions
    train_groups = {
        get_subject_group(record)
        for record in train_records
    }

    val_groups = {
        get_subject_group(record)
        for record in val_records
    }

    test_groups = {
        get_subject_group(record)
        for record in test_records
    }

    assert train_groups.isdisjoint(val_groups)
    assert train_groups.isdisjoint(test_groups)
    assert val_groups.isdisjoint(test_groups)

    return (
        train_records,
        val_records,
        test_records,
    )

# Dataset loading and splitting
def main():

    set_seed(SEED)

    print("Dataset path:", DATASET_PATH)

    records = [
        filename.split(".")[0]
        for filename in os.listdir(DATASET_PATH)
        if filename.endswith(".dat")
    ]

    records = sorted(set(records))

    if not records:
        raise RuntimeError(
            "No MIT-BIH records found (.dat files missing)."
        )

    train_records, val_records, test_records = (
        split_records_subject_wise(records)
    )

    print("\nRecord split:")
    print("Training records:", sorted(train_records))
    print("Validation records:", sorted(val_records))
    print("Test records:", sorted(test_records))

    print(
        f"\nRecords split -> "
        f"train: {len(train_records)} | "
        f"val: {len(val_records)} | "
        f"test: {len(test_records)}"
    )

    X_train, y_train = [], []
    X_val, y_val = [], []
    X_test, y_test = [], []

    # Process ECG records
    for record in records:

        print(f"Processing record {record}...")

        signal, _ = wfdb.rdsamp(
            os.path.join(DATASET_PATH, record)
        )

        annotation = wfdb.rdann(
            os.path.join(DATASET_PATH, record),
            "atr",
        )

        ecg = bandpass(signal[:, 0])

        for i, symbol in enumerate(annotation.symbol):

            pos = annotation.sample[i]

            # Ensure a complete 360 sample beat is available
            if (
                pos - WINDOW_SIZE < 0
                or pos + WINDOW_SIZE > len(ecg)
            ):
                continue

            # Keep only the selected annotation categories
            if (
                symbol not in NORMAL
                and symbol not in ABNORMAL
            ):
                continue

            beat = ecg[
                pos - WINDOW_SIZE:
                pos + WINDOW_SIZE
            ]

            label = (
                0 if symbol in NORMAL else 1
            )

            if record in train_records:

                X_train.append(beat)
                y_train.append(label)

            elif record in val_records:

                X_val.append(beat)
                y_val.append(label)

            elif record in test_records:

                X_test.append(beat)
                y_test.append(label)

    # Convert to NumPy arrays
    X_train = np.asarray(
        X_train,
        dtype=np.float32,
    )

    y_train = np.asarray(
        y_train,
        dtype=np.int32,
    )

    X_val = np.asarray(
        X_val,
        dtype=np.float32,
    )

    y_val = np.asarray(
        y_val,
        dtype=np.int32,
    )

    X_test = np.asarray(
        X_test,
        dtype=np.float32,
    )

    y_test = np.asarray(
        y_test,
        dtype=np.int32,
    )

    # Report class distributions
    print("\n" + "=" * 55)
    print("DATASET CLASS DISTRIBUTION")
    print("=" * 55)

    for name, labels in [
        ("Train", y_train),
        ("Validation", y_val),
        ("Test", y_test),
    ]:

        normal_count = int(
            np.sum(labels == 0)
        )

        abnormal_count = int(
            np.sum(labels == 1)
        )

        print(
            f"{name}: {len(labels)} beats | "
            f"Normal: {normal_count} | "
            f"Abnormal: {abnormal_count}"
        )

    # Check for empty partitions
    if (
        len(X_train) == 0
        or len(X_val) == 0
        or len(X_test) == 0
    ):

        raise RuntimeError(
            "Empty split detected. "
            "Check the dataset and annotation mapping."
        )

    # Check for both classes in every partition
    for name, labels in [
        ("Train", y_train),
        ("Validation", y_val),
        ("Test", y_test),
    ]:

        classes = set(
            np.unique(labels).tolist()
        )

        if classes != {0, 1}:

            raise RuntimeError(
                f"{name} split does not contain both classes. "
                f"Found labels: {sorted(classes)}. "
                "Review the subject-wise split."
            )

    # Training only normalization
    global_mean = float(
        np.mean(X_train)
    )

    global_std = float(
        np.std(X_train)
    )

    X_train = safe_normalize(
        X_train,
        global_mean,
        global_std,
    ).reshape(-1, 360, 1)

    X_val = safe_normalize(
        X_val,
        global_mean,
        global_std,
    ).reshape(-1, 360, 1)

    X_test = safe_normalize(
        X_test,
        global_mean,
        global_std,
    ).reshape(-1, 360, 1)

    # Save all partitions
    np.save("X_train.npy", X_train)
    np.save("y_train.npy", y_train)

    np.save("X_val.npy", X_val)
    np.save("y_val.npy", y_val)

    np.save("X_test.npy", X_test)
    np.save("y_test.npy", y_test)

    np.savez(
        "norm_stats.npz",
        mean=global_mean,
        std=global_std,
    )

    # Final summary
    print("\n" + "=" * 55)
    print("PREPROCESSING COMPLETED")
    print("=" * 55)

    print("Train:", X_train.shape)
    print("Validation:", X_val.shape)
    print("Test:", X_test.shape)

    print("\nNormalization statistics:")
    print("Training mean:", global_mean)
    print("Training std:", global_std)

    print(
        "\nSaved: X/y train-validation-test "
        "and norm_stats.npz"
    )


if __name__ == "__main__":
    main()
