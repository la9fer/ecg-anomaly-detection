# ECG Arrhythmia Detection using Deep Learning and Classical Machine Learning

A comparative machine learning project for binary ECG beat classification using the **MIT-BIH Arrhythmia Database**. This project explores ECG signal preprocessing, statistical feature engineering, one-dimensional convolutional neural networks (1D CNNs), and classical machine learning classifiers.

## Overview

Electrocardiogram (ECG) signals represent the electrical activity of the heart and provide valuable information about cardiac rhythm.

This project investigates automated ECG beat classification through two complementary approaches:

- **Deep Learning:** A one-dimensional Convolutional Neural Network (1D CNN) trained directly on ECG beat segments.
- **Classical Machine Learning:** Support Vector Machine (SVM), Random Forest, and Logistic Regression trained on engineered signal features.

The task is formulated as binary classification:

- **Normal:** N
- **Abnormal:** V, A, L, R, F, E

The selected non-normal annotation categories are grouped into a single abnormal class.

> This is an experimental machine learning project for educational and research purposes. It is not a clinically validated diagnostic system.

## Dataset

**MIT-BIH Arrhythmia Database**

- Source: [PhysioNet – MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/)
- Number of recordings: 48
- Number of subjects: 47
- Sampling frequency: 360 Hz
- Signal channel: First ECG channel
- Beat window: 360 samples
- Window duration: 1 second

Each ECG beat is extracted around its annotated location using:

- 180 samples before the annotation
- 180 samples after the annotation

Annotations outside the selected categories are excluded. Beats without a complete 360-sample window are also discarded.

### Binary label mapping

| Annotation | Class |
|---|---|
| N | Normal (0) |
| V | Abnormal (1) |
| A | Abnormal (1) |
| L | Abnormal (1) |
| R | Abnormal (1) |
| F | Abnormal (1) |
| E | Abnormal (1) |

This is a selected binary classification scheme, not an exhaustive representation of every annotation category in MIT-BIH.

## Methodology

### 1. ECG signal preprocessing

The preprocessing pipeline consists of:

1. Loading ECG recordings and annotations using WFDB.
2. Extracting the first ECG signal channel.
3. Applying a third-order Butterworth bandpass filter between 0.5 Hz and 40 Hz.
4. Extracting fixed-length beat windows around annotated locations.
5. Mapping selected annotation categories to binary labels.
6. Normalizing signals using statistics calculated exclusively from the training partition.

The normalization transformation is:

\[
x_{\mathrm{norm}} =
\frac{x-\mu_{\mathrm{train}}}{\sigma_{\mathrm{train}}}
\]

where the mean and standard deviation are calculated from training data and reused for validation and test data.

### 2. Subject-wise data partitioning

The recordings are divided into three partitions:

- Training: approximately 70%
- Validation: approximately 15%
- Test: approximately 15%

The split is performed at the subject-group level rather than randomly splitting individual beats.

MIT-BIH contains 48 recordings from 47 subjects. Records 201 and 202 are grouped together because they belong to the same subject.

This reduces the risk of recordings from the same subject appearing in different partitions.

### Dataset distribution

The following counts were obtained during the preprocessing run:

| Partition | Total beats | Normal | Abnormal |
|---|---:|---:|---:|
| Training | 73,404 | 53,064 | 20,340 |
| Validation | 12,473 | 10,169 | 2,304 |
| Test | 15,043 | 11,778 | 3,265 |
| **Total** | **100,920** | **75,011** | **25,909** |

The partitions are subject-wise, not beat-wise stratified. Consequently, their class proportions differ.

## Model Architectures

### 1. Convolutional Neural Network (CNN)

A 1D CNN is trained directly on the preprocessed ECG beat segments.

The architecture consists of:

| Layer | Configuration |
|---|---|
| Input | 360 × 1 |
| Convolutional block 1 | 32 filters, kernel size 7 |
| Convolutional block 2 | 64 filters, kernel size 5 |
| Convolutional block 3 | 128 filters, kernel size 3 |
| Convolutional blocks | Batch Normalization, ReLU and Max Pooling |
| Global pooling | Global Average Pooling |
| Regularization | Dropout (0.25) |
| Dense layer | 64 units, ReLU, L2 regularization |
| Output | Single sigmoid neuron |

**Training configuration:**

- Optimizer: Adam
- Initial learning rate: 0.001
- Loss function: Binary cross-entropy
- Batch size: 128
- Maximum epochs: 20
- Random seed: 42

Training uses:

- Early stopping
- Model checkpointing
- Learning-rate reduction on plateau

The best checkpoint is selected using validation PR-AUC.

### 2. Classical machine learning

Five statistical and signal-derived features are extracted from each ECG beat:

1. Mean amplitude
2. Standard deviation
3. Peak amplitude
4. Signal energy
5. Peak-distance proxy

The peak-distance feature is a heuristic calculated from the locations of maximum positive amplitudes in the two halves of a beat. It is not a true physiological RR interval.

The following classifiers are evaluated:

- Support Vector Machine with an RBF kernel
- Random Forest with 300 estimators
- Logistic Regression

Each classifier is evaluated with and without Principal Component Analysis (PCA).

Feature scaling and PCA are implemented using scikit-learn pipelines, ensuring that transformations are fitted on training data only.

Class weighting is used to account for class imbalance. SMOTE is disabled by default.

For computational efficiency, SVM training is limited to 10,000 training samples when the available training partition is larger.

The best configuration within each classifier family is selected using validation PR-AUC.

## Experimental Results

Experiments were executed using Google Colab.

### CNN training

The CNN training process terminated through early stopping.

- Best epoch: 7
- Maximum epochs configured: 20
- Best validation PR-AUC: approximately 0.5734

The best checkpoint was saved as `ecg_cnn_best.keras`.

### Classical model validation results

| Model | PCA | Accuracy | ROC-AUC | PR-AUC |
|---|---|---:|---:|---:|
| SVM | No | 0.5217 | 0.5205 | 0.2774 |
| SVM | Yes | 0.5218 | 0.5205 | 0.2773 |
| Random Forest | No | 0.5859 | 0.4891 | 0.1737 |
| Random Forest | Yes | 0.5815 | 0.4592 | 0.1616 |
| Logistic Regression | No | 0.6272 | 0.5771 | 0.4259 |
| Logistic Regression | Yes | 0.6272 | 0.5771 | 0.4259 |

Logistic Regression achieved the highest validation PR-AUC among the classical model families. Its PCA and non-PCA configurations had identical reported metrics.

### Final held-out test results

The selected classical models and CNN were evaluated on the same held-out test partition.

| Model | Accuracy | Balanced Accuracy | Precision | Recall | Specificity | F1-score | ROC-AUC | PR-AUC | MCC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.6614 | 0.5749 | 0.3006 | 0.4221 | 0.7278 | 0.3511 | **0.6888** | **0.5152** | 0.1342 |
| Random Forest | **0.7752** | 0.6206 | **0.4757** | 0.3473 | **0.8939** | 0.4015 | 0.6738 | 0.4422 | **0.2723** |
| CNN | 0.7443 | **0.6211** | 0.4095 | 0.4034 | 0.8388 | **0.4064** | 0.6773 | 0.4370 | 0.2435 |
| SVM | 0.6582 | 0.5545 | 0.2818 | 0.3712 | 0.7377 | 0.3204 | 0.6485 | 0.3853 | 0.0994 |

### Interpretation and observations

The results reveal that no single model dominates every evaluation metric.

- **Logistic Regression** achieved the highest test ROC-AUC (0.6888), PR-AUC (0.5152), and recall (0.4221).
- **Random Forest** achieved the highest accuracy (0.7752), precision (0.4757), specificity (0.8939), and Matthews correlation coefficient (0.2723).
- **CNN** achieved the highest F1-score (0.4064) and balanced accuracy (0.6211), although its balanced accuracy was very close to Random Forest.
- **SVM** had the weakest overall performance among the evaluated models.

The test-set abnormal-beat prevalence was approximately 21.7%. All four models achieved PR-AUC values above this prevalence baseline.

An important observation is the difference between CNN validation and test performance. Its validation PR-AUC was approximately 0.5734, while its test PR-AUC was 0.4370. This indicates a generalization gap that warrants further investigation.

The experiments demonstrate that a more complex neural network does not necessarily outperform simpler classifiers under a subject-wise evaluation setting.

## Repository Structure

```text
ecg-anomaly-detection/
│
├── main_pipeline/
│   ├── output/
│   ├── prepare_data.py
│   ├── train_cnn.py
│   ├── train_classical_ml.py
│   ├── compare_models.py
│   ├── test_cnn.py
│   ├── predict_realtime.py
│   ├── alert_system.py
│   ├── serial_infer.py
│   ├── run_full_pipeline_output.py
│   ├── sample_quick_test.py
│   ├── requirements.txt
│   ├── README.md
│   └── .gitignore
│
├── README.md
└── .gitignore
```

Generated datasets and trained model artifacts are not necessarily committed to the repository.

## Running the Project

### 1. Install dependencies

```bash
pip install -r main_pipeline/requirements.txt
```

### 2. Download the dataset

Download the MIT-BIH Arrhythmia Database from [PhysioNet](https://physionet.org/content/mitdb/) and place its files in:

```text
main_pipeline/mitdb/
```

Alternatively, configure the `DATASET_PATH` environment variable to point to the dataset directory.

### 3. Prepare the data

```bash
cd main_pipeline
python prepare_data.py
```

This generates:

```text
X_train.npy
y_train.npy
X_val.npy
y_val.npy
X_test.npy
y_test.npy
norm_stats.npz
```

### 4. Train the CNN

```bash
python train_cnn.py
```

### 5. Train classical models

```bash
python train_classical_ml.py
```

### 6. Compare models

```bash
python compare_models.py
```

The final comparison is saved to:

```text
models/final_test_comparison.json
```

## Technologies Used

- Python
- NumPy
- SciPy
- WFDB
- TensorFlow / Keras
- scikit-learn
- Matplotlib
- Joblib

## Limitations and Future Work

The current implementation has several limitations:

1. **Class imbalance:** Normal beats are more frequent than abnormal beats. Class weighting is used in classical models, but further imbalance-handling strategies could be explored.
2. **Feature engineering:** Classical models use only five handcrafted features. More informative morphological, temporal and frequency-domain features may improve performance.
3. **Generalization:** The CNN exhibits a validation-to-test PR-AUC gap. Further investigation of subject variability, model capacity, regularization and split sensitivity is needed.
4. **Subject-level evaluation:** The subject-wise split helps reduce beat-level leakage, but this experiment does not establish broad clinical generalization.
5. **Binary classification:** Combining several abnormal beat categories into one class removes distinctions between arrhythmia types.
6. **Threshold selection:** The current classification threshold is 0.5. Threshold optimization and probability calibration should be investigated using validation data.
7. **Hardware deployment:** Real-time and embedded inference components should be considered prototype functionality until their end-to-end performance and reliability have been independently tested.

Potential future improvements include:

- More extensive ECG morphological and temporal feature extraction.
- Hyperparameter optimization using validation data.
- Evaluation across multiple subject-wise splits.
- Class-specific classification rather than binary grouping.
- Calibration and threshold analysis.
- Investigation of lightweight models for embedded ECG inference.

## Disclaimer

This project is intended for educational and research purposes. It is not a medical device, does not provide a diagnosis, and must not be used as a substitute for professional medical evaluation.
