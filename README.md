# Assignment 2 – Binary Linear Neural Network (Part A)

## Overview

This notebook builds and tunes a **binary Linear Neural Network (LNN)** (logistic regression written as a 1-layer neural net) on the census-income dataset. The focus is on:

- Careful data preprocessing and one-hot encoding
- Creating proper **train / validation / test** splits
- Training a logistic-regression LNN with **Keras + Keras Tuner**
- Comparing performance **with vs without feature standardization**
- Using loss/accuracy, confusion matrix, and classification report to evaluate the model

This is Part A of Assignment 2 in the “ML Toolbox-1” course.

---

## Dataset

- **Source:** Census income dataset (loaded from `adult(in).csv`)
- **Features:**
  - Numeric: age, fnlwgt, education-num, capital-gain, capital-loss, hours-per-week, etc.
  - Categorical: workclass, education, marital-status, occupation, relationship, race, sex, native-country, etc.
- **Target:**  
  - `income` converted to a binary column `income_ >50K`:
    - `0` → `<=50K`
    - `1` → `>50K`

---

## What the Notebook Does

### 1. Loading and Inspecting the Data

- Mounts Google Drive (for Colab).
- Reads `adult(in).csv` into `df_raw`.
- Uses `info()` and `head()` to inspect data types and a few example rows.

### 2. Data Cleaning and Missing Values

- Treats `" ?"` entries as missing and replaces them with `NaN`.
- Fills remaining missing values using the **median age** (simple imputation strategy).
- Verifies that no missing values remain.

### 3. One-Hot Encoding and DataFrame Construction

- Identifies **categorical columns** (object / category types).
- Applies **one-hot encoding** with `pd.get_dummies(..., drop_first=True)`:
  - Produces binary indicator columns for each category.
  - Drops one level per variable to avoid redundancy.
- Collects all **non-categorical (numeric)** columns.
- Concatenates encoded categorical + numeric columns into a single DataFrame `df`.
- Copies `df` into `df_copy` for later experiments.

### 4. Feature and Target Matrices

- Defines:
  - `df_features = df.drop('income_ >50K', axis=1)`
  - `target_binary_income = df['income_ >50K']`
- Converts to NumPy:
  - `X = df_features.to_numpy()`
  - `y = target_binary_income.to_numpy()`
- Prints shapes to confirm sizes and types.

### 5. Train / Validation / Test Split

- Uses `train_test_split` with stratification to preserve class balance:
  - First: 20% as **test** set.
  - Second: from the remaining 80%, takes 20% for **validation**.
- Final splits:
  - `X_train`, `y_train`  
  - `X_val`, `y_val`  
  - `X_test`, `y_test`
- Utility function `shape()` prints the dimensions of each split.

---

## Experiments with Binary LNN (Logistic Regression)

### Standardization Step

- For the scaled experiment, applies `StandardScaler`:
  - Fits on **train only**: `X_train_scaled = scaler.fit_transform(X_train)`
  - Transforms validation & test using the same scaler:
    - `X_val_scaled`, `X_test_scaled`
- This avoids data leakage from validation/test into the scaler statistics.

---

### Experiment 1 – Hyperparameter-Tuned LNN with Standardization

Goal: Find an optimal **logistic regression LNN** on standardized features using **Keras Tuner (Hyperband)**.

Key steps:

1. **LNN architecture:**
   - Input layer with `n_features` inputs.
   - Single `Dense(1)` output with `sigmoid` activation.
   - L2 regularization applied to the weights.

2. **Hyperparameters tuned:**
   - L2 penalty: `[1e-6, 1e-5, 1e-4, 1e-3, 1e-2]`
   - Learning rate: continuous range `[1e-4, 5e-1]` on a log scale.
   - Number of epochs: integers in `[10, 100]` (via tuner’s `fit`).
   - Batch size: `{32, 64, 128, 256}`.

3. **Training setup:**
   - Loss: `binary_crossentropy`
   - Optimizer: `SGD` (with tuned learning rate)
   - Metric: accuracy
   - Tuner: `kt.Hyperband` targeting **validation accuracy**.
   - Early stopping callback to:
     - Stop unpromising trials.
     - Restore best model weights based on validation loss.

4. **Evaluation:**
   - Evaluates best model on:
     - Training set (`X_train_scaled`, `y_train`)
     - Test set (`X_test_scaled`, `y_test`)
   - Reports:
     - Train/test **loss and accuracy**
     - **Confusion matrix** on test labels
     - **Classification report** (precision, recall, F1, per class and overall)
   - Shows how a well-tuned LNN with scaling performs on this binary classification task.

---

### Experiment 2 – Hyperparameter-Tuned LNN Without Feature Scaling

Goal: Repeat the LNN training **without any standardization**, to see how raw feature scales affect optimization and generalization.

Key steps:

1. **Prepare raw feature matrix:**
   - Uses `df_copy` with binary target column.
   - Forms `X_ns` (all features) and `y_ns` (`income_ >50K`) as NumPy arrays.

2. **Train / validation / test split:**
   - Again uses stratified `train_test_split`:
     - 20% for test.
     - From the remaining, 20% for validation.

3. **LNN architecture and hyperparameters:**
   - Same linear model: Input(d) → Dense(1, sigmoid).
   - Hyperparameters:
     - Learning rate: `[1e-5, 1e-1]` (narrower range to handle unscaled inputs).
     - L2 penalty: `[0.0, 1e-6, 1e-5, 1e-4, 1e-3]`
       - Includes **0.0** → no regularization.
   - Uses a separate Hyperband tuner and early stopping for this no-scale experiment.

4. **Evaluation:**
   - Evaluates best no-scale model on train and test splits.
   - Prints:
     - Train/test loss and accuracy
     - Test confusion matrix
     - Test classification report
   - Compares these results conceptually to Experiment 1 to understand:
     - The importance of **feature scaling**.
     - How optimization gets harder when features have very different scales.

---

## Libraries Used

- Python 3.x
- Google Colab + Google Drive
- `pandas`, `numpy`
- `matplotlib`
- `scikit-learn`
  - `train_test_split`
  - `StandardScaler`
  - Metrics: `accuracy_score`, `confusion_matrix`, `classification_report`
- `tensorflow` / `keras`
  - `keras.Sequential`, `Dense`, `regularizers`
- `keras-tuner` (Hyperband tuner for hyperparameter search)

---

## How to Run

1. Open the notebook in **Google Colab**.
2. Make sure `adult(in).csv` is in your Google Drive at the path used in the notebook, or update the path in `pd.read_csv(...)`.
3. Install any missing packages (e.g., Keras Tuner) if Colab does not have them:
   ```bash
   pip install -U keras-tuner

--------------------------------------------------------------------------------


# Assignment 2 – Multiclass Softmax Neural Network (Part B)

## Overview

This notebook trains and analyzes a **multiclass Linear Neural Network (softmax regression)** on the classic Seeds dataset. The focus is on:

- Proper preprocessing and standardization of numeric features  
- Creating **train / validation / test** splits with stratification  
- Training a **baseline softmax classifier**  
- Adding **L2** and **L1** regularization to control model complexity  
- Comparing models using accuracy, confusion matrix, classification report, and learning curves

This is Part B of Assignment 2 in the “ML Toolbox-1” course.

---

## Dataset

- **Source:** Seeds dataset (loaded from `seeds(in).csv`)
- **Input features (7):**
  - `area`, `perimeter`, `compactness`, `length`, `width`, `asymmetry`, `groove`
- **Target:**
  - Wheat variety label (three classes)
  - Original labels `1, 2, 3` are remapped to `0, 1, 2` for Keras:
    - `0` → class 1  
    - `1` → class 2  
    - `2` → class 3

---

## What the Notebook Does

### 1. Loading and Inspecting the Data

- Mounts Google Drive (for Colab usage).
- Reads `seeds(in).csv` into a pandas DataFrame.
- Assigns column names and prints:
  - `df.head()` to see example rows
  - `df.info()` to check dtypes
  - Number of missing values per column  
- Confirms the Seeds dataset has **no missing data**.

### 2. Splitting into Features and Target

- Separates:
  - `features_seeds` → all columns except the last (the 7 numeric features)
  - `target_seeds` → the last column (`class`)
- Converts to NumPy:
  - `X` = feature matrix
  - `y_label` = original class labels
- Remaps labels to start at 0:
  - `y = y_label - 1`
- Prints shapes and a few examples of `(X, y)` to verify.

### 3. Train / Validation / Test Split

- Casts arrays to suitable dtypes:
  - `X_all = X.astype(np.float32)`
  - `y_all = y.astype(np.int32).ravel()`
- Uses **stratified** `train_test_split`:
  1. First split: 20% of the data as **test set**.
  2. Second split: from the remaining 80%, takes 20% as **validation set**.
- Final sets:
  - `X_train`, `X_valid`, `X_test`
  - `y_train`, `y_valid`, `y_test`
- Prints shapes and class counts (using `Counter`) to show that all three classes appear in each split.

### 4. Standardization

- Fits a `StandardScaler` **only on the training data**:
  - `scaler_seeds.fit(X_train)`
- Transforms:
  - `X_train_std = scaler_seeds.transform(X_train)`
  - `X_valid_std = scaler_seeds.transform(X_valid)`
  - `X_test_std  = scaler_seeds.transform(X_test)`
- This keeps validation and test sets “clean” from train information (no leakage).

---

## Experiments with Softmax Regression (Multiclass LNN)

### Experiment 1 – Baseline Softmax Neural Network

Goal: Train a **linear softmax classifier** without explicit L1/L2 penalties.

- **Model architecture**
  - Input: 7 standardized features
  - Output: `Dense(3, activation='softmax')` for 3 classes

- **Training setup**
  - Loss: `sparse_categorical_crossentropy`
  - Optimizer: `SGD` with a fixed learning rate
  - Metric: accuracy
  - Train on `X_train_std`, validate on `X_valid_std`
  - Runs for a fixed number of epochs with a chosen batch size

- **Evaluation**
  - Predicts labels for:
    - Training set (`X_train_std`)
    - Test set (`X_test_std`)
  - Computes:
    - Training accuracy
    - Test accuracy
    - **Confusion matrix** on test set
    - **Classification report** (precision, recall, F1 score for each class)
  - Plots **learning curves**:
    - Accuracy vs epochs (train vs validation)
    - Loss vs epochs (train vs validation)

---

### Experiment 2 – Softmax Neural Network with L2 Regularization

Goal: Add **L2 weight decay** to reduce overfitting and smooth the model weights.

- **Model architecture**
  - Same as baseline, but:
    - `Dense(3, activation='softmax', kernel_regularizer=regularizers.l2(lambda_l2))`

- **Hyperparameters**
  - L2 penalty `lambda_l2` set to a small positive value  
  - Learning rate, epochs, and batch size chosen for stable training

- **Training and evaluation**
  - Trains on `X_train_std` and validates on `X_valid_std`
  - Tracks training and validation accuracy/loss across epochs
  - Evaluates performance on the test set:
    - Test accuracy
    - Learning curves (showing how regularization changes the training/validation gap)

---

### Experiment 3 – Softmax Neural Network with L1 Regularization

Goal: Use **L1 regularization** to encourage sparse weights and compare behaviour with L2.

- **Model architecture**
  - Same structure but with:
    - `Dense(3, activation='softmax', kernel_regularizer=regularizers.l1(lambda_l1))`

- **Hyperparameters**
  - L1 penalty `lambda_l1` set in the code
  - Uses fixed learning rate, epochs, and batch size

- **Training and evaluation**
  - Trains on `X_train_std`, validates on `X_valid_std`
  - Plots:
    - Training vs validation accuracy over epochs
    - Training vs validation loss over epochs
  - Evaluates test accuracy and compares it to:
    - Baseline (no regularization)
    - L2-regularized model

---

## Libraries Used

- Python 3.x
- Google Colab + Google Drive
- `pandas`, `numpy`
- `matplotlib`
- `scikit-learn`
  - `train_test_split`
  - `StandardScaler`
  - Metrics: `accuracy_score`, `confusion_matrix`, `classification_report`
- `tensorflow` / `keras`
  - `Sequential`, `Dense`, `regularizers`

---

## How to Run

1. Open the notebook in **Google Colab** or local Jupyter.
2. Make sure `seeds(in).csv` is available at the path used in `pd.read_csv(...)`, or update the path accordingly.
3. Install the required packages if needed:
   ```bash
   pip install tensorflow scikit-learn pandas numpy matplotlib
