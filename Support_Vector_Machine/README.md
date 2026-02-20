# 🔷 Support Vector Machine – Red Wine Quality Classification

## 📌 Overview

This project implements a complete **Support Vector Machine (SVM)** pipeline to predict red wine quality as a **binary classification** task (**bad** vs **good**). The notebook walks from raw data to a tuned, class‑weighted SVM model, including:

- 📊 Exploratory Data Analysis (EDA)  
- 🧹 Targeted outlier removal  
- ⚙️ Feature scaling and preprocessing  
- 🔎 Hyperparameter tuning (Grid & Random search)  
- ✅ Final evaluation with classification reports and confusion matrices  

The objective is to build a well‑documented, reproducible ML workflow that generalizes well to unseen wine samples.

---

## 🎯 Problem Definition

The original dataset contains physicochemical properties of red wines and a human‑assigned quality score (integer from 3 to 8). For this project, the `quality` label is transformed into a binary target:

- `0` → **bad** quality wine (scores 2–6.5)  
- `1` → **good** quality wine (scores 6.5–8)  

The task is to learn a classifier that predicts this binary label from the chemical features.

---

## 📂 Dataset

- **Source:** Red wine quality dataset (UCI Wine Quality)  
- **File:** `winequality-red.csv`  
- **Samples:** 1599  
- **Features (11 numeric):**

  - fixed acidity  
  - volatile acidity  
  - citric acid  
  - residual sugar  
  - chlorides  
  - free sulfur dioxide  
  - total sulfur dioxide  
  - density  
  - pH  
  - sulphates  
  - alcohol  

- **Target:**  
  - `quality` (3–8) → transformed to `bad` / `good` → encoded as `0` / `1`

**EDA includes:**

- Basic info: `shape`, `info()`, `describe()`  
- Missing‑value check: `isnull().sum()` (no missing values)  
- Class balance: `quality.value_counts()`  
- Visuals: correlation heatmap, histograms, and boxplots for outlier inspection  

---

## 🧪 Methodology

The notebook follows a clear, step‑by‑step pipeline:

### 1️⃣ Exploratory Data Analysis (EDA)

- Inspect data types and summary statistics for all features.  
- Plot a correlation heatmap to see how features relate to `quality`.  
- Draw histograms to observe distributions, skewness, and potential outliers.  
- Use boxplots on the original data to visually highlight extreme values.

---

### 2️⃣ Outlier Handling

Outliers are removed **only** from features with strong skew or heavy tails:

- `residual sugar`  
- `chlorides`  
- `free sulfur dioxide`  
- `total sulfur dioxide`  

For each selected feature:

- Compute **Q1**, **Q3**, and **IQR**  
- Define bounds:  
  - **lower** = Q1 − 2.0 × IQR  
  - **upper** = Q3 + 2.0 × IQR  
- Filter rows outside `[lower, upper]`

The cleaned dataframe `df_clean` keeps most samples while reducing the impact of extreme values. Updated boxplots confirm that the most severe outliers are removed without destroying the overall distributions.

---

### 3️⃣ Target Transformation

The `quality` column is converted to a binary target in three steps:

1. Ensure `quality` is numeric and drop any invalid entries.  
2. Use `pd.cut` to map scores into two bins:

   - (2, 6.5] → `bad`  
   - (6.5, 8] → `good`

3. Apply `LabelEncoder`:

   - `bad` → `0`  
   - `good` → `1`  

This turns the original multi‑class problem into a clean binary classification task.

---

### 4️⃣ Feature–Target Split & Scaling

- **Features (`X`):** all 11 numeric physicochemical columns  
- **Target (`y`):** encoded binary `quality` (0 / 1)

Train–test split:

- 80% training, 20% testing  
- `stratify=y` to preserve class ratios  
- Fixed `random_state` for reproducibility  

Scaling:

- Use `StandardScaler`  
- Fit the scaler on `X_train` only  
- Transform both `X_train` and `X_test` with the fitted scaler  

This avoids data leakage and ensures SVM sees features on a comparable scale.

---

### 5️⃣ Baseline SVM Model

A first SVM classifier is trained with:

- `class_weight='balanced'` ⚖️ to handle label imbalance  

The baseline model is trained on the scaled training data and evaluated on the test set to establish a reference accuracy before tuning.

---

### 6️⃣ Hyperparameter Tuning

To improve performance, the following hyperparameters are tuned:

- `C` (regularization strength)  
- `kernel` (`linear`, `rbf`)  
- `gamma` (`scale`, `auto`, and numeric values)

#### 🔷 GridSearchCV

- Exhaustive search over the parameter grid  
- 5‑fold cross‑validation  
- Returns `best_model_1` – the best SVM configuration on the training set

#### 🔹 RandomizedSearchCV (Optional Comparison)

- Random sampling of parameter combinations from the same grid  
- Also uses 5‑fold cross‑validation  
- Returns `best_model_2` – another strong candidate model  

Both tuned models are later compared on the test set.

---

### 7️⃣ Evaluation

For each tuned model:

- Predict on `X_test`  
- Compute:  
  - ✅ Overall accuracy  
  - 📄 Classification report (precision, recall, F1, support)  
  - 🔢 Confusion matrix  

- Plot confusion matrix heatmaps with labelled axes, making it easy to see:

  - True negatives (correctly predicted bad wines)  
  - True positives (correctly predicted good wines)  
  - False positives & false negatives  

**Typical performance (approx.):**

- Test accuracy ≈ **91%**  
- Class `0` (bad): very high precision and recall  
- Class `1` (good): high precision, moderate recall (due to class imbalance)

The model prefers to avoid misclassifying bad wines as good, which is often a reasonable trade‑off.

---

## 📈 Results Summary

- A class‑weighted SVM with an RBF kernel and tuned `C`/`gamma` achieves the best overall performance.  
- Train and test scores are close, suggesting no strong overfitting.  
- The final model serves as a solid baseline for red wine quality prediction and a clean template for SVM workflows on similar tabular datasets.

---

## 📁 Project Structure

Suggested layout for this folder:

```text
Support_Vector_Machine/
├─ SVM.ipynb              # Main notebook with full pipeline
├─ winequality-red.csv    # Dataset (if stored locally)
└─ README.md              # Project documentation
