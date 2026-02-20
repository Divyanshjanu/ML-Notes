# 🔷 Support Vector Machine – Red Wine Quality

## 📌 Overview

This project builds a complete **Support Vector Machine (SVM)** pipeline to predict red wine quality as a **binary classification** task (**bad** vs **good**). It covers:

- 📊 Exploratory Data Analysis (EDA)  
- 🧹 Targeted outlier removal  
- ⚙️ Feature scaling and preprocessing  
- 🔎 Hyperparameter tuning (GridSearchCV & RandomizedSearchCV)  
- ✅ Evaluation with classification reports and confusion matrices  

The goal is a clean, well‑documented example of an end‑to‑end ML workflow.

---

## 🎯 Problem & Dataset

- Original label `quality`: integers 3–8  
- Binary target used here:

  - `0` → **bad** (scores 2–6.5)  
  - `1` → **good** (scores 6.5–8)

- File: `winequality-red.csv` (UCI Wine Quality)  
- Samples: 1599  
- Numeric features (11):

  `fixed acidity`, `volatile acidity`, `citric acid`, `residual sugar`,  
  `chlorides`, `free sulfur dioxide`, `total sulfur dioxide`, `density`,  
  `pH`, `sulphates`, `alcohol`

EDA includes:

- `shape`, `info()`, `describe()`, `isnull().sum()`  
- `quality.value_counts()` to inspect imbalance  
- Correlation heatmap, feature histograms, and boxplots for outliers

---

## 🧪 Methodology (Pipeline)

1️⃣ **EDA**  
Inspect structure, correlations, distributions, and potential outliers.

2️⃣ **Outlier Removal (IQR)**  
Applied only to:

- `residual sugar`, `chlorides`, `free sulfur dioxide`, `total sulfur dioxide`

For each column:

- Compute Q1, Q3, IQR = Q3 − Q1  
- Keep rows with values in  

  \[
  [Q1 - 2.0 \times IQR,\; Q3 + 2.0 \times IQR]
  \]

Result: cleaned dataframe `df_clean` with extreme outliers reduced but most data preserved.

3️⃣ **Target Transformation**

- Ensure `quality` is numeric and drop invalid entries.  
- Use `pd.cut` to map scores into `bad` / `good`.  
- Use `LabelEncoder` so `bad → 0`, `good → 1`.

4️⃣ **Feature / Target Split & Scaling**

- `X`: all 11 numeric features  
- `y`: encoded binary `quality`  

Train–test split:

- 80% train, 20% test  
- `stratify=y`, fixed `random_state`

Scaling:

- Fit `StandardScaler` on `X_train`  
- Transform both `X_train` and `X_test` (no leakage)

5️⃣ **Baseline SVM**

- Train SVC with `class_weight='balanced'` on scaled data.  
- Use test accuracy as baseline before tuning.

6️⃣ **Hyperparameter Tuning**

Parameters explored:

- `C` ∈ {0.1, 1, 10}  
- `kernel` ∈ {`linear`, `rbf`}  
- `gamma` ∈ {`scale`, `auto`, numeric}

- **GridSearchCV** (5‑fold CV) → `best_model_1`  
- **RandomizedSearchCV** (5‑fold CV) → `best_model_2`

7️⃣ **Evaluation**

For each best model:

- Predict on `X_test`  
- Compute accuracy and classification report  
- Compute and plot confusion matrix (heatmap)

Typical performance (approx.):

- Test accuracy ≈ **90–91%**  
- Class 0 (bad): very high precision & recall  
- Class 1 (good): high precision, moderate recall (class imbalance)

Train and test scores are close → no strong overfitting.

---

## 🧠 How SVM Works (Short Math Intuition)

SVM tries to learn a decision boundary (hyperplane):

\[
w^\top x + b = 0
\]

A point is classified by the sign of \(w^\top x + b\).

### Maximum margin idea

For linearly separable data, SVM finds the hyperplane with the **largest margin** to the closest points (support vectors):

\[
\min_{w, b} \frac{1}{2} \lVert w \rVert^2
\]

subject to

\[
y_i (w^\top x_i + b) \ge 1
\]

### Soft margin and \(C\)

Real data is noisy, so SVM allows margin violations using slack variables \(\xi_i\) and a penalty parameter \(C\):

\[
\min_{w, b, \xi} \frac{1}{2} \lVert w \rVert^2 + C \sum_i \xi_i
\]

Larger \(C\) → less tolerance to errors (tighter fit).  
Smaller \(C\) → more tolerance (stronger regularization).  
In this project, \(C\) is tuned during Grid/Random search.

### Kernels & \(\gamma\)

For non‑linear patterns, SVM uses kernels, e.g. RBF:

\[
K(x_i, x_j) = \exp(-\gamma \lVert x_i - x_j \rVert^2)
\]

- Small \(\gamma\): smoother, more global decision boundary.  
- Large \(\gamma\): more complex, risk of overfitting.

The best models here use an RBF kernel with tuned `C` and `gamma`.

### Class weights

With imbalanced labels, `class_weight='balanced'` automatically scales penalties so mistakes on the minority class (good wines) matter more during training.
