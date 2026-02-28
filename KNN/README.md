# 🩺 Diabetes Prediction using KNN


> An end-to-end machine learning project for diabetes prediction using K-Nearest Neighbors. Part of my **ML Notes & Projects** series.

---

## 📋 Dataset

| Property | Details |
|:---:|:---:|
| 📦 Name | Pima Indians Diabetes Dataset |
| 👥 Samples | 768 patients |
| 🔢 Features | 8 (Glucose, BMI, Age, etc.) |
| 🎯 Target | 0 = No Diabetes / 1 = Diabetes |

---

## 🧠 KNN — Theory & Math

> KNN classifies a new point by **majority vote** of its K nearest neighbours.

**📐 Euclidean Distance**

$$d(x, y) = \sqrt{\sum_{i=1}^{n}(x_i - y_i)^2}$$

**⚖️ StandardScaler** — mandatory for KNN

$$z = \frac{x - \mu}{\sigma}$$

**✂️ IQR Outlier Removal**

$$IQR = Q3 - Q1 \qquad \text{Bounds} = Q1 \pm 1.5 \times IQR$$

**🔁 SMOTE Synthesis**

$$x_{new} = x_i + \lambda \times (x_{nn} - x_i)$$

**📉 K-Tuning Error Rate**

$$\text{Error Rate} = \frac{1}{n}\sum_{i=1}^{n} \mathbb{1}(\hat{y}_i \neq y_i)$$

---

## 🔄 ML Pipeline

| # | Step | Purpose |
|---|------|---------|
| 1 | EDA + Heatmap | Understand feature relationships |
| 2 | Zero Imputation | Fix medically invalid entries |
| 3 | IQR Outlier Removal | Remove noisy data points |
| 4 | StandardScaler | Fair distance calculation |
| 5 | Train/Test Split | Unbiased evaluation |
| 6 | SMOTE | Balance minority class |
| 7 | K-Tuning | Find optimal K value |
| 8 | Evaluation | Measure model performance |

---

## 📈 Results

| Model | Accuracy | Macro F1 |
|:-----:|:--------:|:--------:|
| Default K=5 | 71% | 0.69 |
| ⭐ **Best K=4** | **73%** | **0.71** |

**Confusion Matrix (K=4)**

| | Predicted: 0 | Predicted: 1 |
|--|-------------|-------------|
| **Actual: 0** | 97 ✅ | 30 ❌ |
| **Actual: 1** | 21 ❌ | 43 ✅ |

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} = 73\%$$

$$\text{F1 Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = 0.71$$

---

## 💡 Key Learnings

- ✅ KNN is distance-based — scaling is non-negotiable
- ✅ Zeros in medical data = hidden missing values
- ✅ SMOTE prevents bias towards majority class
- ✅ Glucose & BMI are strongest diabetes predictors
- ✅ K-Tuning improved accuracy from 71% → 73%

---

## 🛠️ Tech Stack

| Library | Purpose |
|---------|---------|
| pandas | Data manipulation |
| numpy | Numerical operations |
| matplotlib | Plotting |
| seaborn | Statistical visualization |
| scikit-learn | KNN, Scaler, Metrics |
| imbalanced-learn | SMOTE |


---

## 📁 Project Structure

    KNN-Diabetes-Prediction/
    │
    ├── 📓 KNN.ipynb       ← Main notebook
    ├── 📊 diabetes.csv    ← Dataset
    └── 📄 README.md       ← Documentation

---

<div align="center">

## 👤 Author

**Divyansh Janu**
*Aspiring ML Engineer | ML Notes & Projects*

*⭐ Star this repo if you found it helpful!*

</div>
