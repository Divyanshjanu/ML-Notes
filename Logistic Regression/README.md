# Logistic Regression – Diabetes Prediction

This folder contains an end‑to‑end binary classification project using **Logistic Regression** to predict whether a person has diabetes based on basic medical measurements.

## Project overview

The goal is to predict the target variable **Outcome** (0 = non‑diabetic, 1 = diabetic) using features such as:

- Pregnancies  
- Glucose  
- BloodPressure  
- SkinThickness  
- Insulin  
- BMI  
- DiabetesPedigreeFunction  
- Age  

This project focuses on building a clean preprocessing pipeline, dealing with class imbalance, and understanding how Logistic Regression separates the two classes.

---

## Files in this folder

- `Predictive_Analysis_in_Diabetes.ipynb` – main notebook with EDA, preprocessing, model training, and evaluation.  
- `diabetes.csv` – Pima Indians Diabetes dataset used in this project.  
- `corr_coeff_heatmap.png` – correlation heatmap generated during exploratory analysis.  
- `diabetes_logistic_model.pkl` – trained Logistic Regression model saved for reuse.  
- `diabetes_scaler.pkl` – fitted `StandardScaler` used to scale features before prediction.

---

## Mathematical intuition of Logistic Regression

### From linear model to probability

Logistic Regression starts with a linear combination of the input features:

\[
z = w_0 + w_1 x_1 + w_2 x_2 + \dots + w_n x_n
\]

Instead of predicting \(z\) directly, we pass it through the **sigmoid function**:

\[
\sigma(z) = \frac{1}{1 + e^{-z}}
\]

This maps any real value to the interval \((0, 1)\), which we interpret as the estimated probability:

\[
\hat{p}(y = 1 \mid x) = \sigma(z)
\]

During prediction, a threshold (usually 0.5) is applied: if \(\hat{p} \geq 0.5\), the model predicts class 1 (diabetic), otherwise class 0 (non‑diabetic).

### Odds and log‑odds

Probabilities can also be written as **odds**:

\[
\text{odds} = \frac{p}{1 - p}
\]

Logistic Regression assumes that the **log‑odds** are a linear function of the inputs:

\[
\log\left(\frac{p}{1 - p}\right) = w_0 + w_1 x_1 + \dots + w_n x_n
\]

Each weight \(w_j\) tells us how much the log‑odds change when feature \(x_j\) increases by one unit. In the diabetes setting, a positive weight for a feature (for example, Glucose or BMI) means higher values of that feature increase the odds of being diabetic.

### Loss function and learning

The model is trained by minimizing the **binary cross‑entropy (log loss)**:

\[
L = -\frac{1}{m} \sum_{i=1}^{m} \Big[ y^{(i)} \log(\hat{p}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{p}^{(i)}) \Big]
\]

- If the model assigns low probability to the true class, the loss is high.  
- If the predicted probabilities are close to the true labels, the loss is low.

The weights \(w\) are updated using gradient‑based optimization (such as gradient descent) to minimize this loss and learn good probability estimates.

### Decision boundary

Logistic Regression learns a **linear decision boundary** defined by:

\[
w_0 + w_1 x_1 + \dots + w_n x_n = 0
\]

Points on one side of this boundary are classified as non‑diabetic and points on the other side as diabetic. When the classes are approximately linearly separable in feature space, Logistic Regression is a strong and interpretable baseline model.

---

## Project workflow

The notebook follows this end‑to‑end workflow:

1. Load the diabetes dataset and inspect basic statistics and data structure.  
2. Clean medically invalid zero values in features such as Glucose, BloodPressure, SkinThickness, Insulin, and BMI.  
3. Perform exploratory data analysis, including distributions, boxplots, and a correlation heatmap.  
4. Split the data into training and test sets.  
5. Scale numerical features using `StandardScaler`.  
6. Address class imbalance using **SMOTE** on the training data only.  
7. Train a Logistic Regression model on the scaled, balanced training data.  
8. Evaluate performance using accuracy, confusion matrix, and a classification report (precision, recall, F1‑score).  
9. Save the trained model and scaler as `.pkl` files for future inference.

---

## Learning focus

This project is part of my ongoing effort to document core Machine Learning algorithms topic‑by‑topic. In this notebook I focused on:

- Building intuition for Logistic Regression as a probabilistic linear classifier.  
- Practising data cleaning, scaling, and handling class imbalance.  
- Evaluating classification models using multiple metrics, not just accuracy.  
- Saving models and preprocessing steps for reuse in future experiments.
