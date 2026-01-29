# Linear Regression – Notes & Applied Project

This folder contains my learning notes and an end-to-end implementation of
**Linear Regression** using the **California Housing dataset**.  
The project focuses on understanding the mathematical intuition behind the
model and applying it using `scikit-learn`.

---

## Dataset
**California Housing Dataset**
- Target variable: Median House Value
- Features include:
  - Median income
  - House age
  - Average rooms
  - Average bedrooms
  - Population
  - Latitude & Longitude

---

## Mathematical Foundation

### 1. Linear Regression Model
Linear Regression assumes a linear relationship between features and target:

\[
y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_n x_n
\]

Where:
- \(y\) = predicted house price  
- \(x_1, x_2, \dots, x_n\) = input features  
- \(\beta_0\) = intercept  
- \(\beta_i\) = coefficients learned by the model  

The model learns coefficients by **minimizing the error** between actual and
predicted values.

---

### 2. Cost Function (Mean Squared Error)

\[
MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
\]

- Penalizes large errors
- Used internally by Linear Regression during training

---

## Feature Scaling

Before training, features are standardized using **StandardScaler**:

\[
z = \frac{x - \mu}{\sigma}
\]

Where:
- \(\mu\) = mean of feature  
- \(\sigma\) = standard deviation  

This ensures:
- All features are on the same scale
- Faster and more stable model convergence

The fitted scaler is saved as `scaler.pkl`.

---

## Model Training

- Model used: `LinearRegression()` from `scikit-learn`
- Data split into **training** and **testing** sets
- Model learns coefficients using Ordinary Least Squares (OLS)

The trained model is saved as `Model.pkl`.

---

## Regularization

To reduce overfitting, regularization techniques are explored:

### Ridge Regression (L2)
\[
Cost = MSE + \lambda \sum \beta^2
\]

- Shrinks coefficients
- Keeps all features

### Lasso Regression (L1)
\[
Cost = MSE + \lambda \sum |\beta|
\]

- Can reduce some coefficients to zero
- Performs feature selection

---

## Model Evaluation

The model is evaluated using:

### Root Mean Squared Error (RMSE)
\[
RMSE = \sqrt{MSE}
\]

- Measures average prediction error
- Lower RMSE → better model

### R² Score
\[
R^2 = 1 - \frac{SS_{res}}{SS_{tot}}
\]

- Explains how much variance in target is captured
- Value closer to 1 indicates good fit

---

## Files in This Folder
- `Predicting_House_prices.ipynb` → Full implementation
- `Model.pkl` → Trained Linear Regression model
- `scaler.pkl` → StandardScaler used for preprocessing
- `README.md` → Project documentation

---

## Tools & Libraries
- Python
- NumPy
- Pandas
- Matplotlib
- scikit-learn

---

## Key Learning Outcomes
- Understood the math behind Linear Regression
- Applied feature scaling correctly
- Built and evaluated a regression model
- Learned the impact of regularization
- Created a reusable ML pipeline

This folder will be updated as I continue learning Machine Learning.
