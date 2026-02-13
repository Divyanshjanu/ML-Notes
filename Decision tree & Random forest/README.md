# Wine Quality Prediction (Red Wine) – Decision Tree & Random Forest

## Project Overview
This project predicts the quality of red wine using tree-based machine learning models, specifically **Decision Tree** and **Random Forest** classifiers.  
The notebook is written as learning notes so that readers can follow each step and understand how these models work in practice.

## Objectives
- Understand the intuition behind Decision Trees and Random Forests.  
- Train, evaluate, and visualise tree-based models using scikit-learn.  
- Apply hyperparameter tuning to improve model performance.  

---

## Dataset Information
- **Name:** Red Wine Quality Dataset  
- **Source:** UCI Machine Learning Repository [web:4]  
- **Samples:** 1,599 red wine records [web:4]  
- **Total Columns:** 12 (11 input features + 1 target label) [web:4][web:9]  
- **Features (inputs):**  
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
- **Target (output):**  
  - `quality` (integer quality score, e.g., 3–8)  

This is a **multi-class classification** problem: given the 11 physicochemical features, the task is to predict the wine’s quality score. [web:4]

---

## Quick Theory

### Decision Tree
- A Decision Tree splits the data by asking a sequence of feature-based questions (e.g., “alcohol > 10.5?”).  
- At each node, it chooses the split that best separates the classes using criteria like **Gini impurity** or **entropy**.  
- Leaf nodes represent final class predictions.  
- **Advantages:** easy to visualise, simple to explain, handles non-linear relationships.  
- **Limitations:** can overfit when the tree is very deep or not regularised.  

### Random Forest
- A Random Forest is an **ensemble** of many Decision Trees.  
- Each tree is trained on a random subset of rows (bootstrap sampling) and a random subset of features.  
- Final prediction is made by **majority voting** across all trees.  
- **Advantages:** better generalisation, less overfitting, more robust than a single tree.  
- **Limitations:** less interpretable than one tree, requires more computation.  

---

## Project Structure

```text
.
├── Practical_Implementation_of_Decision_Tree_and_Random_forest.ipynb
├── winequality-red.csv
└── README.md
