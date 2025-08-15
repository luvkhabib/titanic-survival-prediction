# Titanic Survival Prediction: A Machine Learning Approach
[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green)](https://opensource.org/licenses/MIT)

A machine learning project predicting passenger survival on the Titanic disaster using advanced techniques.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Methodology](#methodology)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)

## Project Overview
This project demonstrates:
- A complete ML pipeline from data cleaning to a deployment-ready model.
- Advanced hyperparameter tuning with Optuna.
- Feature importance analysis and model interpretability techniques.
- Visualization of model performance through confusion matrices.

## Features
### Data Preprocessing
- Missing value imputation
- Feature encoding (one-hot, label)
- Feature scaling

### Modeling
- Random Forest with hyperparameter tuning
- Comparison with XGBoost
- Neural Network experiment

### Evaluation
- Metrics: Accuracy, Precision, Recall
- SHAP value interpretation for model insights
- Learning curves to visualize model performance

## Methodology
1. **Data Exploration**: Initial analysis of features and distributions.
2. **Feature Engineering**:
   - Title extraction from names
   - Family size calculation
   - Cabin letter extraction
3. **Model Development**:
   - Hyperparameter tuning using Optuna:
   ```python
   def objective(trial):
       params = {
           'n_estimators': trial.suggest_int('n_estimators', 50, 200),
           'max_depth': trial.suggest_int('max_depth', 1, 30),
           'min_samples_split': trial.suggest_int('min_samples_split', 2, 10)
       }
       model = RandomForestClassifier(**params)
       return cross_val_score(model, X, y, cv=5).mean()
