# titanic-survival-prediction
# Titanic Survival Prediction Project

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
- Complete ML pipeline from data cleaning to deployment-ready model
- Advanced hyperparameter tuning with Optuna
- Feature importance analysis
- Model interpretability techniques

<img src="https://placehold.co/600x400?text=Confusion+Matrix+Visualization" alt="Confusion matrix showing model performance" width="400"/>

## Features
- **Data Preprocessing**:
  - Missing value imputation
  - Feature encoding (one-hot, label)
  - Feature scaling
- **Modeling**:
  - Random Forest with hyperparameter tuning
  - XGBoost comparison
  - Neural Network experiment
- **Evaluation**:
  - Accuracy, Precision, Recall metrics
  - SHAP value interpretation
  - Learning curves

## Methodology
1. **Data Exploration**: Initial analysis of features and distributions
2. **Feature Engineering**:
   - Title extraction from names
   - Family size calculation
   - Cabin letter extraction
3. **Model Development**:
   ```python
   def objective(trial):
       params = {
           'n_estimators': trial.suggest_int('n_estimators', 50, 200),
           'max_depth': trial.suggest_int('max_depth', 1, 30),
           'min_samples_split': trial.suggest_int('min_samples_split', 2, 10)
       }
       model = RandomForestClassifier(**params)
       return cross_val_score(model, X, y, cv=5).mean()
