<img width="788" height="473" alt="Screenshot 2025-08-15 011408" src="https://github.com/user-attachments/assets/cc970504-1b84-4974-8535-0b998b761855" /># Titanic Survival Prediction: A Machine Learning Approach
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


## Results
The model performance was evaluated using various metrics, and the key findings are summarized below:

- **Accuracy Achieved**: The Optuna-tuned RandomForestClassifier achieved an accuracy of **83-85%**, demonstrating a significant improvement over the baseline model, which had an accuracy of approximately **70-73%**.
- **Model Comparison**: 
  - **Random Forest**: Tuned with hyperparameters, showing robust performance.
  - **XGBoost**: Compared against Random Forest, providing insights into model performance differences.
  - **Neural Network**: Experimented with to explore deep learning capabilities.

### Visualizations
- **Confusion Matrix**: Visual representation of model performance, illustrating true positives, true negatives, false positives, and false negatives.
- **Learning Curves**: Graphs showing model performance over training iterations, helping to identify overfitting or underfitting.


<img width="788" height="473" alt="Screenshot 2025-08-15 011408" src="https://github.com/user-attachments/assets/d6754b39-9c75-4b20-9ac5-eeae756df3af" />

## Installation
To install the necessary dependencies, run the following command in your terminal:

``bash
pip install -r requirements.txt

### Usage
To run the project, execute the following command in your terminal:

``bash
python main.py


### Contributing
Contributions are welcome! Please follow these steps to contribute to the project:

# Fork the repository: Click on the "Fork" button at the top right of the repository page.
# Create a new branch:

``bash
git checkout -b feature-branch

# Make your changes: Implement your feature or fix.
# Commit your changes:
bash
git commit -m 'Add new feature'

# Push to the branch:
bash
git push origin feature-branch

# Create a pull request: Go to the original repository and click on "New Pull Request".
