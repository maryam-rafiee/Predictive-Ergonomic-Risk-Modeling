# Predictive Ergonomic Risk Modeling

## Overview

This project uses the UCI Human Activity Recognition (HAR) dataset to build a machine learning model that classifies human activities based on smartphone sensor data.

The project demonstrates a complete machine learning workflow, including:

- Data loading
- Data preprocessing
- Model training
- Hyperparameter tuning
- Model evaluation
- Feature importance analysis
- Model serialization

---

## Dataset

Dataset:
UCI Human Activity Recognition Using Smartphones Dataset

Number of training samples: 7352

Number of testing samples: 2947

Number of features: 561

Number of activity classes: 6

---

## Machine Learning Model

Algorithm:

- Random Forest Classifier

Hyperparameter Optimization:

- GridSearchCV

Evaluation Metrics:

- Accuracy
- Classification Report
- Feature Importance

---

## Results

Best Accuracy:

92.2%

Top 10 most important features were visualized using a bar chart.

The trained model was saved as:

results/random_forest_model.pkl

---

## Project Structure

```
Predictive-Ergonomic-Risk-Modelling/

├── data/
├── figures/
├── notebooks/
├── results/
├── src/
│ ├── load_data.py
│ └── train_model.py
├── requirements.txt
└── README.md
```

---

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib
- Git
- GitHub

---

## Author

Maryam Rafiee