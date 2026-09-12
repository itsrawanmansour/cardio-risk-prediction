# ❤️ Cardiovascular Risk Prediction

A machine learning and deep learning project analyzing risk factors and building predictive models for cardiovascular disease (CVD), comparing ensemble methods against a neural network on 70,000 patient records.

## Overview
Cardiovascular diseases are the leading cause of mortality worldwide. This project explores how machine learning can improve on traditional risk scores (like Framingham or QRISK) by capturing non-linear relationships between clinical and lifestyle risk factors.

The project covers:

* **Preprocessing** — missing value imputation (SimpleImputer, KNNImputer), normalization (StandardScaler), outlier handling
* **Feature Engineering** — derived features such as BMI, pulse pressure, and mean arterial pressure (MAP)
* **Class Balancing** — oversampling with SMOTE to address class imbalance
* **Modeling** — Random Forest, Gradient Boosting, Extra Trees Classifier, and a Neural Network
* **Evaluation** — accuracy, precision, recall, F1-score, confusion matrix, AUC-ROC
* **Risk Factor Analysis** — visualizing relationships between age, BMI, blood pressure, glucose, and CVD

## Dataset
`cardio_train.csv` — 70,000 patient records with clinical and behavioral attributes, including:

* Age, gender, height, weight
* Systolic/diastolic blood pressure (ap_hi, ap_lo)
* Cholesterol and glucose levels
* Smoking, alcohol intake, physical activity
* Target: presence of cardiovascular disease (`cardio`)

Source: https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset

## Results

| Model | Accuracy |
|---|---|
| **Random Forest** | **96.9%** |
| Extra Trees Classifier | 95.7% |
| Neural Network | 86.8% (test) |
| Gradient Boosting | 78.6% |

* **Random Forest** and **Extra Trees Classifier** were the strongest and most consistent performers across accuracy, precision, recall, and F1-score.
* The **Neural Network** showed solid, well-balanced performance after training on scaled and engineered features.
* **Gradient Boosting** lagged behind the other models in this setup.
* Feature importance analysis identified **age, systolic blood pressure, cholesterol, and BMI** as the most significant predictors of CVD, with smoking and alcohol use contributing to a lesser extent.

## Tech Stack

* **Language**: Python
* **Libraries**: `scikit-learn`, `tensorflow`/`keras`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`, `imbalanced-learn` (SMOTE)

## Techniques Implemented

* Missing value imputation (`SimpleImputer`, `KNNImputer`)
* Feature engineering: BMI, pulse pressure, mean arterial pressure
* Box-Cox transformation for skewed features
* SMOTE oversampling for class imbalance
* One-hot encoding of categorical features (cholesterol, glucose)
* Feature scaling with `StandardScaler`
* Ensemble classifiers: Random Forest, Gradient Boosting, Extra Trees
* Neural network with dense layers and dropout regularization
* Model evaluation: accuracy, precision, recall, F1-score, confusion matrix, AUC-ROC
* Interactive prediction function for new patient data

## Project Structure

```
cardio-risk-prediction/
├── notebooks/
│   └── cardio_risk_prediction.ipynb   # Full analysis & modeling notebook (Python)
└── data/
    └── cardio_train.csv                # Source patient dataset
```

## How to Use

1. Open the notebook in Jupyter or Google Colab:

```
jupyter notebook notebooks/cardio_risk_prediction.ipynb
```

2. Make sure `cardio_train.csv` is available in the working directory (update the file path if needed, e.g. `data/cardio_train.csv`). Note the CSV uses `;` as a separator:

```python
data = pd.read_csv("data/cardio_train.csv", sep=";")
```

3. Run the cells in order — preprocessing and feature engineering, model training/evaluation for the ensemble methods, then the neural network section, followed by risk factor visualizations and the interactive prediction function.

## Author
Rawan Mansour

This was a team project.
