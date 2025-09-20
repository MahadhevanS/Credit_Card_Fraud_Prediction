# Credit Card Fraud Prediction
  This project demonstrates a credit card fraud detection system using machine learning. It's a classic example of an anomaly detection problem where a highly imbalanced dataset is used to train a model to identify fraudulent transactions.

## Steps involved
- Exploratory Data Analysis (EDA) to understand the dataset's characteristics, such as feature distributions and class imbalance.

- Application of three unsupervised machine learning algorithms for outlier detection:
  - Isolation Forest
  - Local Outlier Factor (LOF)
  - One-Class SVM

- Evaluation of the models' performance using accuracy scores and classification reports.

- Saving the best-performing model for future use.

## Dataset
The dataset used is the Credit Card Fraud Detection (dataset)[https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud], which is publicly available on Kaggle.

## Installation
To run the app locally or to replicate and analyse the model follow the steps below

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```
