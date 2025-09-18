# End-To-End-Predictive-Pipeline-Churn-Prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

The primary objective of this project is to determine which model (Logistic Regression, Decision Trees, Neural Networks, etc.) is most effective for predicting customer churn and to build an end-to-end predictive pipeline.

This project has 2 parts-
- First part reimplements a research paper that helps us to identify the best model for the prediction.
- The second part focuses on building an end-to-end predictive pipeline for Churn prediction.
---
**Why is customer churn a major factor in various industries?**

Customer churn impacts revenue and loyalty, as retaining customers is cheaper than acquiring new ones. Predicting churn helps companies take targeted actions in competitive markets.

## First Part of the project-

- In this part we have trained 4 models- Logistic Regression, Gradient Boosting Trees, Random Forest and Neural Network.
- <img width="819" height="416" alt="image" src="https://github.com/user-attachments/assets/c2cf4f11-3542-41a3-8504-89143a3c7669" />
- We can see the metrics used to evaluate the models-
- 1. Log loss(log): Lower is better.
  2. AUC (auc): Area under the ROC curve, which measures how well the model separates churn vs non-churn. Higher is better.
  3. Accuracy (accuracy): Percentage of correct predictions.
- Interpretation of Results-
- 1. Log Loss - Gradient Boosting has the lowest log loss, meaning its probabilities are most reliable.
  2. AUC - Random Forest and Neural Network show the highest AUC values (~0.85+), which means they’re the best at distinguishing churners from non-churners.
  3. Accuracy- Logistic Regression lags a little behind the other 3.
- We also tested using the Kolmogorov-Smirnov Statistic and the Churn Detection Rate Statistic.
- For upsampling of the Logistic Regression model, we used SMOTE(Synthetic Minority Oversampling Technique), which creates synthetic samples by interpolating b/w existing minority class points.
- Conclusion-
- Simple Logistic Regression (LR) showed a ~78% accuracy, Stacked models (ensemble) increased the accuracy to ~79% and Logistic Regression with Upsampling was found to be the best i.e ~81% accuracy.

---

## Second Part of the project-

- The prediction pipeline begins with preprocessing raw customer data, handling missing values, and engineering features such as average charges per month. To address class imbalance, the project applies SMOTE (Synthetic Minority Oversampling Technique) before training a Logistic Regression model.

- Once trained, the model and preprocessing pipeline are saved for deployment. The Streamlit application allows both bulk predictions via CSV uploads and single-customer predictions through a manual form. The app outputs churn predictions along with probabilities and supports downloading the enriched dataset. This makes it useful both for data analysts validating churn drivers and for business teams to proactively engage with at-risk customers.

**Data preprocessing & cleaning-**
1. Handles missing values and converts columns (e.g., TotalCharges to numeric).
2. Replaces redundant categories like "No internet service" → "No".
3. Engineers' new feature: Average Charges Per Month.

**Model training (train.py)-**
1. Uses Logistic Regression with SMOTE to balance churn classes.
2. Automatic detection of categorical & numeric features.
3. Preprocessing pipelines with OneHotEncoding (categorical) and StandardScaler (numeric).
4. Saves trained artifacts (preprocessor.pkl, model.pkl, feature names, defaults).
5. Prints classification report and ROC AUC score for evaluation.

**Interactive web app (app.py)-**
1. Built with Streamlit.
2. Two input modes:
Upload CSV (bulk predictions with preview & CSV download).
Manual single-row form (enter customer details to predict churn).
3. Displays predictions with probabilities.
4. Downloadable predictions as CSV.

**Outputs-**
1. Predicted churn label (Yes/No).
2. Churn probability score.3.
3. CSV download of full dataset with predictions.

---

## Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/AnshumanNigam/End-To-End-Predictive-Pipeline-Churn-Prediction.git
cd End-To-End-Predictive-Pipeline-Churn-Prediction
pip install -r requirements.txt
