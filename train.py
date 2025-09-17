# train.py
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE

# 1. Load
df = pd.read_csv("teleco-churn.csv")

# 2. Lightweight cleaning / feature engineering
if 'customerID' in df.columns:
    df = df.drop('customerID', axis=1)

# convert TotalCharges to numeric and fill missing
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# Example engineered feature (optional)
df['AvgChargesPerMonth'] = df['TotalCharges'] / (df['tenure'] + 1)

# Replace obvious "No internet service" kind of labels with "No"
replace_cols = ['OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport',
                'StreamingTV','StreamingMovies','MultipleLines']
for col in replace_cols:
    if col in df.columns:
        df[col] = df[col].replace({'No internet service':'No', 'No phone service':'No'})

# 3. Target
df = df.dropna(subset=['Churn'])   # just in case
y = df['Churn'].map({'No':0,'Yes':1})

# 4. Features
X = df.drop('Churn', axis=1)

# 5. Identify numeric and categorical feature lists (auto)
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

print("Numeric features:", numeric_features)
print("Categorical features:", categorical_features)

# 6. Preprocessing pipelines
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
], remainder='drop')

# 7. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 8. Fit preprocessor and transform training data
X_train_prep = preprocessor.fit_transform(X_train)
X_test_prep = preprocessor.transform(X_test)

# 9. Apply SMOTE on preprocessed features
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train_prep, y_train)

# 10. Train classifier (Logistic Regression as in paper)
clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X_train_res, y_train_res)

# 11. Evaluate
y_pred = clf.predict(X_test_prep)
y_proba = clf.predict_proba(X_test_prep)[:,1]
print("Classification report:\n", classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_proba))

# 12. Save artifacts
import os
os.makedirs("models", exist_ok=True)
joblib.dump(preprocessor, "models/preprocessor.pkl")
joblib.dump(clf, "models/model.pkl")

# Save original feature list (order matters for manual input template)
feature_names = X.columns.tolist()
with open("models/feature_names.json", "w") as f:
    json.dump(feature_names, f)

# Save defaults (median for numeric, mode for categorical) - used by the app for single-row manual form
defaults = {}
for c in feature_names:
    if c in numeric_features:
        defaults[c] = float(X[c].median())
    else:
        defaults[c] = str(X[c].mode().iloc[0])
with open("models/feature_defaults.json", "w") as f:
    json.dump(defaults, f)

print("Saved preprocessor, model, feature names and defaults to 'models/'")
