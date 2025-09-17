# app.py
import os
import streamlit as st
import pandas as pd
import joblib
import json
import numpy as np

# ---------------------- Page config ----------------------
st.set_page_config(page_title="Churn Predictor", layout="wide")
st.title("Customer Churn Prediction (Streamlit)")

# ---------------------- Paths ----------------------
BASE_DIR = os.path.dirname(__file__)  # directory where app.py resides
preprocessor_path = os.path.join(BASE_DIR, "models", "preprocessor.pkl")
model_path = os.path.join(BASE_DIR, "models", "model.pkl")
feature_names_path = os.path.join(BASE_DIR, "models", "feature_names.json")
defaults_path = os.path.join(BASE_DIR, "models", "feature_defaults.json")

# ---------------------- Load artifacts ----------------------
try:
    preprocessor = joblib.load(preprocessor_path)
    model = joblib.load(model_path)
    with open(feature_names_path) as f:
        feature_names = json.load(f)
    with open(defaults_path) as f:
        defaults = json.load(f)
except FileNotFoundError as e:
    st.error(f"Required file not found: {e}")
    st.stop()
except Exception as e:
    st.error(f"Error loading model artifacts: {e}")
    st.stop()

# ---------------------- Sidebar input mode ----------------------
st.sidebar.header("Input mode")
mode = st.sidebar.radio("Choose", ["Upload CSV (recommended)", "Manual single-row"])

# ---------------------- CSV upload mode ----------------------
if mode == "Upload CSV (recommended)":
    uploaded_file = st.file_uploader("Upload CSV with original raw columns (same as training data)", type=["csv"])
    if uploaded_file is not None:
        try:
            raw = pd.read_csv(uploaded_file)
            raw['TotalCharges'] = pd.to_numeric(raw['TotalCharges'], errors='coerce')
            raw['tenure'] = pd.to_numeric(raw['tenure'], errors='coerce')

            # engineer feature
            raw['AvgChargesPerMonth'] = raw['TotalCharges'] / (raw['tenure'] + 1)
            st.write("Preview of uploaded data:", raw.head())

            missing = [c for c in feature_names if c not in raw.columns]
            if missing:
                st.error(f"Uploaded CSV is missing required columns (first few): {missing[:5]}. Please upload correct file.")
            else:
                X_to_pred = raw[feature_names]
                X_prep = preprocessor.transform(X_to_pred)
                preds = model.predict(X_prep)
                probs = model.predict_proba(X_prep)[:, 1]
                raw['Churn_Pred'] = preds
                raw['Churn_Prob'] = probs
                st.write(raw[[*feature_names[:6], 'Churn_Pred', 'Churn_Prob']].head())  # small preview
                csv = raw.to_csv(index=False).encode('utf-8')
                st.download_button("Download predictions CSV", data=csv, file_name="predictions.csv")
        except Exception as e:
            st.error(f"Error processing uploaded CSV: {e}")

# ---------------------- Manual single-row mode ----------------------
else:
    st.write("Fill the form below and click Predict")
    with st.form("manual_form"):
        tenure = st.number_input("tenure (months)", value=int(defaults.get('tenure', 12)))
        MonthlyCharges = st.number_input("MonthlyCharges", value=float(defaults.get('MonthlyCharges', 70.0)))
        TotalCharges = st.number_input("TotalCharges", value=float(defaults.get('TotalCharges', MonthlyCharges*tenure)))
        Contract = st.selectbox("Contract", options=["Month-to-month", "One year", "Two year"], index=0)
        InternetService = st.selectbox("InternetService", options=["DSL", "Fiber optic", "No"], index=0)
        PaymentMethod = st.selectbox(
            "PaymentMethod",
            options=["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
            index=0
        )
        submit = st.form_submit_button("Predict")

    if submit:
        try:
            row = {c: defaults[c] for c in feature_names}  # start from defaults
            row['tenure'] = int(tenure)
            row['MonthlyCharges'] = float(MonthlyCharges)
            row['TotalCharges'] = float(TotalCharges)
            if 'Contract' in feature_names: row['Contract'] = Contract
            if 'InternetService' in feature_names: row['InternetService'] = InternetService
            if 'PaymentMethod' in feature_names: row['PaymentMethod'] = PaymentMethod

            df_row = pd.DataFrame([row])[feature_names]
            X_prep = preprocessor.transform(df_row)
            pred = model.predict(X_prep)[0]
            prob = model.predict_proba(X_prep)[:,1][0]
            st.metric("Prediction", "Churn" if pred==1 else "No churn", delta=f"Prob: {prob:.2f}")
        except Exception as e:
            st.error(f"Error making prediction: {e}")
