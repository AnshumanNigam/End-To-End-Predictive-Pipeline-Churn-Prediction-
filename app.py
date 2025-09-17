# app.py
import streamlit as st
import pandas as pd
import joblib
import json
import numpy as np

st.set_page_config(page_title="Churn Predictor", layout="wide")
st.title("Customer Churn Prediction (Streamlit)")

# Load artifacts
preprocessor = joblib.load("models/preprocessor.pkl")
model = joblib.load("models/model.pkl")
with open("models/feature_names.json") as f:
    feature_names = json.load(f)
with open("models/feature_defaults.json") as f:
    defaults = json.load(f)

st.sidebar.header("Input mode")
mode = st.sidebar.radio("Choose", ["Upload CSV (recommended)", "Manual single-row"])

if mode == "Upload CSV (recommended)":
    uploaded_file = st.file_uploader("Upload CSV with original raw columns (same as training data)", type=["csv"])
    if uploaded_file is not None:
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
            probs = model.predict_proba(X_prep)[:,1]
            raw['Churn_Pred'] = preds
            raw['Churn_Prob'] = probs
            st.write(raw[[*feature_names[:6], 'Churn_Pred','Churn_Prob']].head())  # small preview
            csv = raw.to_csv(index=False).encode('utf-8')
            st.download_button("Download predictions CSV", data=csv, file_name="predictions.csv")
else:
    st.write("Fill the form below and click Predict")
    # Build a simple form using defaults; we won't create UI for every column — show common ones + rest use defaults
    with st.form("manual_form"):
        # common fields (change if you want)
        tenure = st.number_input("tenure (months)", value=int(defaults.get('tenure', 12)))
        MonthlyCharges = st.number_input("MonthlyCharges", value=float(defaults.get('MonthlyCharges', 70.0)))
        TotalCharges = st.number_input("TotalCharges", value=float(defaults.get('TotalCharges', MonthlyCharges*tenure)))
        Contract = st.selectbox("Contract", options=["Month-to-month","One year","Two year"], index=0)
        InternetService = st.selectbox("InternetService", options=["DSL","Fiber optic","No"], index=0)
        PaymentMethod = st.selectbox("PaymentMethod", options=["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"], index=0)
        submit = st.form_submit_button("Predict")

    if submit:
        # create single-row raw dataframe using defaults and override with user inputs
        row = {c: defaults[c] for c in feature_names}
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
