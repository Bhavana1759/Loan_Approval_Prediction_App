import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import os

st.set_page_config(page_title="Loan Approval Prediction", layout="centered")

# ----------------------
# Load Data Function
# ----------------------
@st.cache_data
def load_data():
    file_name = "credtech_data.csv"
    if not os.path.exists(file_name):
        st.error(f"❌ File '{file_name}' not found. Please place it in the project folder.")
        st.stop()

    try:
        df = pd.read_csv(file_name, sep="\t")  # most likely tab-separated
    except Exception:
        df = pd.read_csv(file_name, sep=None, engine="python")  # auto-detect separator

    # Ensure target column exists
    if "loan_status" not in df.columns:
        df.rename(columns={df.columns[-1]: "loan_status"}, inplace=True)
    return df
# ----------------------
# Preprocess Data
# ----------------------
def preprocess_data(df):
    # Encode categorical columns
    label_encoders = {}
    for col in df.select_dtypes(include=["object"]).columns:
        if col != "loan_status":
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le

    # Encode target separately
    target_encoder = LabelEncoder()
    df["loan_status"] = target_encoder.fit_transform(df["loan_status"])

    X = df.drop("loan_status", axis=1)
    y = df["loan_status"]

    return X, y, target_encoder

# ----------------------
# Train Model
# ----------------------
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

# ----------------------
# Streamlit UI
# ----------------------
def main():
    st.title("💳 Loan Approval Prediction App")
    st.write("Predict loan approval based on applicant details.")

    df = load_data()
    X, y, target_encoder = preprocess_data(df)
    model = train_model(X, y)

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    st.subheader("📝 Enter Applicant Details")

    user_data = {}
    for col in X.columns:
        if df[col].dtype in [np.int64, np.float64]:
            user_data[col] = st.number_input(f"Enter {col}", value=float(df[col].mean()))
        else:
            options = df[col].unique().tolist()
            user_data[col] = st.selectbox(f"Select {col}", options)

    user_df = pd.DataFrame([user_data])

    # Ensure same column order
    user_df = user_df[X.columns]

    if st.button("Predict Loan Status"):
        prediction = model.predict(user_df)[0]
        prediction_proba = model.predict_proba(user_df)[0]

        result = target_encoder.inverse_transform([prediction])[0]
        st.success(f"✅ Prediction: **{result}**")
        st.info(f"📈 Probability: {prediction_proba.max():.2f}")

if __name__ == "__main__":
    main()
