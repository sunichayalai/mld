import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load trained model
model = joblib.load("model.pkl")

# You can update this with your actual best model's metrics
best_model_name = "Balanced Random Forest"
model_metrics = {
    "Accuracy": 0.83,
    "Precision": 0.76,
    "Recall": 0.80,
    "F1 Score": 0.78
}

# Streamlit config
st.set_page_config(page_title="Loan Predictor", page_icon="💰", layout="centered")

# App header
st.markdown("""
    <h1 style='text-align: center; color: #003366;'>💳 Loan Default Risk Predictor</h1>
    <p style='text-align: center;'>Fill in the form to predict if a loan will be approved.</p>
    <hr />
""", unsafe_allow_html=True)

# Input form layout
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 18, 100, value=30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"])
    education = st.selectbox("Education", ["High School", "Bachelor", "Master", "PhD", "Other"])
    employment = st.selectbox("Employment Status", ["Employed", "Self-employed", "Unemployed", "Retired", "Student"])

with col2:
    income = st.number_input("Annual Income ($)", 0, 1_000_000, value=50000)
    loan = st.number_input("Loan Amount Requested ($)", 0, 1_000_000, value=10000)
    purpose = st.selectbox("Purpose of Loan", ["Personal", "Home", "Car", "Education", "Business"])
    credit_score = st.slider("Credit Score", 300, 850, 650)
    existing_loans = st.slider("Existing Loans Count", 0, 10, 1)
    late_payments = st.slider("Late Payments Last Year", 0, 12, 0)

loan_to_income = loan / (income + 1)

# One-hot encoding for 12 features
gender_male = 1 if gender == "Male" else 0
married = 1 if marital == "Married" else 0
edu_master = 1 if education == "Master" else 0
emp_self = 1 if employment == "Self-employed" else 0
purpose_home = 1 if purpose == "Home" else 0

input_data = np.array([[ 
    age,
    income,
    loan,
    credit_score,
    existing_loans,
    late_payments,
    loan_to_income,
    gender_male,
    married,
    edu_master,
    emp_self,
    purpose_home
]])

# Prediction logic
if st.button("📊 Predict Loan Outcome"):
    try:
        prediction = model.predict(input_data)[0]
        confidence = model.predict_proba(input_data)[0][prediction]

        result_text = "✅ Loan Approved" if prediction == 0 else "❌ Loan Not Approved"
        result_color = "green" if prediction == 0 else "red"

        st.markdown(f"<h2 style='color:{result_color}; text-align:center;'>{result_text}</h2>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align:center;'>Confidence: <b>{confidence:.2%}</b></p>", unsafe_allow_html=True)

        # Show model info
        st.markdown("---")
        st.subheader("🧠 Model Info")
        st.markdown(f"**Best Model Used:** `{best_model_name}`")
        st.dataframe(pd.DataFrame(model_metrics, index=["Score"]).T.rename(columns={"Score": "Value"}).round(2))

    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")

st.markdown("---")
st.markdown("<p style='text-align:center;'>© 2025 Loan Predictor | Powered by Streamlit</p>", unsafe_allow_html=True)
