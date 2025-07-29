import streamlit as st
import numpy as np
import joblib

# Load model (match the filename you used when saving)
model = joblib.load('model.pkl')  # change if needed

# Features must match training time exactly
st.set_page_config(page_title="Loan Default Predictor", layout="centered")
st.markdown("<h1 style='text-align: center; color: navy;'>Loan Default Risk Prediction</h1>", unsafe_allow_html=True)
st.markdown("### Applicant Details")

# Inputs
age = st.number_input("Age", min_value=18, max_value=100)
gender = st.selectbox("Gender", ["Male", "Female"])
marital = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"])
education = st.selectbox("Education", ["High School", "Bachelor", "Master", "PhD", "Other"])
employment = st.selectbox("Employment Status", ["Employed", "Self-employed", "Unemployed", "Retired", "Student"])
income = st.number_input("Annual Income ($)", min_value=0)
loan = st.number_input("Loan Amount Requested ($)", min_value=0)
purpose = st.selectbox("Purpose of Loan", ["Personal", "Home", "Car", "Education", "Business"])
credit_score = st.slider("Credit Score", 300, 850)
existing_loans = st.slider("Existing Loans Count", 0, 10)
late_payments = st.slider("Late Payments Last Year", 0, 12)

loan_to_income = loan / (income + 1)

# Manual encoding (must match training time!)
gender_male = 1 if gender == "Male" else 0
married = 1 if marital == "Married" else 0
single = 1 if marital == "Single" else 0
widowed = 1 if marital == "Widowed" else 0
divorced = 1 if marital == "Divorced" else 0
edu_master = 1 if education == "Master" else 0
edu_other = 1 if education == "Other" else 0
edu_phd = 1 if education == "PhD" else 0
emp_self = 1 if employment == "Self-employed" else 0
emp_unemp = 1 if employment == "Unemployed" else 0
emp_retired = 1 if employment == "Retired" else 0
emp_student = 1 if employment == "Student" else 0
purpose_home = 1 if purpose == "Home" else 0
purpose_personal = 1 if purpose == "Personal" else 0
purpose_bus = 1 if purpose == "Business" else 0

# This must match model training feature order
input_data = np.array([[ 
    age, income, loan, credit_score, existing_loans, late_payments, loan_to_income,
    gender_male,
    married, single, widowed, divorced,
    edu_master, edu_other, edu_phd,
    emp_self, emp_unemp, emp_retired, emp_student,
    purpose_home, purpose_personal, purpose_bus
]])

if st.button("Predict Loan Approval"):
    try:
        prediction = model.predict(input_data)[0]
        result = "✅ Loan Approved" if prediction == 0 else "❌ Loan Not Approved"
        color = "green" if prediction == 0 else "red"
        st.markdown(f"<h3 style='color:{color}; text-align:center;'>{result}</h3>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")

st.markdown("---")
st.info("Ensure all applicant details are accurate before submitting.")
