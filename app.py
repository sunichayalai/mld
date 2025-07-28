import streamlit as st
import numpy as np
import joblib

model = joblib.load("model.pkl")

st.set_page_config(page_title="Loan Default Predictor", layout="centered")
st.markdown("<h1 style='text-align: center; color: navy;'>Loan Default Risk Prediction</h1>", unsafe_allow_html=True)

st.write("Enter the applicant's details below:")

# Collect inputs
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

# Derived features
loan_to_income = loan / (income + 1)
debt_to_income = loan / (income + 1)

# Manual one-hot encoding (as in notebook)
gender_male = 1 if gender == "Male" else 0
gender_female = 1 if gender == "Female" else 0
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

# Final input vector (must match training order!)
input_data = np.array([[
    age, income, loan, credit_score, existing_loans, late_payments,
    loan_to_income,
    gender_male, gender_female,
    married, single, widowed, divorced,
    edu_master, edu_other, edu_phd,
    emp_self, emp_unemp, emp_retired, emp_student,
    purpose_home, purpose_personal, purpose_bus,
    debt_to_income
]])

# Prediction
if st.button("Predict Default Risk"):
    prediction = model.predict(input_data)[0]
    result = "🔴 Likely to Default" if prediction == 1 else "🟢 Low Risk"
    st.markdown(f"### Result: {result}")

st.markdown("---")
st.markdown("Designed for financial analysts and loan officers. Ensure all information is correct before prediction.")

st.write("Expected features:", model.n_features_in_)
st.write("Your input shape:", input_data.shape)
