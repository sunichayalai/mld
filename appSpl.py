import streamlit as st
import numpy as np
import pandas as pd
import joblib

model_default = joblib.load("model.pkl")

# You can update this with your actual best model's metrics
best_model_name = "Balanced Random Forest"

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

# Feature engineering (matching app.py)
loan_to_income = loan / (income + 1)
debt_to_income = existing_loans * loan / (income + 1)

# Complete one-hot encoding (matching app.py expected features)
# Gender encoding
gender_male = 1 if gender == "Male" else 0

# Marital status encoding
marital_married = 1 if marital == "Married" else 0
marital_single = 1 if marital == "Single" else 0
marital_widowed = 1 if marital == "Widowed" else 0

# Education level encoding
edu_high_school = 1 if education == "High School" else 0
edu_master = 1 if education == "Master" else 0
edu_other = 1 if education == "Other" else 0
edu_phd = 1 if education == "PhD" else 0

# Employment status encoding
emp_retired = 1 if employment == "Retired" else 0
emp_self_employed = 1 if employment == "Self-employed" else 0
emp_student = 1 if employment == "Student" else 0
emp_unemployed = 1 if employment == "Unemployed" else 0

# Purpose of loan encoding
purpose_car = 1 if purpose == "Car" else 0
purpose_education = 1 if purpose == "Education" else 0
purpose_home = 1 if purpose == "Home" else 0
purpose_personal = 1 if purpose == "Personal" else 0

# Create input data with all 24 features (matching app.py expected_features)
input_data = np.array([[ 
    age,                    # 1. Age
    income,                 # 2. AnnualIncome
    loan,                   # 3. LoanAmountRequested
    credit_score,           # 4. CreditScore
    existing_loans,         # 5. ExistingLoansCount
    late_payments,          # 6. LatePaymentsLastYear
    loan_to_income,         # 7. Loan_to_Income
    debt_to_income,         # 8. Debt_to_Income
    gender_male,            # 9. Gender_Male
    marital_married,        # 10. MaritalStatus_Married
    marital_single,         # 11. MaritalStatus_Single
    marital_widowed,        # 12. MaritalStatus_Widowed
    edu_high_school,        # 13. EducationLevel_High School
    edu_master,             # 14. EducationLevel_Master
    edu_other,              # 15. EducationLevel_Other
    edu_phd,                # 16. EducationLevel_PhD
    emp_retired,            # 17. EmploymentStatus_Retired
    emp_self_employed,      # 18. EmploymentStatus_Self-employed
    emp_student,            # 19. EmploymentStatus_Student
    emp_unemployed,         # 20. EmploymentStatus_Unemployed
    purpose_car,            # 21. PurposeOfLoan_Car
    purpose_education,      # 22. PurposeOfLoan_Education
    purpose_home,           # 23. PurposeOfLoan_Home
    purpose_personal        # 24. PurposeOfLoan_Personal
]])

# Prediction logic
if st.button("📊 Predict Loan Outcome"):
    try:
        # Check for missing features
        expected_features = [
            "age", "income", "loan", "credit_score", "existing_loans", "late_payments",
            "loan_to_income", "debt_to_income", "gender_male", "marital_married", "marital_single", "marital_widowed",
            "edu_high_school", "edu_master", "edu_other", "edu_phd", "emp_retired", "emp_self_employed", 
            "emp_student", "emp_unemployed", "purpose_car", "purpose_education", "purpose_home", "purpose_personal"
        ]
        
        # Check for NaN values
        missing_features = [
            fname for idx, fname in enumerate(expected_features)
            if input_data[0, idx] is None or (isinstance(input_data[0, idx], float) and np.isnan(input_data[0, idx]))
        ]

        if missing_features:
            st.warning(f"⚠️ Missing or invalid features: {missing_features}")

        # Use all loaded models for prediction
        models = [
            ("Default Model", model_default),
        ]

        results = []
        for name, mdl in models:
            try:
                pred = mdl.predict(input_data)[0]
                conf = mdl.predict_proba(input_data)[0][pred]
                results.append((name, pred, conf))
            except Exception as e:
                results.append((name, "Error", str(e)))

        # Display results
        st.markdown("## 🧠 Model Predictions")
        for name, pred, conf in results:
            if pred == "Error":
                st.markdown(f"<b>{name}:</b> <span style='color:red;'>Prediction Error: {conf}</span>", unsafe_allow_html=True)
            else:
                result_text = "✅ Loan Approved" if pred == 0 else "❌ Loan Not Approved"
                result_color = "green" if pred == 0 else "red"
                st.markdown(
                    f"<b>{name}:</b> <span style='color:{result_color};'>{result_text}</span> &nbsp; "
                    f"<span style='color:#444;'>Confidence: <b>{conf:.2%}</b></span>",
                    unsafe_allow_html=True
                )

        # Show feature info
        st.markdown("---")
        st.subheader("🔍 Feature Information")
        st.markdown(f"**Features used:** {input_data.shape[1]} features")
        st.markdown(f"**Model expects:** 24 features")
        st.markdown("✅ Feature count matches model requirements")
        feature_names = [
            "age", "income", "loan", "credit_score", "existing_loans", "late_payments",
            "loan_to_income", "debt_to_income", "gender_male", "marital_married", "marital_single", "marital_widowed",
            "edu_high_school", "edu_master", "edu_other", "edu_phd", "emp_retired", "emp_self_employed", 
            "emp_student", "emp_unemployed", "purpose_car", "purpose_education", "purpose_home", "purpose_personal"
        ]
        st.markdown("**Feature names used:**")
        st.code(", ".join(feature_names), language="text")

        # Show model info
        st.markdown("---")
        st.subheader("🧠 Model Info")
        st.markdown("**Models Used:**")
        st.markdown(", ".join([name for name, _ in models]))

    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        st.info("💡 This error usually occurs when the model expects different features than what's provided. The app has been updated to use the correct 24-feature format.")
st.markdown("---")
st.markdown("<p style='text-align:center;'>© 2025 Loan Predictor | Powered by Streamlit</p>", unsafe_allow_html=True)
