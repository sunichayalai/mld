import streamlit as st
import numpy as np
import joblib
import numpy as np

# Load model
model = joblib.load('model.pkl')

# Set background color and title
st.set_page_config(page_title="Loan Default Predictor", layout="centered")
st.markdown("<h1 style='text-align: center; color: navy;'>Loan Default Risk Prediction</h1>", unsafe_allow_html=True)

st.write("Enter the applicant's loan details below:")

# Collect user input (adjust fields as per your dataset)
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
applicant_income = st.number_input("Applicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
credit_history = st.selectbox("Credit History", ["Good (1.0)", "Bad (0.0)"])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# Preprocessing for model input (adjust based on your preprocessing steps)
input_data = np.array([[
    1 if gender == "Male" else 0,
    1 if married == "Yes" else 0,
    1 if education == "Graduate" else 0,
    applicant_income,
    loan_amount,
    1.0 if credit_history == "Good (1.0)" else 0.0,
    1 if property_area == "Urban" else 0,
    1 if property_area == "Semiurban" else 0
]])  # Make sure this matches your trained model's feature order

# Predict
if st.button("Predict Default Risk"):
    prediction = model.predict(input_data)[0]
    result = "🔴 Likely to Default" if prediction == 1 else "🟢 Low Risk"
    st.markdown(f"### Result: {result}")

# Footer
st.markdown("---")
st.markdown("Designed for financial analysts and loan officers. Ensure all information is correct before prediction.")
