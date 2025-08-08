import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Load dataset for visualizations (replace 'loan_data.csv' with your dataset)
@st.cache_data
def load_data():
    return pd.read_csv("loan_data.csv")  # Ensure this file exists in your directory

df = load_data()

# App title
st.title("🏦 Loan Default Prediction App")
st.write("Enter applicant details to predict loan default risk.")

# --- Input fields ---
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
applicant_income = st.number_input("Applicant Income", min_value=0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount (in thousands)", min_value=0)
loan_amount_term = st.number_input("Loan Amount Term (in months)", min_value=0)
credit_history = st.selectbox("Credit History", ["Good (1)", "Bad (0)"])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# Convert inputs to numeric
gender = 1 if gender == "Male" else 0
married = 1 if married == "Yes" else 0
dependents = 3 if dependents == "3+" else int(dependents)
education = 0 if education == "Graduate" else 1
self_employed = 1 if self_employed == "Yes" else 0
credit_history = 1 if credit_history == "Good (1)" else 0
property_area = {"Urban": 2, "Semiurban": 1, "Rural": 0}[property_area]

# Input array
input_data = np.array([[gender, married, dependents, education,
                        self_employed, applicant_income, coapplicant_income,
                        loan_amount, loan_amount_term, credit_history, property_area]])

# Scale numeric features
input_scaled = scaler.transform(input_data)

# --- Prediction ---
if st.button("Predict Loan Default"):
    prediction = model.predict(input_scaled)
    if prediction[0] == 1:
        st.error("❌ High Risk: Loan Likely to Default.")
    else:
        st.success("✅ Low Risk: Loan Likely to be Approved.")

# --- Visualizations ---
st.subheader("📊 Data Visualizations")

# 1. Applicant Income Distribution
st.markdown("### 🔹 Applicant Income Distribution")
fig1, ax1 = plt.subplots()
sns.histplot(df['ApplicantIncome'], bins=30, kde=True, ax=ax1, color="skyblue")
ax1.set_xlabel("Applicant Income")
ax1.set_ylabel("Count")
st.pyplot(fig1)

# 2. Loan Amount vs Applicant Income
st.markdown("### 🔹 Loan Amount vs Applicant Income")
fig2, ax2 = plt.subplots()
sns.scatterplot(data=df, x='ApplicantIncome', y='LoanAmount', hue='Loan_Status', ax=ax2)
ax2.set_xlabel("Applicant Income")
ax2.set_ylabel("Loan Amount")
st.pyplot(fig2)

# 3. Credit History vs Loan Status
st.markdown("### 🔹 Credit History vs Loan Status")
fig3, ax3 = plt.subplots()
sns.countplot(data=df, x='Credit_History', hue='Loan_Status', ax=ax3)
ax3.set_xlabel("Credit History (0 = Bad, 1 = Good)")
ax3.set_ylabel("Count")
st.pyplot(fig3)

# 4. Loan Amount Distribution
st.markdown("### 🔹 Loan Amount Distribution")
fig4, ax4 = plt.subplots()
sns.histplot(df['LoanAmount'], bins=30, kde=True, color='lightcoral', ax=ax4)
ax4.set_xlabel("Loan Amount")
st.pyplot(fig4)

