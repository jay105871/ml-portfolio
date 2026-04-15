import streamlit as st
import joblib
import pandas as pd

model = joblib.load("model.pkl")

st.title("Customer Churn Prediction App")

tenure = st.slider("Tenure", 0, 72, 10)
charges = st.slider("Monthly Charges", 0, 150, 50)

input_data = pd.DataFrame([[tenure, charges]],
                          columns=["tenure", "monthly_charges"])

if st.button("Predict"):
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error("High Risk of Churn")
    else:
        st.success("Low Risk of Churn")
