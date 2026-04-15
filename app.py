import streamlit as st
import joblib
import pandas as pd

st.title("Customer Analytics Dashboard")

# Sidebar menu
menu = st.sidebar.selectbox(
    "Select View",
    ["Churn Prediction", "Customer Segmentation"]
)

# ---------------- CHURN ----------------
if menu == "Churn Prediction":
    model = joblib.load("churn_model.pkl")

    st.header("Churn Prediction")

    tenure = st.slider("Tenure", 0, 72, 10)
    charges = st.slider("Monthly Charges", 0, 150, 50)

    input_data = pd.DataFrame([[tenure, charges]],
                              columns=["tenure", "MonthlyCharges"])

    if st.button("Predict"):
        prediction = model.predict(input_data)[0]

        if prediction == 1:
            st.error("High Risk of Churn")
        else:
            st.success("Low Risk of Churn")

# ---------------- SEGMENTATION ----------------
if menu == "Customer Segmentation":
    st.header("Customer Segments")

    df = pd.read_csv("segmented_customers.csv")

    st.write("Sample Data:")
    st.dataframe(df.head())

    st.write("Segment Counts:")
    st.bar_chart(df["Segment"].value_counts())

    st.write("Segment Meaning:")
    st.write("0 = Low Value Customers")
    st.write("1 = Mid Value Customers")
    st.write("2 = High Value Customers")
