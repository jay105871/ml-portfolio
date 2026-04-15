from tensorflow import keras
import numpy as np
import streamlit as st
import joblib
import pandas as pd

st.title("Customer Analytics Dashboard")

# Sidebar menu
menu = st.sidebar.selectbox(
    "Select View",
    ["Churn Prediction", "Customer Segmentation", "Recommendations"]
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

# ---------------- RECOMMENDER ----------------
if menu == "Recommendations":
    st.header("Product Recommendations")

    model = keras.models.load_model("recommender_model.keras")

    user_id = st.slider("Select User ID", 0, 100, 1)

    items = np.arange(0, 50)
    user_array = np.full(len(items), user_id)

    predictions = model.predict([user_array, items])

    top_items = items[np.argsort(predictions.flatten())[-5:]]

    st.write("Top Recommended Items:")
    st.write(top_items)
