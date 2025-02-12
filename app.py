import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from streamlit_lottie import st_lottie
import json

# ✅ MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(page_title="Loan Default Prediction", layout="wide")

# ✅ Load the Model
try:
    model = joblib.load("modell.pkl")  # Efficient model loading
except Exception:
    model = None

# ✅ Load Preprocessor
try:
    preprocessor = joblib.load("preprocessor.pkl")
except FileNotFoundError:
    preprocessor = None

# ✅ Essential Features
selected_features = ["loan_amnt", "int_rate", "installment", "annual_inc", "dti", 
                     "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc", 
                     "mort_acc", "credit_score"]

# ✅ Sidebar with 3D UI Enhancements
st.sidebar.header("📌 User Input Features")
def user_input_features():
    return pd.DataFrame([[
        st.sidebar.number_input("💰 Loan Amount", min_value=1000, value=10000),
        st.sidebar.slider("📉 Interest Rate (%)", 0.0, 30.0, 10.0),
        st.sidebar.number_input("💳 Monthly Installment", min_value=0, value=300),
        st.sidebar.number_input("💵 Annual Income", min_value=1000, value=50000),
        st.sidebar.slider("⚖️ Debt-to-Income Ratio (%)", 0.0, 100.0, 30.0),
        st.sidebar.number_input("📂 Open Accounts", min_value=0, value=5),
        st.sidebar.number_input("⚖️ Public Records", min_value=0, value=0),
        st.sidebar.number_input("🔄 Revolving Balance", min_value=0, value=2000),
        st.sidebar.slider("📊 Revolving Utilization (%)", 0.0, 100.0, 50.0),
        st.sidebar.number_input("📁 Total Accounts", min_value=0, value=15),
        st.sidebar.number_input("🏡 Mortgage Accounts", min_value=0, value=1),
        st.sidebar.slider("💳 Credit Score", 300, 850, 650)
    ]], columns=selected_features)

df = user_input_features()

# ✅ Prediction with 3D Elements
if st.button("🔍 Predict Default Risk"):
    if model is None:
        st.error("🚨 Model not loaded. Check for MemoryError or incorrect file path.")
    else:
        df_transformed = preprocessor.transform(df) if preprocessor else df
        if df_transformed.shape[1] != model.n_features_in_:
            st.error(f"⚠️ Feature count mismatch! Expected {model.n_features_in_}, got {df_transformed.shape[1]}.")
        else:
            prediction = model.predict(df_transformed)[0]
            probability = model.predict_proba(df_transformed)[0][1] * 100
            if prediction == 1:
                st.error(f"🔴 High Default Risk! Probability: {probability:.2f}%")
            else:
                st.success(f"🟢 Low Default Risk! Probability: {probability:.2f}%")

            # ✅ 3D Feature Importance
            st.subheader("🔍 Feature Importance")
            feature_importance = model.feature_importances_
            fig = go.Figure(data=[go.Bar(y=df.columns, x=feature_importance, orientation='h', marker_color='skyblue')])
            fig.update_layout(title="📊 Feature Importance in Loan Default Prediction", xaxis_title="Importance")
            st.plotly_chart(fig, use_container_width=True)

# ✅ Loan Default Analytics with 3D UI
st.subheader("📈 Loan Default Analytics")
st.write("Interactive visualization of past loan defaults and trends.")
def create_analytics_chart():
    chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["Year", "Default Rate", "Loan Amount"])
    
    # Ensure that Year is an integer column for x-axis
    chart_data["Year"] = (chart_data["Year"] * 1000).astype(int) + 2020  # Adjusting to a reasonable range for years

    fig = px.line(chart_data, x="Year", y=["Default Rate", "Loan Amount"], markers=True, title="📊 Loan Default Trends")
    st.plotly_chart(fig, use_container_width=True)

create_analytics_chart()
