import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# App Configuration
st.set_page_config(page_title="Machine Failure Predictor", layout="wide")
st.markdown("<h1 style='text-align: center;'>⚙️ AI for Manufacturing</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: gray;'>Real-time Machine Failure Prediction with Visual Insights</h4>", unsafe_allow_html=True)
st.markdown("---")

# Load model and scaler
model = joblib.load('model.joblib')
scaler = joblib.load('scaler.joblib')

# File Upload
st.sidebar.header("📤 Upload CSV File")
uploaded_file = st.sidebar.file_uploader("Upload your machine sensor data", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.markdown("### 🧾 Preview of Uploaded Data")
    st.dataframe(df.head(), use_container_width=True)

    # Preprocess
    df_encoded = pd.get_dummies(df, columns=['Type'], drop_first=True)
    df_encoded.drop(['UDI', 'Product ID'], axis=1, inplace=True, errors='ignore')
    drop_cols = ['Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    df_encoded = df_encoded.drop(columns=[col for col in drop_cols if col in df_encoded.columns], errors='ignore')

    # Scale and Predict
    X_scaled = scaler.transform(df_encoded)
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)[:, 1]

    result_df = df.copy()
    result_df['Failure_Prediction'] = predictions
    result_df['Failure_Probability'] = probabilities.round(3)

    # Metrics
    st.markdown("### 📊 Key Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", len(result_df))
    col2.metric("Predicted Failures", int(result_df['Failure_Prediction'].sum()))
    col3.metric("Average Failure Probability", f"{result_df['Failure_Probability'].mean():.2%}")

    # Visual: Prediction Distribution
    st.markdown("### 🔍 Prediction Distribution")
    fig1 = px.histogram(result_df, x='Failure_Prediction', color='Failure_Prediction',
                        color_discrete_sequence=['#63cdda', '#e15f41'],
                        labels={'Failure_Prediction': 'Predicted Failure'},
                        title="Machine Failure vs Normal Cases")
    st.plotly_chart(fig1, use_container_width=True)

    # Visual: Probability Heatmap by Type
    if 'Type' in df.columns:
        st.markdown("### 🔥 Failure Probability by Machine Type")
        fig2 = px.box(result_df, x="Type", y="Failure_Probability", color="Type",
                      color_discrete_sequence=px.colors.qualitative.Bold)
        st.plotly_chart(fig2, use_container_width=True)

    # Table of Top Predictions
    st.markdown("### 📋 Top 10 High-Risk Machines")
    top_failures = result_df.sort_values(by='Failure_Probability', ascending=False).head(10)
    st.dataframe(top_failures[['Product ID', 'Type', 'Failure_Probability']], use_container_width=True)

    st.success("✅ Prediction complete!")

else:
    st.info("Please upload a CSV file to get started.")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>Built for Veesure Animal Health • AI for Manufacturing Project</p>", unsafe_allow_html=True)
