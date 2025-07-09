import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load model and scaler
model = joblib.load('model.joblib')
scaler = joblib.load('scaler.joblib')

st.set_page_config(page_title="Machine Failure Prediction", layout="wide")
st.title("⚙️ AI for Manufacturing – Machine Failure Prediction")
st.write("Upload your machine sensor data (CSV) and get instant failure predictions.")

# File uploader
uploaded_file = st.file_uploader("📤 Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("🧾 Uploaded Data")
    st.dataframe(df.head())

    # Preprocess
    df_encoded = pd.get_dummies(df, columns=['Type'], drop_first=True)
    df_encoded.drop(['UDI', 'Product ID'], axis=1, inplace=True)

    # Drop targets (if present)
    drop_cols = ['Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    df_encoded = df_encoded.drop(columns=[col for col in drop_cols if col in df_encoded.columns])

    # Scale
    X_scaled = scaler.transform(df_encoded)

    # Predict
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)[:, 1]

    st.subheader("📊 Predictions")
    result_df = df.copy()
    result_df['Failure_Prediction'] = predictions
    result_df['Failure_Probability'] = probabilities.round(3)
    st.dataframe(result_df[['Product ID', 'Type', 'Failure_Prediction', 'Failure_Probability']].head(10))

    # Visualize
    st.subheader("🔍 Failure Prediction Summary")
    fig, ax = plt.subplots()
    sns.countplot(x='Failure_Prediction', data=result_df, palette='Set2', ax=ax)
    ax.set_title("Prediction Distribution")
    st.pyplot(fig)

    st.success("✅ Prediction complete!")
