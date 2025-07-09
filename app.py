import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load model and scaler
model = joblib.load('model.joblib')
scaler = joblib.load('scaler.joblib')

# Page config
st.set_page_config(page_title="Machine Failure Prediction", layout="wide")
st.title("⚙️ AI for Manufacturing – Machine Failure Prediction")
st.markdown("Upload your machine sensor data (CSV) and get instant failure predictions.")

# Sidebar
st.sidebar.title("📂 Navigation")
view = st.sidebar.radio("Go to:", ["📤 Upload", "📊 Predictions", "📈 Visualizations", "ℹ️ Summary"])
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Preprocess
    df_encoded = pd.get_dummies(df, columns=['Type'], drop_first=True)
    df_encoded.drop(['UDI', 'Product ID'], axis=1, inplace=True)
    drop_cols = ['Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    df_encoded = df_encoded.drop(columns=[col for col in drop_cols if col in df_encoded.columns])

    X_scaled = scaler.transform(df_encoded)
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)[:, 1]

    result_df = df.copy()
    result_df['Failure_Prediction'] = predictions
    result_df['Failure_Probability'] = probabilities.round(3)

    if view == "📤 Upload":
        st.header("📤 Uploaded Data")
        st.dataframe(df.head())

    elif view == "📊 Predictions":
        st.header("📊 Failure Predictions")
        st.dataframe(result_df[['Product ID', 'Type', 'Failure_Prediction', 'Failure_Probability']].head(15))

    elif view == "📈 Visualizations":
        st.header("📈 Prediction Visualizations")

        col1, col2 = st.columns(2)

        with col1:
            fig1, ax1 = plt.subplots()
            sns.countplot(x='Failure_Prediction', data=result_df, palette='Set2', ax=ax1)
            ax1.set_title("Prediction Distribution")
            st.pyplot(fig1)

        with col2:
            fig2, ax2 = plt.subplots()
            sns.histplot(result_df['Failure_Probability'], bins=20, kde=True, ax=ax2, color='skyblue')
            ax2.set_title("Failure Probability Distribution")
            st.pyplot(fig2)

        st.markdown("### 🔢 Feature Correlation Heatmap")
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        sns.heatmap(df_encoded.corr(), cmap='coolwarm', ax=ax3)
        st.pyplot(fig3)

    elif view == "ℹ️ Summary":
        st.header("📋 Dataset Summary")
        st.write("Shape:", df.shape)
        st.write("Missing Values:")
        st.write(df.isnull().sum())
        st.write("Descriptive Statistics:")
        st.write(df.describe())

    st.sidebar.success("✅ Prediction complete!")

else:
    st.info("Please upload a CSV file to proceed.")
