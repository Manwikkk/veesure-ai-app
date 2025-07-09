import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import plotly.express as px

# Load model and scaler
model = joblib.load('model.joblib')
scaler = joblib.load('scaler.joblib')

# Page config
st.set_page_config(page_title="AI for Manufacturing", layout="wide")
st.title("⚙️ AI for Manufacturing – Machine Failure Prediction")
st.markdown("Upload your machine sensor data (CSV) to predict potential machine failures using AI.")

# File upload
uploaded_file = st.file_uploader("📤 Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    st.markdown("### 🧾 Uploaded Data Sample")
    st.dataframe(df.head())

    # Preprocess
    df_encoded = pd.get_dummies(df, columns=['Type'], drop_first=True)
    df_encoded.drop(['UDI', 'Product ID'], axis=1, inplace=True)

    drop_cols = ['Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    df_encoded = df_encoded.drop(columns=[col for col in drop_cols if col in df_encoded.columns])

    # Scale
    X_scaled = scaler.transform(df_encoded)

    # Predict
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)[:, 1]

    # Append predictions
    result_df = df.copy()
    result_df['Failure_Prediction'] = predictions
    result_df['Failure_Probability'] = probabilities.round(3)

    st.markdown("### 📊 Prediction Results")
    st.dataframe(result_df[['Product ID', 'Type', 'Failure_Prediction', 'Failure_Probability']].head(10))

    # Prediction distribution (Seaborn)
    st.markdown("### 🔍 Prediction Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x='Failure_Prediction', data=result_df, palette='Set2', ax=ax)
    ax.set_title("Predicted Failures vs Normal Operations")
    st.pyplot(fig)

    # Failure probability histogram (Plotly)
    st.markdown("### 📈 Failure Probability Distribution")
    fig2 = px.histogram(result_df, x='Failure_Probability', nbins=20, color='Failure_Prediction',
                        color_discrete_map={0: 'green', 1: 'red'},
                        title="Distribution of Failure Probabilities")
    st.plotly_chart(fig2, use_container_width=True)

    # Boxplot: Torque vs Failure
    st.markdown("### 📦 Sensor Insight: Torque vs Failure by Type")
    fig3 = px.box(result_df, x='Type', y='Torque (Nm)', color='Failure_Prediction',
                  title='Torque Levels per Machine Type',
                  color_discrete_map={0: 'blue', 1: 'red'})
    st.plotly_chart(fig3, use_container_width=True)

    # Correlation Heatmap
    st.markdown("### 🔥 Feature Correlation Heatmap")
    correlation_cols = result_df.select_dtypes(include=[np.number]).drop(columns=["Failure_Prediction", "Failure_Probability"])
    corr_matrix = correlation_cols.corr()

    fig4, ax4 = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, ax=ax4)
    ax4.set_title("Correlation Between Sensor Features")
    st.pyplot(fig4)

    st.success("✅ Analysis & Prediction complete!")

else:
    st.info("Upload a CSV file to begin analysis.")
