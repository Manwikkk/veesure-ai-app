<h1 align="center">⚙️ Veesure AI – Machine Failure Predictor</h1>

<p align="center">
  <img src="assets/banner.png" width="90%" alt="banner">
</p>

<p align="center">
  <a href="https://veesure-ai-app.streamlit.app">
    <img src="https://img.shields.io/badge/Streamlit-App-green?style=for-the-badge&logo=streamlit" />
  </a>
  <a href="https://github.com/Manwikkk/veesure-ai-app/blob/main/Veesure_AI_Project_Report.pdf">
    <img src="https://img.shields.io/badge/Download-Report-blue?style=for-the-badge&logo=adobeacrobatreader" />
  </a>
</p>

---

### 📌 About the Project

Veesure AI is a machine failure prediction system built for **Veesure Animal Health** under the **Intel AI for Manufacturing** initiative.  
It uses real-time sensor data to predict failures and provides insightful visualizations for industrial operations.

---

### 📸 App Screenshots

| 🖼️ Upload Interface | 📊 Prediction Output |
|---------------------|----------------------|
| ![](assets/1-upload.png) | ![](assets/2-prediction.png) |

| 📈 Prediction Visualization | 🔥 Correlation Heatmap |
|-----------------------------|-------------------------|
| ![](assets/3-visualization.png) | ![](assets/4-heatmap.png) |

| 📋 Dataset Summary | 📌 Descriptive Analysis |
|--------------------|-------------------------|
| ![](assets/5-summary.png) | ![](assets/6-descriptive.png) |

---

### 🚀 Live Demo

Click here to try the app:  
👉 [https://veesure-ai-app.streamlit.app](https://veesure-ai-app.streamlit.app)

---

### 📊 Features

- 📥 Upload sensor data (CSV)
- 🔍 Predict machine failure using a trained AI model
- 📈 Visualizations:
  - Prediction distribution (Seaborn)
  - Failure probability histogram (Plotly)
  - Torque analysis by machine type
  - 🔥 Correlation heatmap
- 🧠 Powered by XGBoost and Scikit-learn

---

### 🔧 Tech Stack

| Category     | Tools |
|--------------|-------|
| Language     | Python 3.10 |
| ML Models    | XGBoost, scikit-learn |
| Dashboard    | Streamlit |
| Visualization | Matplotlib, Seaborn, Plotly |
| Hosting      | Streamlit Cloud |
| Dev Tools    | VS Code, Git, GitHub |

---

### 📁 Project Structure

├── app.py # Streamlit app logic
├── model.joblib # Trained ML model
├── scaler.joblib # Preprocessing scaler
├── requirements.txt # Dependencies
├── README.md # This file
├── Veesure_AI_Project_Report.pdf
├── assets/ # Screenshots
└── .gitignore

---

### 📥 How to Run Locally

```bash
git clone https://github.com/Manwikkk/veesure-ai-app.git
cd veesure-ai-app
pip install -r requirements.txt
streamlit run app.py
