# ⚙️ Veesure AI – Machine Failure Prediction App

> Predict machine failures from sensor data using an AI model, and explore insights through interactive dashboards.  
> Built for **Veesure Animal Health** as part of the **Intel AI for Manufacturing** project.

![App Preview](assets/preview.png)

---

## 🚀 Live App

👉 [Click here to try the app](https://veesure-ai-app.streamlit.app)

---

## 📊 Features

- 📥 Upload sensor data (CSV)
- 🔍 Predict machine failure using a trained AI model
- 📈 Visualizations:
  - Prediction distribution (Seaborn)
  - Failure probability histogram (Plotly)
  - Torque analysis by machine type
  - 🔥 Correlation heatmap
- 🧠 Powered by XGBoost and Scikit-learn

---

## 🧪 Tech Stack

| Category     | Tools |
|--------------|-------|
| Language     | Python 3.10 |
| ML Models    | XGBoost, scikit-learn |
| Dashboard    | Streamlit |
| Visualization | Matplotlib, Seaborn, Plotly |
| Hosting      | Streamlit Cloud |
| Dev Tools    | VS Code, Git, GitHub |

---

## 📁 How to Run Locally

```bash
git clone https://github.com/Manwikkk/veesure-ai-app.git
cd veesure-ai-app
pip install -r requirements.txt
streamlit run app.py
