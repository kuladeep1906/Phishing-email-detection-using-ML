# 🛡️ Phishing Email Detection using Machine Learning

This project is a Flask-based web application that detects phishing emails using machine learning models. It enables users to upload email files or paste content directly into the interface, and the system classifies the email as **phishing** or **legitimate** using three different models:

- ✅ Logistic Regression  
- 🌲 Random Forest  
- ⚡ XGBoost (Gradient Boosting)

---

## 📌 Project Highlights

- Built with Python and Flask for real-time detection
- Feature extraction using **TF-IDF**, structural patterns, and sender-based analysis
- Three classification models with performance comparison
- Integrated SHAP visualizations for explainability
- Achieved **95.2% accuracy** using Logistic Regression

---

## 📂 Dataset

This project uses a publicly available dataset from Kaggle, which contains over **82,000** emails (both phishing and legitimate).

🔗 [Phishing Email Dataset on Kaggle](https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset/data)


---

## 🧪 Features Extracted

- **Text-Based**: TF-IDF from email subject and body
- **Content-Based**: URL count, keyword frequency, HTML presence, punctuation usage
- **Sender-Based**: Domain name analysis
- **Advanced (XGBoost)**: Suspicious URLs, repeated characters, subject line flags

---

## 🚀 Installation

Clone this repository and install the required packages:

```bash
git clone https://github.com/kuladeep1906/Phishing-email-detection-using-ML.git
cd Phishing-email-detection-using-ML
pip install -r requirements.txt --break-system-packages
```

## 🧠 Run the Application
```bash
python3 app.py
```
Then, open your browser and go to:

http://127.0.0.1:5000/
---

## 🖥️ How It Works

1. User uploads an email (.txt or .eml) or pastes text into the interface
2. System extracts features (TF-IDF, structure, sender info)
3. Pre-trained ML models (LR, RF, XGB) predict if it's phishing
4. Displays results with confidence scores and SHAP feature insights
