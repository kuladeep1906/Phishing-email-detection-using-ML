# Import necessary libraries
import pandas as pd
import numpy as np
import re
import email
import os
import pickle
import shap
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report



def extract_email_content(file_path):
    """Extracts text from .eml or .txt files."""
    try:
        if file_path.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        elif file_path.endswith(".eml"):
            with open(file_path, "rb") as f:
                msg = email.message_from_bytes(f.read())
                return extract_eml_text(msg)
    except Exception as e:
        print(f"Error reading file: {e}")
    return ""

def extract_eml_text(msg):
    """Extracts body text from an .eml email object."""
    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            content_disposition = str(part.get("Content-Disposition"))

            if "attachment" not in content_disposition and content_type == "text/plain":
                try:
                    body += part.get_payload(decode=True).decode()
                except:
                    pass
    else:
        body = msg.get_payload(decode=True).decode()
    return body


# ---------------------------
# 1️⃣ Function to Preprocess Email Text
# ---------------------------
def preprocess_text(text):
    """Clean email text by removing URLs, emails, and special characters."""
    if pd.isnull(text):
        return ""
    text = text.lower()
    text = re.sub(r'http[s]?://\S+', '', text)  # Remove URLs
    text = re.sub(r'\S+@\S+', '', text)  # Remove email addresses
    text = re.sub(r'[^a-z\s]', '', text)  # Remove special characters & numbers
    return text

# ---------------------------
# 2️⃣ Load & Merge Multiple Phishing Datasets
# ---------------------------
dataset_folder = r"/Users/kuladeep/Desktop/CNS Project/datasets/"
dataset_paths = [
    "phishing_email.csv", "Enron.csv", "Ling.csv", "Nazario.csv", 
    "Nigerian_Fraud.csv", "SpamAssasin.csv", "CEAS_08.csv"
]

print("📂 Loading datasets...")
dfs = [pd.read_csv(os.path.join(dataset_folder, filename), encoding='latin1', on_bad_lines='skip') for filename in dataset_paths]
print("✅ Datasets Loaded Successfully")

for df in dfs:
    if 'text_combined' in df.columns:
        df.rename(columns={'text_combined': 'body'}, inplace=True)
    df['subject'] = df.get('subject', "")
    df['body'] = df.get('body', "")
    df['label'] = df.get('label', np.nan)

merged_df = pd.concat([df[['subject', 'body', 'label']] for df in dfs], ignore_index=True)
merged_df.dropna(subset=['label'], inplace=True)
merged_df.drop_duplicates(inplace=True)
print("✅ Data Preprocessing Completed")

# ---------------------------
# 3️⃣ Feature Engineering
# ---------------------------
print("📝 Applying Feature Extraction...")
merged_df['processed_text'] = (merged_df['subject'] + " " + merged_df['body']).apply(preprocess_text)

# Ensure y is defined before splitting data
y = merged_df['label'].astype(int)

# ---------------------------
# 4️⃣ Convert Text to Numerical Format (TF-IDF)
# ---------------------------
vectorizer = TfidfVectorizer(max_features=8000)
X_text = vectorizer.fit_transform(merged_df['processed_text'])

print("✅ TF-IDF applied successfully.")

# ---------------------------
# 5️⃣ Splitting Data into Training and Testing Sets
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(X_text, y, test_size=0.2, random_state=42)
print("✅ Data Split Completed")

# ---------------------------
# 6️⃣ Train & Save Models if Not Already Saved
# ---------------------------
model_files = ["logistic_regression.pkl", "random_forest.pkl", "xgboost_model.pkl", "tfidf_vectorizer.pkl"]

if all(os.path.exists(file) for file in model_files):
    print("✅ Pre-trained models found. Skipping training.")
else:
    print("🚀 Training Models...")

    # Logistic Regression
    logistic_model = LogisticRegression(max_iter=1000, solver='liblinear')
    logistic_model.fit(X_train, y_train)

    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=30, max_depth=8, min_samples_split=10, n_jobs=2)
    rf_model.fit(X_train, y_train)

    # XGBoost
    xgb_model = xgb.XGBClassifier(n_estimators=50, max_depth=6, learning_rate=0.05, verbosity=1)
    xgb_model.fit(X_train, y_train)

    print("✅ Model Training Completed")

    # Save Models and Vectorizer
    with open("logistic_regression.pkl", "wb") as f:
        pickle.dump(logistic_model, f)
    with open("random_forest.pkl", "wb") as f:
        pickle.dump(rf_model, f)
    with open("xgboost_model.pkl", "wb") as f:
        pickle.dump(xgb_model, f)
    with open("tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    print("✅ Models and Vectorizer Saved Successfully")

# ---------------------------
# 7️⃣ Load Models for API Use
# ---------------------------
with open("logistic_regression.pkl", "rb") as f:
    logistic_model = pickle.load(f)

with open("random_forest.pkl", "rb") as f:
    rf_model = pickle.load(f)

with open("xgboost_model.pkl", "rb") as f:
    xgb_model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

print("✅ Models Loaded Successfully!")

# ---------------------------
# 8️⃣ Function for Classification
# ---------------------------




# SHAP Explainer for Logistic Regression
explainer = shap.LinearExplainer(logistic_model, shap.maskers.Independent(vectorizer.transform([""])))

def classify_email(email_text):
    processed_email = preprocess_text(email_text)
    email_vector = vectorizer.transform([processed_email])

    # Get probability predictions (convert to percentage)
    proba_lr = int(logistic_model.predict_proba(email_vector)[0][1] * 100)
    proba_rf = int(rf_model.predict_proba(email_vector)[0][1] * 100)
    proba_xgb = int(xgb_model.predict_proba(email_vector)[0][1] * 100)

    # Majority Voting Logic
    threshold = 50
    phishing_votes = sum([proba_lr > threshold, proba_rf > threshold, proba_xgb > threshold])

    # Determine Final Prediction
    final_prediction = "Phishing" if phishing_votes >= 2 else "Safe"

    # Confidence Score = Highest probability among models
    max_confidence = max(proba_lr, proba_rf, proba_xgb)

    # 🔹 Compute SHAP Explanation
    try:
        shap_values = explainer(email_vector)
        feature_names = vectorizer.get_feature_names_out()
        feature_importance = sorted(zip(feature_names, shap_values.values[0]), key=lambda x: abs(x[1]), reverse=True)
        top_features = feature_importance[:15]  

        # 🔹 DEBUG: Print SHAP values
        print("\n🔹 DEBUG: SHAP Values Computed Successfully 🔹")
        for word, impact in top_features:
            print(f"{word}: {impact:.3f}")

    except Exception as e:
        print(f"\n❌ DEBUG: Error computing SHAP values: {e}")
        top_features = [("No SHAP Data Available", 0)]  # Default if SHAP fails

    return final_prediction, max_confidence, proba_lr, proba_rf, proba_xgb, top_features