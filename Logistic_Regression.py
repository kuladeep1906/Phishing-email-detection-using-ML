# Import necessary libraries
import pandas as pd
import numpy as np
import re
import os
import pickle  # For saving/loading model
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from flask import Flask, request, jsonify

# ---------------------------
# 1️⃣ Function to Preprocess Email Text
# ---------------------------
def preprocess_text(text):
    """Clean email text by removing URLs, emails, and special characters."""
    if pd.isnull(text):
        return ""
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'http[s]?://\S+', '', text)  # Remove URLs
    text = re.sub(r'\S+@\S+', '', text)  # Remove email addresses
    text = re.sub(r'[^a-z\s]', '', text)  # Remove special characters & numbers
    return text

# ---------------------------
# 2️⃣ Additional Feature Extraction Functions
# ---------------------------
def extract_sender_features(email_sender):
    """Extract domain name from sender email."""
    if pd.isnull(email_sender) or "@" not in email_sender:
        return "unknown"
    return email_sender.split("@")[-1]  # Extract domain (e.g., gmail.com)

def count_urls(text):
    """Count the number of URLs in email content."""
    if pd.isnull(text):
        return 0
    urls = re.findall(r'http[s]?://\S+', text)
    return len(urls)

def count_phishing_keywords(text):
    """Check for phishing-related words in email content."""
    phishing_keywords = ["urgent", "winner", "claim now", "free", 
                         "account verification", "credit card", "bank", "prize"]
    if pd.isnull(text):
        return 0
    return sum([1 for word in phishing_keywords if word in text.lower()])

def excessive_punctuation(text):
    """Count excessive punctuation marks (e.g., !!!, $$$, ???)."""
    if pd.isnull(text):
        return 0
    return len(re.findall(r'[!?$]{2,}', text))  # Counts repeated !, ?, $

def average_sentence_length(text):
    """Calculate average sentence length (words per sentence)."""
    if pd.isnull(text) or text.strip() == "":
        return 0
    sentences = re.split(r'[.!?]', text)  # Split by sentence-ending punctuation
    words = text.split()
    return len(words) / max(1, len(sentences))  # Avoid division by zero

def detect_html(text):
    """Detect whether an email contains HTML formatting."""
    if pd.isnull(text):
        return 0
    return int(bool(re.search(r'<[^>]+>', text)))  # Detects <html> tags

print("✅ Starting Phishing Email Detection Tool...")

# ---------------------------
# 3️⃣ Load & Merge Multiple Phishing Datasets
# ---------------------------
dataset_folder = r"""c:\Users\nitin\Downloads\Phishing-Email-Detection-Using-Machine-Learning"""
dataset_paths = [
    "phishing_email.csv", "Enron.csv", "Ling.csv", "Nazario.csv", 
    "Nigerian_Fraud.csv", "SpamAssasin.csv", "CEAS_08.csv"
]

print("📂 Loading datasets...")
dfs = [pd.read_csv(os.path.join(dataset_folder, filename), encoding='latin1', on_bad_lines='skip') for filename in dataset_paths]
print("✅ Datasets Loaded Successfully")

# Standardize column names and merge datasets
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
# 4️⃣ Apply Text Preprocessing & Extract Features
# ---------------------------
print("📝 Applying Feature Extraction...")

merged_df['processed_text'] = (merged_df['subject'] + " " + merged_df['body']).apply(preprocess_text)
merged_df['sender_domain'] = merged_df['subject'].apply(extract_sender_features)
merged_df['url_count'] = merged_df['body'].apply(count_urls)
merged_df['phishing_word_count'] = merged_df['body'].apply(count_phishing_keywords)
merged_df['excessive_punctuation'] = merged_df['body'].apply(excessive_punctuation)
merged_df['avg_sentence_length'] = merged_df['body'].apply(average_sentence_length)
merged_df['contains_html'] = merged_df['body'].apply(detect_html)

# ---------------------------
# 5️⃣ Convert Text to Numerical Format (TF-IDF)
# ---------------------------
print("🔠 Applying TF-IDF Vectorization...")
vectorizer = TfidfVectorizer(max_features=6260)  # Fixed feature count
X_text = vectorizer.fit_transform(merged_df['processed_text'])

# Convert categorical sender domains into numerical format
X_sender = pd.get_dummies(merged_df[['sender_domain']]).values

# Combine all features 
X_combined = np.hstack((X_text.toarray(), X_sender, 
                        merged_df[['url_count', 'phishing_word_count', 'excessive_punctuation', 
                                   'avg_sentence_length', 'contains_html']].values))

y = merged_df['label'].astype(int)
print("✅ Feature Extraction Completed")

# ---------------------------
# 6️⃣ Splitting Data into Training and Testing Sets
# ---------------------------
print("✂️ Splitting Data into Training and Testing Sets...")
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)
print("✅ Data Split Completed")

# ---------------------------
# 7️⃣ Training Logistic Regression Model
# ---------------------------
print("🤖 Training Logistic Regression Model...")
model = LogisticRegression(max_iter=1000, solver='liblinear')
model.fit(X_train, y_train)
print("✅ Model Training Completed")

# ---------------------------
# 8️⃣ Evaluating Model Performance
# ---------------------------
print("📊 Evaluating Model...")
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("✅ Model Evaluation Completed")

# ---------------------------
# 9️⃣ Saving the Model and Vectorizer
# ---------------------------
print("💾 Saving Model and TF-IDF Vectorizer...")
with open("logistic_regression.pkl", "wb") as f:
    pickle.dump(model, f)

with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
print("✅ Model and Vectorizer Saved Successfully")

# ---------------------------
# 🔟 Example Testing - Predict on a Sample Email
# ---------------------------
sample_email = ["Subject: Congratulations! You Won $1,000,000\n\nDear Winner,\n\n"
                "You have been selected as the lucky winner of $1,000,000!\n"
                "To claim your prize, please send us your bank details.\n\n"
                "Claim now before the deadline expires.\n\nBest wishes,\nLottery Office"]

print("📩 Classifying Sample Email...")
sample_vectorized = vectorizer.transform(sample_email).toarray()
missing_features = np.zeros((1, X_train.shape[1] - sample_vectorized.shape[1]))
final_sample_features = np.hstack((sample_vectorized, missing_features))

prediction = model.predict(final_sample_features)[0]

print("📩 Sample Email Classification Result:")
print(f"🟢 Safe Email" if prediction == 0 else "🔴 Phishing Email")
print("✅ Classification Completed")

# ---------------------------
# 🌍 Flask API for Real-Time Email Classification
# ---------------------------
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    """API Endpoint for real-time email classification"""
    data = request.json.get('email', "")

    processed_email = preprocess_text(data)
    vectorized_email = vectorizer.transform([processed_email]).toarray()
    email_features = np.zeros((1, X_train.shape[1] - vectorized_email.shape[1]))
    final_features = np.hstack((vectorized_email, email_features))

    prediction = model.predict(final_features)[0]
    return jsonify({"prediction": "Phishing" if prediction == 1 else "Safe"})

if __name__ == '__main__':
    print("🚀 Starting Flask API...")
    app.run(debug=True)
