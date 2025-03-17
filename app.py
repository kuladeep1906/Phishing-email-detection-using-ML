import os
from flask import Flask, render_template, request
import main  # Import main.py where models are loaded
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Set upload folder (temporary storage)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Allowed file extensions
ALLOWED_EXTENSIONS = {"eml", "txt"}

def allowed_file(filename):
    """Check if uploaded file has an allowed extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    email_text = request.form.get("email_text", "").strip()
    file = request.files.get("file")

    # If user uploaded a file, extract its contents
    if file and file.filename != "":
        if allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)

            # Extract email content from file
            email_text = main.extract_email_content(file_path)
        else:
            return render_template("index.html", error="Invalid file type! Only .eml and .txt are allowed.")

    if not email_text:
        return render_template("index.html", error="No email content provided!")

    # Get prediction & confidence scores
    prediction, confidence, confidence_lr, confidence_rf, confidence_xgb, top_features = main.classify_email(email_text)

    return render_template(
        "index.html",
        email_text=email_text,
        prediction=prediction,
        confidence=confidence,
        confidence_lr=confidence_lr,
        confidence_rf=confidence_rf,
        confidence_xgb=confidence_xgb,
        top_features=top_features,  
        is_phishing=(prediction == "Phishing")
    )

if __name__ == "__main__":
    print("🚀 Starting Flask API...")
    app.run(debug=True)
