from flask import Flask, render_template, request, redirect, url_for, flash, session
import os
import uuid
import cv2
import numpy as np
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder
import joblib
from scipy.ndimage import generic_filter
from sklearn.cluster import KMeans
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

# === Flask App Setup ===
app = Flask(__name__)
app.secret_key = 'lungcancersecretkey'
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# === Load ML Models ===
model = load_model("step_outputs_vgg_and_ann/combined_classifier.h5")
lda = joblib.load("step_outputs_vgg_and_ann/lda.pkl")
le = joblib.load("step_outputs_vgg_and_ann/label_encoder.pkl")

# === Load VGG16 for Feature Extraction ===
vgg_base = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
vgg_base.trainable = False

# === MongoDB setup ===
client = MongoClient("mongodb://localhost:27017/")
db = client['lung_cancer_detection']
users_collection = db['users']
predictions_collection = db['predictions']

# === Utility Functions ===
def geometric_mean_filter(image, size=3):
    return generic_filter(image + 1e-5, lambda x: np.exp(np.mean(np.log(x))), size=(size, size))

def kmeans_segment(image, k=2):
    pixels = image.reshape(-1, 1).astype(np.float32)
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42).fit(pixels)
    return kmeans.labels_.reshape(image.shape)

def preprocess_image(path, upload_id):
    img = cv2.imread(path)
    if img is None:
        raise ValueError("Failed to read uploaded image. Check file format or corrupt file.")

    original_resized = cv2.resize(img, (128, 128))
    original_path = os.path.join(UPLOAD_FOLDER, f"{upload_id}_original.png")
    cv2.imwrite(original_path, original_resized)

    gray = cv2.cvtColor(original_resized, cv2.COLOR_BGR2GRAY)
    filtered = geometric_mean_filter(gray)
    filtered_path = os.path.join(UPLOAD_FOLDER, f"{upload_id}_filtered.png")
    cv2.imwrite(filtered_path, filtered.astype(np.uint8))

    segmented = kmeans_segment(filtered)
    segmented_vis = (segmented * 255).astype(np.uint8)
    segmented_path = os.path.join(UPLOAD_FOLDER, f"{upload_id}_segmented.png")
    cv2.imwrite(segmented_path, segmented_vis)

    flat_segmented = segmented.flatten().reshape(1, -1)
    lda_feature = lda.transform(flat_segmented)

    rgb_img = np.expand_dims(original_resized.astype(np.float32), axis=0)
    vgg_input = preprocess_input(rgb_img)
    vgg_feature = vgg_base.predict(vgg_input).reshape(1, -1)

    combined_features = np.hstack([lda_feature, vgg_feature])

    return combined_features, original_path, filtered_path, segmented_path


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        file = request.files.get('ctImage')
        if file and file.filename != '':
            try:
                upload_id = str(uuid.uuid4().hex)
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{upload_id}_{filename}")
                file.save(filepath)

                features, original_path, filtered_path, segmented_path = preprocess_image(filepath, upload_id)
                prediction = model.predict(features)
                confidence = float(np.max(prediction)) * 100
                predicted_label = le.inverse_transform([np.argmax(prediction)])[0]

                # --- Save prediction to MongoDB ---
                predictions_collection.insert_one({
                    "user_id": session.get('user_id'),  # None if not logged in
                    "user_name": session.get('user_name'),  # None if not logged in
                    "original_image": os.path.basename(original_path),
                    "filtered_image": os.path.basename(filtered_path),
                    "segmented_image": os.path.basename(segmented_path),
                    "prediction": predicted_label,
                    "confidence": round(confidence, 2),
                    "timestamp": datetime.utcnow()
                })
                # --- End save ---

                return render_template(
                    'result.html',
                    prediction=predicted_label,
                    accuracy=round(confidence, 2),
                    original=os.path.basename(original_path),
                    filtered=os.path.basename(filtered_path),
                    segmented=os.path.basename(segmented_path)
                )
            except Exception as e:
                flash(f"Error during processing: {str(e)}", "danger")
                return redirect(url_for('home'))
        else:
            flash("No file selected or empty file uploaded.", "warning")
            return redirect(url_for('home'))

    return render_template('home.html')
# === Registration Route ===
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        fullname = request.form['fullname'].strip()
        email = request.form['email'].strip().lower()
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if password != confirm_password:
            flash("Passwords do not match.", "danger")
            return render_template('register.html')

        if users_collection.find_one({"email": email}):
            flash("Email already registered.", "warning")
            return render_template('register.html')

        hashed_password = generate_password_hash(password)
        user_data = {
            "fullname": fullname,
            "email": email,
            "password": hashed_password
        }

        users_collection.insert_one(user_data)
        flash("Registration successful. Please login.", "success")
        return redirect(url_for('login'))

    return render_template('register.html')

# === Login Route ===
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email'].strip().lower()
        password = request.form['password']

        user = users_collection.find_one({'email': email})

        if user and check_password_hash(user['password'], password):
            session['user_id'] = str(user['_id'])
            session['user_name'] = user['fullname']
            flash('Login successful!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid credentials. Please try again.', 'danger')

    return render_template('login.html')



# === Index Route (after login) ===
@app.route('/index', methods=['GET', 'POST'])
def index():
    if 'user_id' not in session:
        flash("You need to login first.", "warning")
        return redirect(url_for('login'))

    if request.method == 'POST':
        file = request.files.get('ctImage')
        if file and file.filename != '':
            try:
                upload_id = str(uuid.uuid4().hex)
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{upload_id}_{filename}")
                file.save(filepath)

                features, original_path, filtered_path, segmented_path = preprocess_image(filepath, upload_id)
                prediction = model.predict(features)
                confidence = float(np.max(prediction)) * 100
                predicted_label = le.inverse_transform([np.argmax(prediction)])[0]

                   # --- Save prediction to MongoDB ---
                predictions_collection.insert_one({
                    "user_id": session.get('user_id'),
                    "user_name": session.get('user_name'),
                    "original_image": os.path.basename(original_path),
                    "filtered_image": os.path.basename(filtered_path),
                    "segmented_image": os.path.basename(segmented_path),
                    "prediction": predicted_label,
                    "confidence": round(confidence, 2),
                    "timestamp": datetime.utcnow()
                })

                return render_template(
                    'result.html',
                    prediction=predicted_label,
                    accuracy=round(confidence, 2),
                    original=os.path.basename(original_path),
                    filtered=os.path.basename(filtered_path),
                    segmented=os.path.basename(segmented_path)
                )
            except Exception as e:
                flash(f"Error during processing: {str(e)}", "danger")
                return redirect(url_for('index'))
        else:
            flash("No file selected or empty file uploaded.", "warning")
            return redirect(url_for('index'))

    return render_template('index.html')



from bson import ObjectId

@app.route('/history')
def history():
    if 'user_id' not in session:
        flash("You need to login first.", "warning")
        return redirect(url_for('login'))

    user_id = session['user_id']
    predictions = list(predictions_collection.find({"user_id": user_id}).sort("timestamp", -1))

    history = []
    for p in predictions:
        history.append({
            "id": str(p.get("_id")),  # Pass the MongoDB document ID
            "filename": p.get("original_image", ""),
            "result": p.get("prediction", ""),
            "date": p.get("timestamp", "").strftime("%Y-%m-%d %H:%M"),
            "user": p.get("user_name", "Unknown")
        })

    return render_template('history.html', history=history)

from bson import ObjectId

@app.route('/delete_history/<prediction_id>', methods=['POST'])
def delete_history(prediction_id):
    if 'user_id' not in session:
        flash("You need to login first.", "warning")
        return redirect(url_for('login'))

    # Only allow deleting user's own predictions
    result = predictions_collection.delete_one({
        "_id": ObjectId(prediction_id),
        "user_id": session['user_id']
    })
    if result.deleted_count:
        flash("Prediction deleted successfully.", "success")
    else:
        flash("Failed to delete prediction.", "danger")
    return redirect(url_for('history'))

# ...existing imports...
from flask_mail import Mail, Message

# --- Flask-Mail Configuration ---
app.config['MAIL_SERVER'] = 'smtp.gmail.com'  # or your SMTP server
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] =os.getenv("EMAIL")     # admin email
app.config['MAIL_PASSWORD'] = 'ytgm wxgd lvwy mbmp'
app.config['MAIL_DEFAULT_SENDER'] =os.getenv("EMAIL")

mail = Mail(app)

# --- Contact Route ---
@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        subject = request.form.get('subject')
        message = request.form.get('message')

        # Compose email
        msg = Message(
            subject=f"[Contact] {subject}",
            recipients=os.getenv("EMAIL"),  # admin email
            body=f"Name: {name}\nEmail: {email}\n\nMessage:\n{message}"
        )
        try:
            mail.send(msg)
            flash("Your message has been sent successfully!", "success")
        except Exception as e:
            flash(f"Failed to send message: {str(e)}", "danger")
        return redirect(url_for('contact'))

    return render_template('contact.html')

# === Dashboard Route ===
@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        flash("You need to login first.", "warning")
        return redirect(url_for('login'))
    return f"Welcome, {session['user_name']}! This is your dashboard."

# === Logout Route ===
@app.route('/logout')
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for('home'))



# === Other Pages ===
@app.route('/about')
def about():
    return render_template('about.html')



@app.route('/forgot_password')
def forgot_password():
    return render_template('forgot_password.html')

# === Run Server ===
    if __name__ == '__main__':
        port = int(os.environ.get('PORT',5000))
        app.run(host='0.0.0.0', port=port, debug=True)
