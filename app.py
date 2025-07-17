# from flask import Flask, render_template, request, redirect, url_for, flash, session, make_response
# import os
# import uuid
# import cv2
# import numpy as np
# import re # Import regex module
# from werkzeug.utils import secure_filename
# from tensorflow.keras.models import load_model
# from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.preprocessing import LabelEncoder
# import joblib
# from scipy.ndimage import generic_filter
# from sklearn.cluster import KMeans
# from pymongo import MongoClient
# from werkzeug.security import generate_password_hash, check_password_hash
# from datetime import datetime
# from bson import ObjectId
# from fpdf import FPDF
# from flask_mail import Mail, Message

# # === Flask App Setup ===
# app = Flask(__name__)
# app.secret_key = 'lungcancersecretkey'
# UPLOAD_FOLDER = 'static/uploads'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# # === Load ML Models ===
# # (Your model loading code remains the same)
# model = load_model("step_outputs_vgg_and_ann/combined_classifier.h5")
# lda = joblib.load("step_outputs_vgg_and_ann/lda.pkl")
# le = joblib.load("step_outputs_vgg_and_ann/label_encoder.pkl")
# vgg_base = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
# vgg_base.trainable = False

# # === MongoDB setup ===
# client = MongoClient("mongodb://localhost:27017/")
# db = client['lung_cancer_detection']
# users_collection = db['users']
# predictions_collection = db['predictions']

# # --- Flask-Mail Configuration ---
# app.config['MAIL_SERVER'] = 'smtp.gmail.com'
# app.config['MAIL_PORT'] = 587
# app.config['MAIL_USE_TLS'] = True
# app.config['MAIL_USERNAME'] = 'koundinyejiten@gmail.com'
# app.config['MAIL_PASSWORD'] = 'ytgm wxgd lvwy mbmp'
# app.config['MAIL_DEFAULT_SENDER'] = 'koundinyejiten@gmail.com'
# mail = Mail(app)


# # === Utility Functions ===
# # (Your utility functions remain the same)
# def geometric_mean_filter(image, size=3):
#     return generic_filter(image + 1e-5, lambda x: np.exp(np.mean(np.log(x))), size=(size, size))

# def kmeans_segment(image, k=2):
#     pixels = image.reshape(-1, 1).astype(np.float32)
#     kmeans = KMeans(n_clusters=k, n_init=10, random_state=42).fit(pixels)
#     return kmeans.labels_.reshape(image.shape)

# def preprocess_image(path, upload_id):
#     img = cv2.imread(path)
#     if img is None:
#         raise ValueError("Failed to read uploaded image. Check file format or corrupt file.")
#     original_resized = cv2.resize(img, (128, 128))
#     original_path = os.path.join(UPLOAD_FOLDER, f"{upload_id}_original.png")
#     cv2.imwrite(original_path, original_resized)
#     gray = cv2.cvtColor(original_resized, cv2.COLOR_BGR2GRAY)
#     filtered = geometric_mean_filter(gray)
#     filtered_path = os.path.join(UPLOAD_FOLDER, f"{upload_id}_filtered.png")
#     cv2.imwrite(filtered_path, filtered.astype(np.uint8))
#     segmented = kmeans_segment(filtered)
#     segmented_vis = (segmented * 255).astype(np.uint8)
#     segmented_path = os.path.join(UPLOAD_FOLDER, f"{upload_id}_segmented.png")
#     cv2.imwrite(segmented_path, segmented_vis)
#     flat_segmented = segmented.flatten().reshape(1, -1)
#     lda_feature = lda.transform(flat_segmented)
#     rgb_img = np.expand_dims(original_resized.astype(np.float32), axis=0)
#     vgg_input = preprocess_input(rgb_img)
#     vgg_feature = vgg_base.predict(vgg_input).reshape(1, -1)
#     combined_features = np.hstack([lda_feature, vgg_feature])
#     return combined_features, original_path, filtered_path, segmented_path


# # === Registration Route with Server-Side Validation ===
# @app.route('/register', methods=['GET', 'POST'])
# def register():
#     if request.method == 'POST':
#         fullname = request.form.get('fullname', '').strip()
#         email = request.form.get('email', '').strip().lower()
#         password = request.form.get('password', '')
#         confirm_password = request.form.get('confirm_password', '')

#         # --- Server-Side Validation Logic ---
#         # Name validation: only letters and spaces
#         if not re.match(r"^[A-Za-z\s]+$", fullname):
#             flash("Full name can only contain letters and spaces.", "danger")
#             return redirect(url_for('register'))

#         # Password complexity validation
#         if len(password) < 8:
#             flash("Password must be at least 8 characters long.", "danger")
#             return redirect(url_for('register'))
#         if not re.search(r"[a-z]", password):
#             flash("Password must contain a lowercase letter.", "danger")
#             return redirect(url_for('register'))
#         if not re.search(r"[A-Z]", password):
#             flash("Password must contain an uppercase letter.", "danger")
#             return redirect(url_for('register'))
#         if not re.search(r"\d", password):
#             flash("Password must contain a number.", "danger")
#             return redirect(url_for('register'))
#         if not re.search(r"[!@#$%^&*]", password):
#             flash("Password must contain a special character (!@#$%^&*).", "danger")
#             return redirect(url_for('register'))
#         # --- End of Server-Side Validation ---

#         if password != confirm_password:
#             flash("Passwords do not match.", "danger")
#             return redirect(url_for('register'))

#         if users_collection.find_one({"email": email}):
#             flash("Email already registered.", "warning")
#             return redirect(url_for('register'))

#         hashed_password = generate_password_hash(password)
#         user_data = {
#             "fullname": fullname,
#             "email": email,
#             "password": hashed_password
#         }

#         users_collection.insert_one(user_data)
#         flash("Registration successful. Please login.", "success")
#         return redirect(url_for('login'))

#     return render_template('register.html')
    
# # (The rest of your app.py routes remain the same)

# @app.route('/', methods=['GET', 'POST'])
# def home():
#     if request.method == 'POST':
#         file = request.files.get('ctImage')
#         if file and file.filename != '':
#             try:
#                 upload_id = str(uuid.uuid4().hex)
#                 filename = secure_filename(file.filename)
#                 filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{upload_id}_{filename}")
#                 file.save(filepath)

#                 features, original_path, filtered_path, segmented_path = preprocess_image(filepath, upload_id)
#                 prediction = model.predict(features)
#                 confidence = float(np.max(prediction)) * 100
#                 predicted_label = le.inverse_transform([np.argmax(prediction)])[0]
                
#                 # --- NEW: Get patient details from form ---
#                 patient_name = request.form.get('patientName')
#                 patient_age = request.form.get('patientAge')
#                 patient_gender = request.form.get('patientGender')


#                 # --- Save prediction to MongoDB ---
#                 prediction_data = {
#                     "user_id": session.get('user_id'),
#                     "user_name": session.get('user_name'),
#                     "patient_name": patient_name, # New
#                     "patient_age": patient_age,   # New
#                     "patient_gender": patient_gender, # New
#                     "original_image": os.path.basename(original_path),
#                     "filtered_image": os.path.basename(filtered_path),
#                     "segmented_image": os.path.basename(segmented_path),
#                     "prediction": predicted_label,
#                     "confidence": round(confidence, 2),
#                     "timestamp": datetime.utcnow()
#                 }
#                 result = predictions_collection.insert_one(prediction_data)
#                 prediction_id = result.inserted_id

#                 return render_template(
#                     'result.html',
#                     prediction=predicted_label,
#                     accuracy=round(confidence, 2),
#                     original=os.path.basename(original_path),
#                     filtered=os.path.basename(filtered_path),
#                     segmented=os.path.basename(segmented_path),
#                     prediction_id=prediction_id # Pass prediction ID to result page
#                 )
#             except Exception as e:
#                 flash(f"Error during processing: {str(e)}", "danger")
#                 return redirect(url_for('home'))
#         else:
#             flash("No file selected or empty file uploaded.", "warning")
#             return redirect(url_for('home'))

#     return render_template('home.html')

# # === NEW: PDF Report Generation Route ===
# @app.route('/download_report/<prediction_id>')
# def download_report(prediction_id):
#     if 'user_id' not in session:
#         flash("You need to login first.", "warning")
#         return redirect(url_for('login'))

#     prediction = predictions_collection.find_one({"_id": ObjectId(prediction_id), "user_id": session['user_id']})
#     if not prediction:
#         flash("Prediction not found or you do not have permission to view it.", "danger")
#         return redirect(url_for('history'))

#     pdf = FPDF()
#     pdf.add_page()
#     pdf.set_font("Arial", "B", 16)
    
#     pdf.cell(200, 10, txt="Lung Cancer Detection Report", ln=True, align="C")
#     pdf.ln(10)

#     pdf.set_font("Arial", "", 12)
#     pdf.cell(200, 10, txt=f"Patient Name: {prediction.get('patient_name', 'N/A')}", ln=True)
#     pdf.cell(200, 10, txt=f"Patient Age: {prediction.get('patient_age', 'N/A')}", ln=True)
#     pdf.cell(200, 10, txt=f"Patient Gender: {prediction.get('patient_gender', 'N/A')}", ln=True)
#     pdf.cell(200, 10, txt=f"Date: {prediction.get('timestamp').strftime('%Y-%m-%d %H:%M:%S')} UTC", ln=True)
#     pdf.ln(10)

#     pdf.set_font("Arial", "B", 14)
#     pdf.cell(200, 10, txt=f"Prediction: {prediction.get('prediction')}", ln=True)
#     pdf.cell(200, 10, txt=f"Confidence: {prediction.get('confidence')}%", ln=True)
#     pdf.ln(10)

#     # Add images
#     pdf.set_font("Arial", "B", 12)
#     pdf.cell(200, 10, "Original Image", ln=True)
#     pdf.image(os.path.join(UPLOAD_FOLDER, prediction['original_image']), w=100)
#     pdf.ln(5)

#     pdf.cell(200, 10, "Filtered Image", ln=True)
#     pdf.image(os.path.join(UPLOAD_FOLDER, prediction['filtered_image']), w=100)
#     pdf.ln(5)

#     pdf.cell(200, 10, "Segmented Image", ln=True)
#     pdf.image(os.path.join(UPLOAD_FOLDER, prediction['segmented_image']), w=100)

#     response = make_response(pdf.output(dest='S').encode('latin-1'))
#     response.headers.set('Content-Disposition', 'attachment', filename='report.pdf')
#     response.headers.set('Content-Type', 'application/pdf')
#     return response

# @app.route('/login', methods=['GET', 'POST'])
# def login():
#     if request.method == 'POST':
#         email = request.form['email'].strip().lower()
#         password = request.form['password']

#         user = users_collection.find_one({'email': email})

#         if user and check_password_hash(user['password'], password):
#             session['user_id'] = str(user['_id'])
#             session['user_name'] = user['fullname']
#             flash('Login successful!', 'success')
#             return redirect(url_for('index'))
#         else:
#             flash('Invalid credentials. Please try again.', 'login')

#     return render_template('login.html')

# @app.route('/index', methods=['GET', 'POST'])
# def index():
#     if 'user_id' not in session:
#         flash("You need to login first.", "warning")
#         return redirect(url_for('login'))

#     if request.method == 'POST':
#         file = request.files.get('ctImage')
#         if file and file.filename != '':
#             try:
#                 upload_id = str(uuid.uuid4().hex)
#                 filename = secure_filename(file.filename)
#                 filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{upload_id}_{filename}")
#                 file.save(filepath)

#                 features, original_path, filtered_path, segmented_path = preprocess_image(filepath, upload_id)
#                 prediction = model.predict(features)
#                 confidence = float(np.max(prediction)) * 100
#                 predicted_label = le.inverse_transform([np.argmax(prediction)])[0]
                
#                 patient_name = request.form.get('patientName')
#                 patient_age = request.form.get('patientAge')
#                 patient_gender = request.form.get('patientGender')

#                 prediction_data = {
#                     "user_id": session.get('user_id'),
#                     "user_name": session.get('user_name'),
#                     "patient_name": patient_name,
#                     "patient_age": patient_age,
#                     "patient_gender": patient_gender,
#                     "original_image": os.path.basename(original_path),
#                     "filtered_image": os.path.basename(filtered_path),
#                     "segmented_image": os.path.basename(segmented_path),
#                     "prediction": predicted_label,
#                     "confidence": round(confidence, 2),
#                     "timestamp": datetime.utcnow()
#                 }
#                 result = predictions_collection.insert_one(prediction_data)
#                 prediction_id = result.inserted_id

#                 return render_template(
#                     'result.html',
#                     prediction=predicted_label,
#                     accuracy=round(confidence, 2),
#                     original=os.path.basename(original_path),
#                     filtered=os.path.basename(filtered_path),
#                     segmented=os.path.basename(segmented_path),
#                     prediction_id=prediction_id
#                 )
#             except Exception as e:
#                 flash(f"Error during processing: {str(e)}", "danger")
#                 return redirect(url_for('index'))
#         else:
#             flash("No file selected or empty file uploaded.", "warning")
#             return redirect(url_for('index'))

#     return render_template('index.html')


# @app.route('/history')
# def history():
#     if 'user_id' not in session:
#         flash("You need to login first.", "warning")
#         return redirect(url_for('login'))

#     user_id = session['user_id']
#     predictions = list(predictions_collection.find({"user_id": user_id}).sort("timestamp", -1))

#     history_data = []
#     for p in predictions:
#         history_data.append({
#             "id": str(p.get("_id")),
#             "filename": p.get("original_image", ""),
#             "result": p.get("prediction", ""),
#             "date": p.get("timestamp", "").strftime("%Y-%m-%d %H:%M"),
#             "user": p.get("user_name", "Unknown"),
#             "patient_name": p.get("patient_name", "N/A")
#         })

#     return render_template('history.html', history=history_data)


# @app.route('/delete_history/<prediction_id>', methods=['POST'])
# def delete_history(prediction_id):
#     if 'user_id' not in session:
#         flash("You need to login first.", "warning")
#         return redirect(url_for('login'))

#     result = predictions_collection.delete_one({
#         "_id": ObjectId(prediction_id),
#         "user_id": session['user_id']
#     })
#     if result.deleted_count:
#         flash("Prediction deleted successfully.", "success")
#     else:
#         flash("Failed to delete prediction.", "danger")
#     return redirect(url_for('history'))

# @app.route('/contact', methods=['GET', 'POST'])
# def contact():
#     if request.method == 'POST':
#         name = request.form.get('name')
#         email = request.form.get('email')
#         subject = request.form.get('subject')
#         message = request.form.get('message')

#         msg = Message(
#             subject=f"[Contact] {subject}",
#             recipients=['koundinyejiten@gmail.com'],
#             body=f"Name: {name}\nEmail: {email}\n\nMessage:\n{message}"
#         )
#         try:
#             mail.send(msg)
#             flash("Your message has been sent successfully!", "success")
#         except Exception as e:
#             flash(f"Failed to send message: {str(e)}", "danger")
#         return redirect(url_for('contact'))

#     return render_template('contact.html')


# @app.route('/logout')
# def logout():
#     session.clear()
#     flash("You have been logged out.", "info")
#     return redirect(url_for('login'))


# @app.route('/about')
# def about():
#     return render_template('about.html')


# if __name__ == '__main__':
#     app.run(debug=True)





from flask import Flask, render_template, request, redirect, url_for, flash, session, make_response
import os
import uuid
import cv2
import numpy as np
import re # For validation
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
from bson import ObjectId
from fpdf import FPDF # For PDF generation
from flask_mail import Mail, Message

# === Flask App Setup ===
app = Flask(__name__)
app.secret_key = 'lungcancersecretkey' # Should be a long, random string in production
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Flask-Mail Configuration ---
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
# IMPORTANT: Use environment variables for credentials in a real project
app.config['MAIL_USERNAME'] = 'koundinyejiten@gmail.com'
app.config['MAIL_PASSWORD'] = 'ytgm wxgd lvwy mbmp' # This is an app password, which is good practice
app.config['MAIL_DEFAULT_SENDER'] = 'koundinyejiten@gmail.com'
mail = Mail(app)

# === Load ML Models ===
# Using a try-except block for robust error handling if models are missing
try:
    model = load_model("step_outputs_vgg_and_ann/combined_classifier.h5")
    lda = joblib.load("step_outputs_vgg_and_ann/lda.pkl")
    le = joblib.load("step_outputs_vgg_and_ann/label_encoder.pkl")
except IOError as e:
    # In a real app, you would log this error extensively
    print(f"Error loading machine learning models: {e}")
    # Depending on the app's requirements, you might want to exit or disable the prediction feature.
    # For now, we'll let it run and it will fail on prediction.

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

# === Routes ===

@app.route('/')
def home_page():
    # This route will show the main landing page.
    # If the user is already logged in, they could be redirected to '/index', but for simplicity, we'll show the landing page.
    return render_template('home.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        fullname = request.form.get('fullname', '').strip()
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')

        # Server-Side Validation Logic
        if not re.match(r"^[A-Za-z\s]+$", fullname):
            flash("Full name can only contain letters and spaces.", "danger")
            return redirect(url_for('register'))
        if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
            flash("Invalid email address.", "danger")
            return redirect(url_for('register'))
        if len(password) < 8 or not re.search(r"[a-z]", password) or not re.search(r"[A-Z]", password) or not re.search(r"\d", password) or not re.search(r"[!@#$%^&*]", password):
            flash("Password does not meet complexity requirements.", "danger")
            return redirect(url_for('register'))
        if password != confirm_password:
            flash("Passwords do not match.", "danger")
            return redirect(url_for('register'))
        if users_collection.find_one({"email": email}):
            flash("Email already registered.", "warning")
            return redirect(url_for('register'))

        hashed_password = generate_password_hash(password)
        users_collection.insert_one({"fullname": fullname, "email": email, "password": hashed_password})
        flash("Registration successful. Please login.", "success")
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session:
        return redirect(url_for('index'))
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        user = users_collection.find_one({'email': email})
        if user and check_password_hash(user['password'], password):
            session['user_id'] = str(user['_id'])
            session['user_name'] = user['fullname']
            flash('Login successful!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid email or password. Please try again.', 'danger')
    return render_template('login.html')

@app.route('/index', methods=['GET', 'POST'])
def index():
    if 'user_id' not in session:
        flash("You need to be logged in to access this page.", "warning")
        return redirect(url_for('login'))
    if request.method == 'POST':
        file = request.files.get('ctImage')
        patient_name = request.form.get('patientName', '').strip()
        if not file or file.filename == '':
            flash("No file selected. Please upload a CT scan.", "warning")
            return redirect(url_for('index'))
        if not patient_name:
            flash("Patient name is required.", "warning")
            return redirect(url_for('index'))
        try:
            upload_id = str(uuid.uuid4().hex)
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{upload_id}_{filename}")
            file.save(filepath)
            features, original_path, filtered_path, segmented_path = preprocess_image(filepath, upload_id)
            prediction = model.predict(features)
            confidence = float(np.max(prediction)) * 100
            predicted_label = le.inverse_transform([np.argmax(prediction)])[0]
            prediction_data = {
                "user_id": session.get('user_id'),
                "user_name": session.get('user_name'),
                "patient_name": patient_name,
                "patient_age": request.form.get('patientAge'),
                "patient_gender": request.form.get('patientGender'),
                "original_image": os.path.basename(original_path),
                "filtered_image": os.path.basename(filtered_path),
                "segmented_image": os.path.basename(segmented_path),
                "prediction": predicted_label,
                "confidence": round(confidence, 2),
                "timestamp": datetime.utcnow()
            }
            result = predictions_collection.insert_one(prediction_data)
            return render_template('result.html', prediction=predicted_label, accuracy=round(confidence, 2), original=os.path.basename(original_path), filtered=os.path.basename(filtered_path), segmented=os.path.basename(segmented_path), prediction_id=result.inserted_id)
        except Exception as e:
            flash(f"An error occurred during processing: {str(e)}", "danger")
            return redirect(url_for('index'))
    return render_template('index.html')

@app.route('/history')
def history():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    user_id = session['user_id']
    pipeline = [
        {"$match": {"user_id": user_id}},
        {"$group": {"_id": "$patient_name", "count": {"$sum": 1}}},
        {"$sort": {"_id": 1}}
    ]
    patients_cursor = predictions_collection.aggregate(pipeline)
    patients = [{"name": doc["_id"], "count": doc["count"]} for doc in patients_cursor]
    return render_template('history.html', patients=patients)

@app.route('/history/<patient_name>')
def patient_history(patient_name):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    user_id = session['user_id']
    predictions = list(predictions_collection.find({"user_id": user_id, "patient_name": patient_name}).sort("timestamp", -1))
    history_data = [{"id": str(p["_id"]), "filename": p.get("original_image"), "result": p.get("prediction"), "date": p.get("timestamp").strftime("%Y-%m-%d %H:%M")} for p in predictions]
    return render_template('patient_history.html', history=history_data, patient_name=patient_name)

@app.route('/delete_history/<prediction_id>', methods=['POST'])
def delete_history(prediction_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    prediction_to_delete = predictions_collection.find_one({"_id": ObjectId(prediction_id), "user_id": session['user_id']})
    if not prediction_to_delete:
        flash("Prediction not found or permission denied.", "danger")
        return redirect(url_for('history'))
    patient_name = prediction_to_delete.get('patient_name')
    # Here you might want to delete the associated image files from 'static/uploads' as well to save space
    # os.remove(os.path.join(UPLOAD_FOLDER, prediction_to_delete['original_image']))
    # os.remove(os.path.join(UPLOAD_FOLDER, prediction_to_delete['filtered_image']))
    # os.remove(os.path.join(UPLOAD_FOLDER, prediction_to_delete['segmented_image']))
    predictions_collection.delete_one({"_id": ObjectId(prediction_id)})
    flash("Prediction deleted successfully.", "success")
    return redirect(url_for('patient_history', patient_name=patient_name))

@app.route('/download_report/<prediction_id>')
def download_report(prediction_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    prediction = predictions_collection.find_one({"_id": ObjectId(prediction_id), "user_id": session['user_id']})
    if not prediction:
        flash("Report not found.", "danger")
        return redirect(url_for('history'))
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, txt="Lung Cancer Detection Report", ln=True, align="C")
    pdf.ln(10)
    pdf.set_font("Arial", "", 12)
    pdf.cell(200, 8, txt=f"Patient Name: {prediction.get('patient_name', 'N/A')}", ln=True)
    pdf.cell(200, 8, txt=f"Patient Age: {prediction.get('patient_age', 'N/A')}", ln=True)
    pdf.cell(200, 8, txt=f"Patient Gender: {prediction.get('patient_gender', 'N/A')}", ln=True)
    pdf.cell(200, 8, txt=f"Date: {prediction.get('timestamp').strftime('%Y-%m-%d %H:%M:%S')} UTC", ln=True)
    pdf.ln(10)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(200, 10, txt=f"Prediction: {prediction.get('prediction')}", ln=True)
    pdf.cell(200, 10, txt=f"Confidence: {prediction.get('confidence')}%", ln=True)
    pdf.ln(10)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(200, 10, "Original Image", ln=True)
    pdf.image(os.path.join(UPLOAD_FOLDER, prediction['original_image']), w=100)
    pdf.ln(5)
    pdf.cell(200, 10, "Filtered Image", ln=True)
    pdf.image(os.path.join(UPLOAD_FOLDER, prediction['filtered_image']), w=100)
    pdf.ln(5)
    pdf.cell(200, 10, "Segmented Image", ln=True)
    pdf.image(os.path.join(UPLOAD_FOLDER, prediction['segmented_image']), w=100)
    response = make_response(pdf.output(dest='S').encode('latin-1'))
    response.headers.set('Content-Disposition', 'attachment', filename=f'report_{prediction.get("patient_name", "").replace(" ", "_")}.pdf')
    response.headers.set('Content-Type', 'application/pdf')
    return response

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        subject = request.form.get('subject')
        message = request.form.get('message')
        # Basic validation
        if not all([name, email, subject, message]):
            flash("All fields are required.", "danger")
            return redirect(url_for('contact'))
        msg = Message(subject=f"[Contact Form] {subject}", recipients=['koundinyejiten@gmail.com'], body=f"From: {name} <{email}>\n\n{message}")
        try:
            mail.send(msg)
            flash("Your message has been sent successfully!", "success")
        except Exception as e:
            flash(f"Failed to send message: {str(e)}", "danger")
        return redirect(url_for('contact'))
    return render_template('contact.html')

@app.route('/logout')
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
