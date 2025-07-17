AI-Powered Lung Cancer Detection from CT Scans
This web application leverages a deep learning model to analyze chest CT scans and predict the likelihood of common types of lung cancer. It provides a user-friendly interface for patients and healthcare professionals to upload scans, view results, and manage a history of diagnoses.

✨ Features
Secure User Authentication: Users can register and log in to a secure account. Passwords are encrypted and validated for complexity.

CT Scan Upload & Analysis: A simple interface to upload patient details and a corresponding CT scan image.

AI-Powered Prediction: The backend processes the image through a sophisticated pipeline involving filtering, segmentation, and a VGG16-based deep learning model to classify the scan into one of four categories:

Adenocarcinoma

Large Cell Carcinoma

Squamous Cell Carcinoma

Normal (Healthy)

Detailed Results Page: Displays the prediction, confidence score, and the processed images (original, filtered, segmented).

Downloadable PDF Reports: Users can download a professional PDF report for each diagnosis, which includes patient details, results, and all associated images.

Organized Patient History:

A main history page lists all patients associated with the user's account.

Clicking on a patient shows a detailed, chronological view of all their past scans and results.

Interactive & Professional UI: The entire application is designed with a modern, clean, and responsive user interface for a professional, industry-level feel.

Contact Form: An integrated contact form that sends inquiries directly to the administrator's email.

🛠️ Tech Stack
Backend:

Framework: Flask

Database: MongoDB (with PyMongo)

Machine Learning: TensorFlow, Keras, Scikit-learn

Image Processing: OpenCV, SciPy

Frontend:

Templating: Jinja2

Styling: Bootstrap 5, Custom CSS

JavaScript: For client-side validation and interactivity

Deployment & Others:

PDF Generation: Pyfpdf

Email Service: Flask-Mail (with Gmail SMTP)

📋 Prerequisites
Before you begin, ensure you have the following installed on your local machine:

Python (3.8 or newer)

pip (Python package installer)

MongoDB Community Server

📧 Contact
Created by Jiten Koundinye & Yash Atre - feel free to reach out!

Jiten Koundinye: koundinyejiten@gmail.com

Yash Atre: atreyash10@gmail.com

This README was last updated on July 18, 2025.
