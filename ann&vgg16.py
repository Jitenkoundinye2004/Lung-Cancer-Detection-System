import os
import cv2
import numpy as np
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.ndimage import generic_filter
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input


# === CONFIG ===
IMAGE_SIZE = (128, 128)
DATASET_DIR = "archive\\Data\\train"
MODEL_DIR = "step_outputs_vgg_and_ann"
OUTPUT_DIR = "step_outputs_vgg_and_ann"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Load grayscale images for ANN pipeline ===
def load_gray_images(folder_path):
    images, labels = [], []
    for label in os.listdir(folder_path):
        label_path = os.path.join(folder_path, label)
        for filename in os.listdir(label_path):
            img = cv2.imread(os.path.join(label_path, filename), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, IMAGE_SIZE)
                images.append(img)
                labels.append(label)
    return np.array(images), np.array(labels)

# === Load RGB images for VGG pipeline ===
def load_rgb_images(folder_path):
    images = []
    for label in os.listdir(folder_path):
        label_path = os.path.join(folder_path, label)
        for filename in os.listdir(label_path):
            img = cv2.imread(os.path.join(label_path, filename))
            if img is not None:
                img = cv2.resize(img, IMAGE_SIZE)
                images.append(img)
    return np.array(images)

print("\nLoading images...")
X_gray, y_labels = load_gray_images(DATASET_DIR)

X_rgb = []
for label in os.listdir(DATASET_DIR):
    label_path = os.path.join(DATASET_DIR, label)
    for filename in os.listdir(label_path):
        img = cv2.imread(os.path.join(label_path, filename))
        if img is not None:
            img = cv2.resize(img, IMAGE_SIZE)
            X_rgb.append(img)
X_rgb = np.array(X_rgb)

print(f"Loaded {len(X_gray)} grayscale images and {len(X_rgb)} RGB images.")

# === Save sample original grayscale image ===
sample_idx = 0  # Index of sample to save
cv2.imwrite(os.path.join(OUTPUT_DIR, "step1_original_sample.png"), X_gray[sample_idx])

# === Preprocessing for ANN pipeline ===
def safe_geometric_mean_filter(image, size=3):
    return generic_filter(image + 1e-5, lambda x: np.exp(np.mean(np.log(x))), size=(size, size))

print("\nApplying geometric mean filtering...")
X_filtered = np.array([safe_geometric_mean_filter(img) for img in X_gray])

# === Save filtered image sample ===
cv2.imwrite(os.path.join(OUTPUT_DIR, "step2_filtered_sample.png"), X_filtered[sample_idx].astype(np.uint8))

def kmeans_segmentation(images, k=2):
    segmented = []
    for img in images:
        pixel_values = img.reshape((-1, 1)).astype(np.float32)
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42).fit(pixel_values)
        clustered = kmeans.labels_.reshape(img.shape)
        segmented.append(clustered)
    return np.array(segmented)

print("\nApplying K-means segmentation...")
X_segmented = kmeans_segmentation(X_filtered)

# === Save segmented image sample ===
segmented_visual = (X_segmented[sample_idx] * 255).astype(np.uint8)
cv2.imwrite(os.path.join(OUTPUT_DIR, "step3_segmented_sample.png"), segmented_visual)

X_flat = X_segmented.reshape(len(X_segmented), -1)

print("\nEncoding labels...")
le = LabelEncoder()
y_encoded = le.fit_transform(y_labels)

print("\nPerforming LDA feature extraction...")
lda = LinearDiscriminantAnalysis()
X_lda = lda.fit_transform(X_flat, y_encoded)

# === Preprocessing for VGG pipeline ===
print("\nPreprocessing RGB images for VGG16...")
X_vgg = preprocess_input(X_rgb.astype(np.float32))  # preprocess for VGG16

# === Extract features from VGG16 ===
print("\nExtracting features from VGG16...")
vgg_base = VGG16(weights='imagenet', include_top=False, input_shape=(*IMAGE_SIZE, 3))
vgg_base.trainable = False  # freeze VGG weights

vgg_features = vgg_base.predict(X_vgg)
vgg_features_flat = vgg_features.reshape(vgg_features.shape[0], -1)

# === Combine LDA and VGG features ===
print("\nCombining LDA and VGG features...")
X_combined = np.hstack([X_lda, vgg_features_flat])

# === Build combined model ===
num_classes = len(np.unique(y_encoded))
model = models.Sequential([
    layers.Input(shape=(X_combined.shape[1],)),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print("\nTraining combined model...")
model.fit(X_combined, y_encoded, epochs=20, batch_size=32, verbose=1)

# === Evaluate and save ===
print("\nEvaluating combined model...")
y_pred_prob = model.predict(X_combined)
y_pred = np.argmax(y_pred_prob, axis=1)
acc = accuracy_score(y_encoded, y_pred)
print("Accuracy:", acc)
print(classification_report(y_encoded, y_pred, target_names=le.classes_))

conf_mat = confusion_matrix(y_encoded, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues",
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title(f"Confusion Matrix (Accuracy: {acc*100:.2f}%)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "confusion_matrix_combined.png"))
plt.close()

print("\nSaving model files...")
model.save(os.path.join(MODEL_DIR, "combined_classifier.h5"))
joblib.dump(lda, os.path.join(MODEL_DIR, "lda.pkl"))
joblib.dump(le, os.path.join(MODEL_DIR, "label_encoder.pkl"))
print("✅ All combined model files saved.")
