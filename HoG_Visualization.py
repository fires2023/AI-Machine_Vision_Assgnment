import numpy as np
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from tensorflow.keras.datasets import cifar10

# =========================================================
# 1. Load CIFAR-10 Dataset
# =========================================================
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# =========================================================
# 2. Convert RGB Images to Grayscale. It produces a 2D grayscale image for each sample.
# =========================================================
X_train_gray = np.array([
    np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
    for img in X_train
])

X_test_gray = np.array([
    np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
    for img in X_test
])

# =========================================================
# 3. Extract HoG Features
# =========================================================
def extract_hog_features(images):
    hog_features = []

    for img in images:
        features = hog(
            img,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            visualize=False
        )

        hog_features.append(features)

    return np.array(hog_features)

print("Extracting HoG features from training images...")
X_train_hog = extract_hog_features(X_train_gray)

print("Extracting HoG features from test images...")
X_test_hog = extract_hog_features(X_test_gray)

# =========================================================
# 4. Train Multi-class Linear SVM Classifier
# =========================================================
print("Training Linear SVM classifier...")

svm_classifier = LinearSVC(
    C=1.0,
    dual=False,
    random_state=42,
    max_iter=5000
)

svm_classifier.fit(X_train_hog, y_train.ravel())

# =========================================================
# 5. Make Predictions
# =========================================================
print("Making predictions...")

y_pred = svm_classifier.predict(X_test_hog)

# =========================================================
# 6. Evaluate Classifier Performance
# =========================================================

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\n========================================")
print("CLASSIFIER PERFORMANCE")
print("========================================")

print(f"\nAccuracy: {accuracy * 100:.2f}%")

# CIFAR-10 Class Names
class_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]

# Classification Report
print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=class_names
))

# Confusion Matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)

print(cm)

# =========================================================
# 7. Optional: Display Sample Predictions
# =========================================================
print("\nSample Predictions:")

for i in range(10):
    true_label = class_names[y_test[i][0]]
    predicted_label = class_names[y_pred[i]]

    print(
        f"Image {i+1}: "
        f"True = {true_label}, "
        f"Predicted = {predicted_label}"
    )import numpy as np
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from tensorflow.keras.datasets import cifar10

# =========================================================
# 1. Load CIFAR-10 Dataset
# =========================================================
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# =========================================================
# 2. Convert RGB Images to Grayscale
# =========================================================
X_train_gray = np.array([
    np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
    for img in X_train
])

X_test_gray = np.array([
    np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
    for img in X_test
])

# =========================================================
# 3. Extract HoG Features
# =========================================================
def extract_hog_features(images):
    hog_features = []

    for img in images:
        features = hog(
            img,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            visualize=False
        )

        hog_features.append(features)

    return np.array(hog_features)

print("Extracting HoG features from training images...")
X_train_hog = extract_hog_features(X_train_gray)

print("Extracting HoG features from test images...")
X_test_hog = extract_hog_features(X_test_gray)

# =========================================================
# 4. Train Multi-class Linear SVM Classifier
# =========================================================
print("Training Linear SVM classifier...")

svm_classifier = LinearSVC(
    C=1.0,
    dual=False,
    random_state=42,
    max_iter=5000
)

svm_classifier.fit(X_train_hog, y_train.ravel())

# =========================================================
# 5. Make Predictions
# =========================================================
print("Making predictions...")

y_pred = svm_classifier.predict(X_test_hog)

# =========================================================
# 6. Evaluate Classifier Performance
# =========================================================

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\n========================================")
print("CLASSIFIER PERFORMANCE")
print("========================================")

print(f"\nAccuracy: {accuracy * 100:.2f}%")

# CIFAR-10 Class Names
class_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]

# Classification Report
print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=class_names
))

# Confusion Matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)

print(cm)

# =========================================================
# 7. Optional: Display Sample Predictions
# =========================================================
print("\nSample Predictions:")

for i in range(10):
    true_label = class_names[y_test[i][0]]
    predicted_label = class_names[y_pred[i]]

    print(
        f"Image {i+1}: "
        f"True = {true_label}, "
        f"Predicted = {predicted_label}"
    )
