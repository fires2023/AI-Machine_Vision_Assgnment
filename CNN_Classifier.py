import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# =========================================================
# 1. LOAD CIFAR-10 DATASET
# =========================================================
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

# =========================================================
# 2. NORMALIZE IMAGES (0–255 → 0–1)
# =========================================================
x_train = x_train / 255.0
x_test = x_test / 255.0

# =========================================================
# 3. CLASS LABELS
# =========================================================
class_names = [
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck'
]

# =========================================================
# 4. DISPLAY SAMPLE IMAGES
# =========================================================
plt.figure(figsize=(10, 5))

for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_train[i])
    plt.title(class_names[y_train[i][0]])
    plt.axis('off')

plt.tight_layout()
plt.savefig("samples.png")
plt.close()

# =========================================================
# 5. BUILD CNN MODEL
# =========================================================
model = models.Sequential([

    # First Convolution Block
    layers.Conv2D(
        32,
        (3, 3),
        activation='relu',
        input_shape=(32, 32, 3)
    ),

    layers.MaxPooling2D((2, 2)),

    # Second Convolution Block
    layers.Conv2D(
        64,
        (3, 3),
        activation='relu'
    ),

    layers.MaxPooling2D((2, 2)),

    # Third Convolution Block
    layers.Conv2D(
        128,
        (3, 3),
        activation='relu'
    ),

    # Fully Connected Layers
    layers.Flatten(),

    layers.Dense(
        128,
        activation='relu'
    ),

    layers.Dropout(0.5),

    layers.Dense(
        10,
        activation='softmax'
    )
])

# =========================================================
# 6. COMPILE MODEL
# =========================================================
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# =========================================================
# 7. TRAIN MODEL
# =========================================================
history = model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=64,
    validation_data=(x_test, y_test)
)

# =========================================================
# 8. SAVE MODEL
# =========================================================
model.save("cnn_cifar10_model.h5")

# =========================================================
# 9. PLOT TRAINING ACCURACY
# =========================================================
plt.figure(figsize=(8, 5))

plt.plot(
    history.history['accuracy'],
    label='Training Accuracy'
)

plt.plot(
    history.history['val_accuracy'],
    label='Validation Accuracy'
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("CNN Accuracy on CIFAR-10")
plt.legend()

plt.savefig("accuracy.png")
plt.close()

# =========================================================
# 10. PLOT TRAINING LOSS
# =========================================================
plt.figure(figsize=(8, 5))

plt.plot(
    history.history['loss'],
    label='Training Loss'
)

plt.plot(
    history.history['val_loss'],
    label='Validation Loss'
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("CNN Loss on CIFAR-10")
plt.legend()

plt.savefig("loss.png")
plt.close()

# =========================================================
# 11. EVALUATE MODEL
# =========================================================
print("\n========================================")
print("CLASSIFIER PERFORMANCE")
print("========================================")

test_loss, test_acc = model.evaluate(x_test, y_test)

print(f"\nTest Accuracy: {test_acc * 100:.2f}%")

# =========================================================
# 12. MAKE PREDICTIONS
# =========================================================
print("\nMaking predictions...")

y_pred = model.predict(x_test)

# Convert probabilities → class labels
y_pred_classes = np.argmax(y_pred, axis=1)

# Flatten true labels
y_true = y_test.flatten()

# =========================================================
# 13. CLASSIFICATION REPORT
# =========================================================
print("\nClassification Report:")

print(classification_report(
    y_true,
    y_pred_classes,
    target_names=class_names
))

# =========================================================
# 14. CONFUSION MATRIX
# =========================================================
print("\nConfusion Matrix:")

cm = confusion_matrix(y_true, y_pred_classes)

print(cm)

# =========================================================
# 15. SAMPLE PREDICTIONS
# =========================================================
print("\nSample Predictions:")

for i in range(10):

    true_label = class_names[y_true[i]]
    predicted_label = class_names[y_pred_classes[i]]

    print(
        f"Image {i+1}: "
        f"True = {true_label}, "
        f"Predicted = {predicted_label}"
    )

# =========================================================
# 16. DISPLAY RANDOM TEST PREDICTIONS
# =========================================================
plt.figure(figsize=(12, 6))

for i in range(10):

    plt.subplot(2, 5, i + 1)

    plt.imshow(x_test[i])

    true_label = class_names[y_true[i]]
    predicted_label = class_names[y_pred_classes[i]]

    plt.title(
        f"T: {true_label}\nP: {predicted_label}"
    )

    plt.axis('off')

plt.tight_layout()

plt.savefig("predictions.png")
plt.close()

print("\nSaved Files:")
print("1. samples.png")
print("2. accuracy.png")
print("3. loss.png")
print("4. predictions.png")
print("5. cnn_cifar10_model.h5")
