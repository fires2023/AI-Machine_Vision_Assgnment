import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


print("TensorFlow version:", tf.__version__)
print("GPU available:", tf.config.list_physical_devices('GPU'))


# ============================================================
# 1. Load CIFAR-10 Dataset
# ============================================================

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Flatten labels
y_train = y_train.flatten()
y_test  = y_test.flatten()

# Convert to float
x_train = x_train.astype("float32")
x_test  = x_test.astype("float32")

# ============================================================
# 2. Preprocessing (IMPORTANT for ResNet)
# ============================================================

x_train = preprocess_input(x_train)
x_test  = preprocess_input(x_test)

# ============================================================
# 3. Data Augmentation
# ============================================================

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)
])

# ============================================================
# 4. Build Transfer Learning Model
# ============================================================

def build_model(num_classes=10):

    inputs = layers.Input(shape=(32, 32, 3))

    # Apply augmentation
    x = data_augmentation(inputs)

    # Resize to 224x224
    x = layers.Lambda(lambda img: tf.image.resize(img, (224, 224)))(x)

    # Load ResNet50
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_tensor=x
    )

    # Freeze base model
    base_model.trainable = False

    # Classification head
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="ResNet50_CIFAR10")

    return model, base_model

# Build model
model, base_model = build_model()

# ============================================================
# 5. Compile Model (Initial Training)
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ============================================================
# 6. Callbacks
# ============================================================

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    verbose=1
)

# ============================================================
# 7. Train (Feature Extraction Phase)
# ============================================================

history = model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# ============================================================
# 8. Fine-Tuning (Unfreeze Top Layers)
# ============================================================

for layer in base_model.layers[-30:]:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

fine_history = model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# ============================================================
# 9. Evaluate Model
# ============================================================

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)

print("\n==============================")
print("Test Performance")
print("==============================")
print(f"Test Accuracy : {test_acc * 100:.2f}%")

# ============================================================
# 10. Predictions
# ============================================================

y_pred_probs = model.predict(x_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# ============================================================
# 11. Metrics
# ============================================================

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall    = recall_score(y_test, y_pred, average='weighted')
f1        = f1_score(y_test, y_pred, average='weighted')

print("\n==============================")
print("Detailed Metrics")
print("==============================")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

# ============================================================
# 12. Classification Report
# ============================================================

class_names = [
    'airplane','automobile','bird','cat','deer',
    'dog','frog','horse','ship','truck'
]

print("\nClassification Report")
print(classification_report(y_test, y_pred, target_names=class_names))

# ============================================================
# 13. Confusion Matrix (Visualization)
# ============================================================

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# ============================================================
# 14. Training History Plot
# ============================================================

def plot_history(history, fine_history):

    plt.figure(figsize=(12,5))

    # Accuracy
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.plot(fine_history.history['accuracy'], label='Fine Train Acc')
    plt.plot(fine_history.history['val_accuracy'], label='Fine Val Acc')
    plt.legend()
    plt.title("Accuracy")

    # Loss
    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.plot(fine_history.history['loss'], label='Fine Train Loss')
    plt.plot(fine_history.history['val_loss'], label='Fine Val Loss')
    plt.legend()
    plt.title("Loss")

    plt.show()

plot_history(history, fine_history)

# ============================================================
# 15. Save Model
# ============================================================

model.save("resnet50_cifar10_final.h5")

