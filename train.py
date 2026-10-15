
"""
Train a CNN on MNIST (handwritten digits) with Keras.
 
Outputs
  mnist_cnn.keras                 trained model (used by app.py)
  images/training_curves.png      accuracy / loss per epoch
  images/confusion_matrix.png     test-set confusion matrix
  images/sample_predictions.png   12 random test predictions
 
Install:  pip install -r requirements.txt
Run:      python train.py
Expected: ~99% test accuracy.
"""
 
import os
 
import numpy as np
import matplotlib
matplotlib.use("Agg")  # save figures to files without needing a display
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
 
MODEL_PATH = "mnist_cnn.keras"
IMAGES_DIR = "images"
os.makedirs(IMAGES_DIR, exist_ok=True)
 
# ---------------------------------------------------------------
# 1. Load and preprocess data
# ---------------------------------------------------------------
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
 
# Scale pixels to [0, 1] and add a channel dimension -> (N, 28, 28, 1)
x_train = (x_train.astype("float32") / 255.0)[..., np.newaxis]
x_test = (x_test.astype("float32") / 255.0)[..., np.newaxis]
 
print("Train:", x_train.shape, y_train.shape)
print("Test: ", x_test.shape, y_test.shape)
 
# ---------------------------------------------------------------
# 2. Build the model (same architecture as app.py)
# ---------------------------------------------------------------
model = keras.Sequential([
    keras.Input(shape=(28, 28, 1)),
    # Light augmentation (only active during training)
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.1),
    layers.RandomTranslation(0.1, 0.1),
 
    layers.Conv2D(32, 3, activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
 
    layers.Conv2D(64, 3, activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
 
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.4),
    layers.Dense(10, activation="softmax"),
])
 
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",  # labels are integers 0-9
    metrics=["accuracy"],
)
model.summary()
 
# ---------------------------------------------------------------
# 3. Train
# ---------------------------------------------------------------
callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3,
                                  restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
]
 
history = model.fit(
    x_train, y_train,
    validation_split=0.1,
    epochs=15,
    batch_size=128,
    callbacks=callbacks,
)
 
# ---------------------------------------------------------------
# 4. Evaluate
# ---------------------------------------------------------------
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest accuracy: {test_acc:.4f}  |  Test loss: {test_loss:.4f}")
 
# Training curves
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(history.history["accuracy"], label="train")
ax[0].plot(history.history["val_accuracy"], label="val")
ax[0].set_title("Accuracy"); ax[0].set_xlabel("Epoch"); ax[0].legend()
ax[1].plot(history.history["loss"], label="train")
ax[1].plot(history.history["val_loss"], label="val")
ax[1].set_title("Loss"); ax[1].set_xlabel("Epoch"); ax[1].legend()
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "training_curves.png"))
plt.close(fig)
 
# Confusion matrix
y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(6, 6))
ConfusionMatrixDisplay(cm).plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title("Confusion matrix (test set)")
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "confusion_matrix.png"))
plt.close(fig)
 
# ---------------------------------------------------------------
# 5. Visualize some predictions
# ---------------------------------------------------------------
idx = np.random.choice(len(x_test), 12, replace=False)
fig = plt.figure(figsize=(10, 5))
for i, j in enumerate(idx):
    plt.subplot(3, 4, i + 1)
    plt.imshow(x_test[j].squeeze(), cmap="gray")
    color = "green" if y_pred[j] == y_test[j] else "red"
    plt.title(f"Pred: {y_pred[j]}  True: {y_test[j]}", color=color, fontsize=9)
    plt.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "sample_predictions.png"))
plt.close(fig)
 
# ---------------------------------------------------------------
# 6. Save the model
# ---------------------------------------------------------------
model.save(MODEL_PATH)
print(f"Saved model to {MODEL_PATH}")
print(f"Saved charts to {IMAGES_DIR}/")
 






