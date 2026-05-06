import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# ==============================
# LOAD MODEL
# ==============================
model = tf.keras.models.load_model("liveness_model_final_v3.keras", compile=False)

# Automatically detect input size
input_shape = model.input_shape
IMG_SIZE = input_shape[1]

print("✅ Model Loaded")
print("📐 Expected Input Size:", IMG_SIZE)

# ==============================
# DATASET PATH
# ==============================
dataset_path = "CelebA_Spoof-mini/val"
classes = ["live", "spoof"]

# ==============================
# LOAD DATA
# ==============================
X = []
y_true = []

print("\n📁 Working Directory:", os.getcwd())
print("📁 Dataset Path:", dataset_path)

for label, cls in enumerate(classes):
    folder = os.path.join(dataset_path, cls)

    print(f"\n🔍 Checking folder: {folder}")

    if not os.path.exists(folder):
        print(f"❌ Folder NOT found: {folder}")
        continue

    files = os.listdir(folder)
    print(f"✅ {cls}: {len(files)} images")

    for img_name in files:
        img_path = os.path.join(folder, img_name)

        img = cv2.imread(img_path)
        if img is None:
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # ✅ auto size
        img = img / 255.0

        X.append(img)
        y_true.append(label)

# Convert to numpy
X = np.array(X)
y_true = np.array(y_true)

print("\n📊 Total samples loaded:", len(X))

# ==============================
# SAFETY CHECK
# ==============================
if len(X) == 0:
    print("❌ ERROR: No images loaded. Fix dataset path.")
    exit()

# ==============================
# PREDICTIONS
# ==============================
print("\n🚀 Running predictions...")
y_pred_prob = model.predict(X, batch_size=32, verbose=1).flatten()

threshold = 0.5
y_pred = (y_pred_prob > threshold).astype(int)

# ==============================
# CONFUSION MATRIX (IMPROVED)
# ==============================
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6, 5))
plt.imshow(cm, cmap='Blues')
plt.colorbar()

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j],
                 ha='center', va='center',
                 fontsize=12, fontweight='bold')

plt.xticks([0, 1], classes)
plt.yticks([0, 1], classes)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")

plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.show()

# ==============================
# ACCURACY GRAPH (CLEAN)
# ==============================
accuracy = accuracy_score(y_true, y_pred)

plt.figure(figsize=(5, 5))
plt.bar(["Accuracy"], [accuracy])
plt.ylim(0, 1)

plt.text(0, accuracy - 0.05, f"{accuracy:.2f}", ha='center', fontsize=12)

plt.title("Model Accuracy")
plt.ylabel("Accuracy")

plt.savefig("accuracy.png", dpi=300)
plt.show()

print(f"\n✅ Accuracy: {accuracy * 100:.2f}%")

# ==============================
# PROBABILITY DISTRIBUTION (IMPROVED)
# ==============================
plt.figure(figsize=(7, 5))

live_index = classes.index("live")
spoof_index = classes.index("spoof")

plt.hist(y_pred_prob[y_true == live_index],
         bins=30, alpha=0.6, label="Live", density=True)

plt.hist(y_pred_prob[y_true == spoof_index],
         bins=30, alpha=0.6, label="Spoof", density=True)

plt.axvline(x=0.5, linestyle='--', label="Threshold")

plt.xlabel("Prediction Probability")
plt.ylabel("Density")
plt.title("Prediction Probability Distribution")

plt.legend()
plt.tight_layout()

plt.savefig("probability_distribution.png", dpi=300)
plt.show()

# ==============================
# CLASSIFICATION REPORT
# ==============================
print("\n📄 Classification Report:\n")
print(classification_report(y_true, y_pred, target_names=classes))

# ==============================
# OPTIONAL: SAVE RESULTS
# ==============================
np.save("y_true.npy", y_true)
np.save("y_pred.npy", y_pred)
np.save("y_pred_prob.npy", y_pred_prob)

print("\n💾 Saved predictions for future analysis")