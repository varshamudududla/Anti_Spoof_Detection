import os
import matplotlib.pyplot as plt

# 🔥 CHANGE THIS PATH
dataset_path = "CelebA_Spoof-mini"

classes = ["live", "spoof"]

def count_images(split):
    counts = []
    for cls in classes:
        folder = os.path.join(dataset_path, split, cls)
        if os.path.exists(folder):
            counts.append(len(os.listdir(folder)))
        else:
            counts.append(0)
    return counts


# =========================
# COUNT DATA
# =========================
train_counts = count_images("train")
val_counts   = count_images("val")

# =========================
# PLOT GRAPH (GROUPED BAR)
# =========================
x = range(len(classes))  # [0, 1]

plt.figure()

# Bars
plt.bar([i - 0.2 for i in x], train_counts, width=0.4, label="Train")
plt.bar([i + 0.2 for i in x], val_counts, width=0.4, label="Validation")

# Labels
plt.xticks(x, ["Live", "Spoof"])
plt.xlabel("Classes")
plt.ylabel("Number of Images")
plt.title("Dataset Class Distribution (Train vs Validation)")

# Legend
plt.legend()

# Save
plt.tight_layout()
plt.savefig("dataset_class_distribution.png")

# Show
plt.show()