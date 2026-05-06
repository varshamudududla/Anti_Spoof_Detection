import os
import cv2
import numpy as np
import pandas as pd

# 🔥 CHANGE THIS PATH
dataset_path = "CelebA_Spoof-mini"

classes = ["live", "spoof"]

def process_folder(base_path, split_name):
    data = []
    pixel_values = []
    total_images = 0

    print(f"\n📊 Processing {split_name.upper()} dataset...\n")

    for class_name in classes:
        folder = os.path.join(base_path, split_name, class_name)

        if not os.path.exists(folder):
            print(f"❌ Missing: {folder}")
            continue

        images = os.listdir(folder)
        count = len(images)

        print(f"➡ {class_name}: {count} images")
        total_images += count

        for img_name in images:
            img_path = os.path.join(folder, img_name)

            try:
                img = cv2.imread(img_path)
                if img is None:
                    continue

                h, w, _ = img.shape
                pixel_values.append(np.mean(img))

                data.append({
                    "split": split_name,
                    "class": class_name,
                    "height": h,
                    "width": w
                })

            except:
                continue

    df = pd.DataFrame(data)

    print("\n📊 SUMMARY")
    print("Total Images:", total_images)

    print("\nClass Distribution:")
    print(df["class"].value_counts())

    print("\nAverage Size:")
    print("Height:", round(df["height"].mean(), 2))
    print("Width :", round(df["width"].mean(), 2))

    print("\nPixel Mean:", round(np.mean(pixel_values), 2))

    return df


# =========================
# PROCESS TRAIN + VAL
# =========================
train_df = process_folder(dataset_path, "train")
val_df   = process_folder(dataset_path, "val")

# Combine
combined_df = pd.concat([train_df, val_df])

# =========================
# SAVE FILES
# =========================
train_df.to_csv("train_statistics.csv", index=False)
val_df.to_csv("val_statistics.csv", index=False)
combined_df.to_csv("full_dataset_statistics.csv", index=False)

print("\n✅ Files saved:")
print("train_statistics.csv")
print("val_statistics.csv")
print("full_dataset_statistics.csv")