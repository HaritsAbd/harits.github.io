import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# Ukuran gambar target
IMG_SIZE = 100

def extract_image_features(dataset_path):
    """
    Ekstrak fitur dari semua gambar dalam dataset terstruktur.
    Menyimpan hasil ke model/features.csv dan model/targets.csv.
    """
    labels = []
    features = []

    print("🔍 Memulai ekstraksi fitur dari dataset...\n")

    # Loop semua label/kategori (misal: acne, redness, bags)
    for label in os.listdir(dataset_path):
        label_path = os.path.join(dataset_path, label)

        if not os.path.isdir(label_path):
            continue

        # Loop setiap folder individu (0, 1, 2, ...)
        for person_folder in tqdm(os.listdir(label_path), desc=f"📁 Label: {label}"):
            person_path = os.path.join(label_path, person_folder)

            if not os.path.isdir(person_path):
                continue

            person_features = []

            # Gabungkan fitur dari 3 gambar (front, left_side, right_side)
            for img_name in ["front.jpg", "left_side.jpg", "right_side.jpg"]:
                img_path = os.path.join(person_path, img_name)

                if not os.path.exists(img_path):
                    print(f"⚠ Gambar tidak ditemukan: {img_path}")
                    continue

                img = cv2.imread(img_path)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                mean = np.mean(img_gray)
                std = np.std(img_gray)

                person_features.extend([mean, std])

            # Pastikan fitur lengkap (2 fitur × 3 gambar = 6 nilai)
            if len(person_features) == 6:
                features.append(person_features)
                labels.append(label)

    # Simpan ke folder model/
    os.makedirs("model", exist_ok=True)

    features_df = pd.DataFrame(features, columns=[
        "front_mean", "front_std",
        "left_mean", "left_std",
        "right_mean", "right_std"
    ])
    labels_df = pd.DataFrame(labels, columns=["label"])

    features_df.to_csv("model/features.csv", index=False)
    labels_df.to_csv("model/targets.csv", index=False)

    print("\n✅ Fitur dan label berhasil disimpan ke folder model/.")

def extract_image_features_single(img_path):
    """
    Ekstrak fitur sederhana dari satu gambar.
    Output: [mean, std]
    """
    if not os.path.exists(img_path):
        print(f"⚠ Gambar tidak ditemukan: {img_path}")
        return []

    img = cv2.imread(img_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    mean = np.mean(img_gray)
    std = np.std(img_gray)

    return [mean, std]