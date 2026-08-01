"""Download and split the vehicle dataset.

Downloads Asseh/Vehicle_Classification from Hugging Face, then splits
it 80/10/10 into data/train, data/val, data/test organized by class.

Usage:
    python scripts/prepare_data.py
"""

import os
import random
import shutil
import zipfile
import urllib.request

DATASET_URL = (
    "https://huggingface.co/datasets/Asseh/Vehicle_Classification/"
    "resolve/main/vehicel_classification.zip"
)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT, "data_raw")
DATA_DIR = os.path.join(ROOT, "data")
SPLITS = [("train", 0.0, 0.8), ("val", 0.8, 0.9), ("test", 0.9, 1.0)]


def main():
    random.seed(42)
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    zip_path = os.path.join(RAW_DIR, "vehicle_classification.zip")
    if not os.path.exists(zip_path):
        print(f"Downloading dataset (~160MB) from Hugging Face...")
        urllib.request.urlretrieve(DATASET_URL, zip_path)
    else:
        print("Zip already downloaded, skipping.")

    extract_dir = os.path.join(RAW_DIR, "extracted")
    if not os.path.isdir(extract_dir):
        print("Extracting...")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(extract_dir)
    else:
        print("Already extracted, skipping.")

    # The zip contains a single folder with class subfolders
    inner = os.path.join(extract_dir, os.listdir(extract_dir)[0])

    for split, _, _ in SPLITS:
        os.makedirs(os.path.join(DATA_DIR, split), exist_ok=True)

    for cls in sorted(os.listdir(inner)):
        class_dir = os.path.join(inner, cls)
        if not os.path.isdir(class_dir):
            continue
        cls_lower = cls.lower()
        images = [f for f in os.listdir(class_dir)
                  if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        random.shuffle(images)
        n = len(images)
        print(f"{cls_lower}: {n} images")

        for split, start, end in SPLITS:
            dest = os.path.join(DATA_DIR, split, cls_lower)
            os.makedirs(dest, exist_ok=True)
            for f in images[int(n * start):int(n * end)]:
                shutil.copy(os.path.join(class_dir, f), os.path.join(dest, f))

    print("\nDone. Split:")
    for split, _, _ in SPLITS:
        total = sum(len(os.listdir(os.path.join(DATA_DIR, split, c)))
                    for c in sorted(os.listdir(os.path.join(DATA_DIR, split))))
        print(f"  {split}: {total} images")


if __name__ == "__main__":
    main()
