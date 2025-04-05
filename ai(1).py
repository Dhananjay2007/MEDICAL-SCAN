import os
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img, img_to_array


def load_dataset(csv_file, base_dir, img_size=(224, 224)):
    # Load the CSV
    data = pd.read_csv(csv_file)

    # Create lists to store images and labels
    images = []
    labels = []

    valid_extensions = (".png", ".jpg", ".jpeg")

    for _, row in data.iterrows():
        dir_path = os.path.join(base_dir, row.iloc[0])  # Combine base directory with folder path
        label = row.iloc[1]  # Extract the label

        # Debugging: Print directory path
        print(f"Checking directory: {dir_path}")

        # Check if directory exists
        if os.path.isdir(dir_path):
            print(f"Directory exists: {dir_path}")
            for file_name in os.listdir(dir_path):
                if file_name.lower().endswith(valid_extensions):
                    img_path = os.path.join(dir_path, file_name)
                    try:
                        img = load_img(img_path, target_size=img_size, color_mode="grayscale")
                        img = img_to_array(img) / 255.0  # Normalize to [0, 1]
                        images.append(img)
                        labels.append(label)
                    except Exception as e:
                        print(f"Error loading image {img_path}: {e}")
        else:
            print(f"Directory NOT found: {dir_path}")

    return images, labels


# Update the base directory
base_dir = "E:/MEDSCAN AI/DATA SET/"  # Adjust based on dataset structure

# Define dataset paths
train_csv = "E:/MEDSCAN AI/DATA SET/MURA-v1.1/train_labeled_studies.csv"
valid_csv = "E:/MEDSCAN AI/DATA SET/MURA-v1.1/valid_labeled_studies.csv"

# Load training and validation datasets
train_images, train_labels = load_dataset(train_csv, base_dir)
valid_images, valid_labels = load_dataset(valid_csv, base_dir)

# Debugging: Print dataset stats
print(f"Number of training images: {len(train_images)}")
print(f"Number of training labels: {len(train_labels)}")
print(f"Number of validation images: {len(valid_images)}")
print(f"Number of validation labels: {len(valid_labels)}")
