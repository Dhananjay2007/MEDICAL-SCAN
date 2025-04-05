import os


def verify_dataset_structure(dataset_path):
    # Dictionary to store class counts
    class_counts = {}

    # Traverse the dataset path
    for class_name in os.listdir(dataset_path):
        class_folder = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_folder):
            # Count the number of images in each class folder
            num_images = len([f for f in os.listdir(class_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])
            class_counts[class_name] = num_images

    return class_counts


# Define the dataset paths
train_path = "E:/MEDSCAN AI/DATA SET/Bone_Fracture_Binary_Classification/Bone_Fracture_Binary_Classification/train"
val_path = "E:/MEDSCAN AI/DATA SET/Bone_Fracture_Binary_Classification/Bone_Fracture_Binary_Classification/val"
test_path = "E:/MEDSCAN AI/DATA SET/Bone_Fracture_Binary_Classification/Bone_Fracture_Binary_Classification/test"

# Verify each split
print("Train Set:")
print(verify_dataset_structure(train_path))

print("\nValidation Set:")
print(verify_dataset_structure(val_path))

print("\nTest Set:")
print(verify_dataset_structure(test_path))
