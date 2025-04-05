# Import required libraries
import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Device configuration (CPU/GPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Dataset class for all body parts
class UnifiedFractureDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None):
        self.data = dataframe.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.data.iloc[idx, 0])
        image = Image.open(img_path).convert('RGB')
        label = torch.tensor(self.data.iloc[idx, 3], dtype=torch.long)  # Ensure label is a tensor
        body_part = self.data.iloc[idx, 2]  # Body part name

        # One-hot encode the body part
        body_part_onehot = torch.zeros(7)  # Assuming 7 body parts
        body_part_idx = ['Elbow', 'Finger', 'Forearm', 'Hand', 'Humerus', 'Shoulder', 'Wrist'].index(body_part)
        body_part_onehot[body_part_idx] = 1

        if self.transform:
            image = self.transform(image)

        return image, body_part_onehot, label


# Training function
def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=25):
    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                data_loader = train_loader
            else:
                model.eval()
                data_loader = val_loader

            running_loss = 0.0
            running_corrects = 0

            for inputs, body_parts, labels in data_loader:
                inputs = inputs.to(device)
                body_parts = body_parts.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs, body_parts)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(data_loader.dataset)
            epoch_acc = running_corrects.double() / len(data_loader.dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

    model.load_state_dict(best_model_wts)
    return model


# Unified model class
class UnifiedModel(nn.Module):
    def __init__(self):
        super(UnifiedModel, self).__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # Remove the original classifier layer

        # Additional layer for body part embedding
        self.body_part_fc = nn.Linear(7, 128)  # Assuming 7 body parts
        self.classifier = nn.Linear(num_ftrs + 128, 2)  # Binary classification

    def forward(self, x, body_part):
        x = self.backbone(x)
        x = torch.flatten(x, start_dim=1)  # Flatten the backbone output
        body_part_embed = self.body_part_fc(body_part)
        combined = torch.cat((x, body_part_embed), dim=1)
        out = self.classifier(combined)
        return out


# Main block to avoid multiprocessing issues on Windows
if __name__ == '__main__':
    # Define root directory and CSV file path
    root_dir = "E:/MEDSCAN AI/DATA SET/"
    csv_file_path = os.path.join(root_dir, "MURA-v1.1/valid_image_paths.csv")

    # Load the dataset paths
    valid_data_paths = pd.read_csv(csv_file_path)  # Load the CSV file
    valid_data_paths.columns = ['img_path']  # Assign a column name for the paths

    # Verify that image paths exist
    valid_data_paths['full_path'] = valid_data_paths['img_path'].apply(lambda x: os.path.join(root_dir, x))
    valid_data_paths = valid_data_paths[valid_data_paths['full_path'].apply(os.path.exists)]

    # Extract and replace body part names
    valid_data_paths['body_part'] = valid_data_paths['img_path'].apply(lambda x: x.split('/')[2]).replace({
        'XR_HAND': 'Hand',
        'XR_ELBOW': 'Elbow',
        'XR_SHOULDER': 'Shoulder',
        'XR_FINGER': 'Finger',
        'XR_WRIST': 'Wrist',
        'XR_FOREARM': 'Forearm',
        'XR_HUMERUS': 'Humerus'
    })

    # Extract labels
    valid_data_paths['label'] = valid_data_paths['img_path'].map(
        lambda x: 'positive' if 'positive' in x else 'negative'
    ).replace({'positive': 1, 'negative': 0}).astype(int)

    # Debug unique body parts and labels
    print("Unique body parts in dataset:", valid_data_paths['body_part'].unique())
    print("Unique labels in dataset:", valid_data_paths['label'].unique())

    # Data augmentation and normalization
    data_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Dataset and DataLoader
    dataset = UnifiedFractureDataset(dataframe=valid_data_paths, root_dir=root_dir, transform=data_transforms)

    # Ensure dataset is not empty
    if len(dataset) == 0:
        raise ValueError("The dataset is empty. Ensure valid image paths and labels.")

    # Split into train and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    if train_size == 0 or val_size == 0:
        raise ValueError("Insufficient data to create train or validation splits.")

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Debug split sizes
    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Initialize model, loss function, and optimizer
    model = UnifiedModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    print("Starting model training...")

    # Train the model
    trained_model = train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=25)

    # Specify the folder to save the model
    save_dir = "E:/MEDSCAN AI/AI's"
    os.makedirs(save_dir, exist_ok=True)

    # Save the trained model
    model_save_path = os.path.join(save_dir, 'unified_model.pth')
    torch.save(trained_model.state_dict(), model_save_path)
    print(f"Unified model saved at: {model_save_path}")
