# Import required libraries
import os
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image

# UnifiedModel class
class UnifiedModel(nn.Module):
    def __init__(self):
        super(UnifiedModel, self).__init__()
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        num_ftrs = self.backbone.fc.in_features

        # Replace the fully connected layer
        self.backbone.fc = nn.Identity()  # Remove the default fc layer

        # Additional layer for body part embedding
        self.body_part_fc = nn.Linear(7, 128)  # Assuming 7 body parts
        self.classifier = nn.Linear(num_ftrs + 128, 2)  # Binary classification

    def forward(self, x, body_part):
        x = self.backbone(x)
        body_part_embed = self.body_part_fc(body_part)
        combined = torch.cat((x, body_part_embed), dim=1)
        out = self.classifier(combined)
        return out


# Load the trained model
model_path = "E:/MEDSCAN AI/AI's/unified_model.pth"  # Update the path if needed
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Initialize and load the model
model = UnifiedModel()
model.load_state_dict(torch.load(model_path, map_location=device))  # Load the saved weights
model = model.to(device)
model.eval()

# Define the transforms for the input image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Test the model with a sample image
image_path = "C:/Users/dhana/Downloads/2d59fd50936098df8ddf49c912e70b_big_gallery.jpg"   # Update the path
body_part_name = "Hand"  # Specify the body part

# Preprocess the image
image = Image.open(image_path).convert('RGB')
input_image = transform(image).unsqueeze(0).to(device)

# One-hot encode the body part
body_parts = ["Elbow", "Finger", "Forearm", "Hand", "Humerus", "Shoulder", "Wrist"]
body_part_onehot = torch.zeros(1, 7).to(device)  # Batch size is 1
body_part_idx = body_parts.index(body_part_name)
body_part_onehot[0, body_part_idx] = 1

# Perform inference
with torch.no_grad():
    output = model(input_image, body_part_onehot)
    _, predicted = torch.max(output, 1)

# Map the prediction to class labels
classes = {0: "Normal", 1: "Fractured"}
print(f"Prediction: {classes[predicted.item()]}")
