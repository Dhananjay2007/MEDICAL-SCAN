# Import required libraries
import os
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

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


# Grad-CAM class
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def generate_cam(self, input_image, body_part, target_class):
        # Forward pass with both input_image and body_part
        model_output = self.model(input_image, body_part)

        # Backward pass
        self.model.zero_grad()
        model_output[:, target_class].backward()

        # Get gradients and activations
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]

        # Compute the weights
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        # Create the CAM
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]

        # Post-process CAM
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (input_image.shape[2], input_image.shape[3]))
        cam -= np.min(cam)
        cam /= np.max(cam)
        return cam




# Load the trained model
model_path = "E:/MEDSCAN AI/AI's/unified_model.pth"  # Update the path if needed
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Initialize and load the model
model = UnifiedModel()
model.load_state_dict(torch.load(model_path, map_location=device))  # Load the saved weights
model = model.to(device)
model.eval()

# Grad-CAM setup
target_layer = model.backbone.layer4[2]  # Use the last ResNet layer for Grad-CAM
grad_cam = GradCAM(model, target_layer)

# Define the transforms for the input image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Test the model with a sample image
image_path = "C:/Users/dhana/Downloads/WhatsApp Image 2025-01-26 at 10.00.58 PM.jpeg"  # Update the path
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

# Generate Grad-CAM
cam = grad_cam.generate_cam(input_image, body_part_onehot, predicted.item())
# Convert image to NumPy for visualization
np_image = np.array(image)
cam_resized = cv2.resize(cam, (np_image.shape[1], np_image.shape[0]))
heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

# Superimpose the heatmap on the original image
superimposed = cv2.addWeighted(np_image, 0.6, heatmap, 0.4, 0)

# Draw bounding box around high-activation areas
threshold = 0.6  # Adjust threshold for bounding box
cam_binary = (cam_resized > threshold).astype(np.uint8)
contours, _ = cv2.findContours(cam_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(superimposed, (x, y), (x + w, y + h), (255, 0, 0), 2)

# Visualization
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(np_image)
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Grad-CAM Heatmap")
plt.imshow(heatmap)
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Grad-CAM with Bounding Box")
plt.imshow(superimposed)
plt.axis("off")

plt.show()
