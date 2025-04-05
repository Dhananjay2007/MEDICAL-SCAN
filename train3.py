# Import required libraries
import os
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

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


# Grad-CAM Implementation
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        target_layer = dict(self.model.named_modules())[self.target_layer]
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

    def generate_cam(self, input_image, body_part, class_idx):
        self.model.eval()
        self.model.zero_grad()

        # Forward pass
        output = self.model(input_image, body_part)
        target = output[:, class_idx]

        # Backward pass
        target.backward(retain_graph=True)

        # Gradients and activations
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]

        # Grad-CAM calculation
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = np.maximum(cam, 0)  # ReLU activation
        cam = cam / cam.max() if cam.max() != 0 else cam

        # Resize Grad-CAM to match the input image size
        cam = cv2.resize(cam, (input_image.shape[2], input_image.shape[3]), interpolation=cv2.INTER_LINEAR)
        return cam


# Load the trained model
model_path = "E:/MEDSCAN AI/AI's/unified_model.pth"  # Update the path if needed
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Initialize and load the model
model = UnifiedModel()
model.load_state_dict(torch.load(model_path, map_location=device))  # Load the saved weights
model = model.to(device)
model.eval()

# Initialize Grad-CAM
grad_cam = GradCAM(model, target_layer="backbone.layer4")

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

# Generate Grad-CAM heatmap
cam = grad_cam.generate_cam(input_image, body_part_onehot, predicted.item())

# Convert Grad-CAM to heatmap
cam_image = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

# Resize heatmap to match the original image
cam_image_resized = cv2.resize(cam_image, (image.width, image.height), interpolation=cv2.INTER_LINEAR)

# Overlay Grad-CAM heatmap on the original image
overlayed_image = cv2.addWeighted(np.array(image), 0.6, cam_image_resized, 0.4, 0)

# Bounding Box Generation
threshold = 0.5  # Adjust this value as needed
binary_mask = cam > threshold
y_indices, x_indices = np.where(binary_mask)

if len(x_indices) > 0 and len(y_indices) > 0:
    x_min, x_max = x_indices.min(), x_indices.max()
    y_min, y_max = y_indices.min(), y_indices.max()
else:
    x_min, x_max, y_min, y_max = 0, 0, 0, 0

# Draw bounding box on the overlayed image
overlayed_image_with_box = cv2.rectangle(
    overlayed_image.copy(),
    (int(x_min * image.width / cam.shape[1]), int(y_min * image.height / cam.shape[0])),
    (int(x_max * image.width / cam.shape[1]), int(y_max * image.height / cam.shape[0])),
    (255, 0, 0), 2
)

# Add label and confidence score
label = f"{classes[predicted.item()]} ({torch.softmax(output, 1)[0, predicted.item()].item():.2f})"
cv2.putText(
    overlayed_image_with_box,
    label,
    (int(x_min * image.width / cam.shape[1]), int(y_min * image.height / cam.shape[0]) - 10),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.5,
    (255, 255, 255),
    1
)

# Display results
plt.figure(figsize=(15, 5))

# Original Image
plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(image)
plt.axis("off")

# Grad-CAM Heatmap
plt.subplot(1, 3, 2)
plt.title("Grad-CAM Heatmap")
plt.imshow(cam, cmap="jet")
plt.axis("off")

# Overlayed Image with Bounding Box
plt.subplot(1, 3, 3)
plt.title("Grad-CAM with Bounding Box")
plt.imshow(cv2.cvtColor(overlayed_image_with_box, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.show()
