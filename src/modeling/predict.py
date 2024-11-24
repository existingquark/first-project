import torch
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from model.model import KeypointModel
from plots import plot_keypoints  # To visualize predictions


# Define a function to load the trained model
def load_model(model_path, device):
    model = KeypointModel(num_landmarks=136)  # Adjust if needed for your model
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()  # Set to evaluation mode
    return model


# Define a function for image preprocessing
def preprocess_image(image_path, transform):
    image = Image.open(image_path).convert("RGB")  # Open image
    image = transform(image)  # Apply the same transforms used in training
    image = image.unsqueeze(0)  # Add a batch dimension (1, C, H, W)
    return image


# Define the transformation pipeline (match with training)
transform = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ]
)

# Path to the model and image for prediction
model_path = "keypoint_model.pth"
image_path = "path_to_image.jpg"  # Replace with the image you want to predict

# Device setup (use GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
model = load_model(model_path, device)

# Preprocess the image
image = preprocess_image(image_path, transform)
image = image.to(device)

# Make predictions
with torch.no_grad():  # Disable gradient computation during inference
    predicted_keypoints = model(image)
    predicted_keypoints = predicted_keypoints.cpu().numpy().flatten()

# Visualize the keypoints on the image
image_pil = Image.open(image_path).convert("RGB")  # Load image for visualization
plot_keypoints(
    np.array(image_pil),
    predicted_keypoints,
    predicted_keypoints,
    filename="predicted_keypoints.png",
)

# Optionally, print the keypoints
print("Predicted Keypoints:", predicted_keypoints)
