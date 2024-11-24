import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import Dataset
from modeling.model import KeypointModel
import os
from PIL import Image
import pandas as pd
import numpy as np
from plots import plot_loss, plot_keypoints  # Importing your plot functions


# Custom Dataset Class to load CelebA images and keypoints
class CelebADataset(Dataset):
    def __init__(self, image_dir, landmarks_file, transform=None):
        self.image_dir = image_dir
        self.landmarks_file = landmarks_file
        self.transform = transform

        # Load the keypoint annotations (e.g., CSV or TXT file)
        self.landmarks = pd.read_csv(landmarks_file, delimiter=" ")
        self.image_files = os.listdir(image_dir)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Get the image path
        img_name = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_name)

        # Get the corresponding keypoints
        keypoints = self.landmarks.iloc[idx].values
        keypoints = keypoints.astype("float32")

        # Normalize the keypoints if needed (e.g., [0, 1] range for image size normalization)
        keypoints = (
            keypoints / [image.width, image.height] * 2 - 1
        )  # Example normalization

        if self.transform:
            image = self.transform(image)

        return image, keypoints


# Dataset loading
transform = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ]
)

# Replace with the actual paths to your CelebA images and annotations
train_dataset = CelebADataset(
    image_dir="path_to_celeba_images",
    landmarks_file="path_to_landmark_file.csv",  # CSV file containing keypoint annotations
    transform=transform,
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Initialize the model
model = KeypointModel(num_landmarks=136)  # 68 landmarks, each having (x, y), hence 136
model = model.to("cuda" if torch.cuda.is_available() else "cpu")

# Loss function and optimizer
criterion = torch.nn.MSELoss()  # Mean squared error loss for keypoint regression
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Initialize a list to track loss over epochs for plotting
epoch_losses = []

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch_idx, (images, keypoints) in enumerate(train_loader):
        images = images.to("cuda" if torch.cuda.is_available() else "cpu")
        keypoints = keypoints.to("cuda" if torch.cuda.is_available() else "cpu")

        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Compute loss (output is expected to be (batch_size, 136) for 68 keypoints)
        loss = criterion(outputs, keypoints)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Track the loss for plotting
    epoch_losses.append(running_loss / len(train_loader))

    # Print loss for the current epoch
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

    # Plot the training loss after each epoch
    plot_loss(epoch_losses, filename="loss_plot.png")

    # Optionally, plot predictions on a sample of images every few epochs (e.g., every 2nd epoch)
    if (epoch + 1) % 2 == 0:
        # Get a batch of images and their corresponding keypoints for visualization
        images, keypoints = next(iter(train_loader))
        images = images.to("cuda" if torch.cuda.is_available() else "cpu")
        keypoints = keypoints.to("cuda" if torch.cuda.is_available() else "cpu")

        with torch.no_grad():
            outputs = model(images)

        # Convert the images back to numpy for plotting (CHW -> HWC)
        images = images.cpu().numpy()
        outputs = outputs.cpu().numpy()
        keypoints = keypoints.cpu().numpy()

        # Plot the keypoints (using the first image in the batch)
        plot_keypoints(
            images[0].transpose((1, 2, 0)),  # Convert to HWC format for displaying
            outputs[0],  # Predicted keypoints
            keypoints[0],  # Ground truth keypoints
            filename=f"keypoints_epoch_{epoch+1}.png",
        )

# Save the model after training
torch.save(model.state_dict(), "keypoint_model.pth")
