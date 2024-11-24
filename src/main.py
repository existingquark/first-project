import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from modeling.model import KeypointModel
from plots import plot_loss, plot_keypoints
from src.dataset import CelebADataset  # Import your custom dataset from dataset.py
from torchvision import transforms
import numpy as np
from PIL import Image


# Define a function to initialize the dataset and data loader
def get_data_loader(image_dir, landmarks_file, batch_size=32, transform=None):
    dataset = CelebADataset(image_dir, landmarks_file, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader


# Define a function to initialize the model
def initialize_model(device):
    model = KeypointModel(
        num_landmarks=136  # 68 landmarks with (x, y), so 136 keypoints
    )
    model = model.to(device)
    return model


# Define a function to train the model
def train_model(
    model, train_loader, criterion, optimizer, num_epochs=10, device="cuda"
):
    model.train()
    losses = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_idx, (images, keypoints) in enumerate(train_loader):
            images, keypoints = images.to(device), keypoints.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Compute loss
            loss = criterion(outputs, keypoints)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        losses.append(avg_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

        # Plot loss every epoch
        plot_loss(losses, filename=f"loss_epoch_{epoch + 1}.png")

    return losses


# Define the function to save the model
def save_model(model, path="keypoint_model.pth"):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


# Define the function to visualize keypoints (optional)
def visualize_predictions(model, device, image_path, ground_truth_keypoints, transform):
    image = preprocess_image(
        image_path, transform
    )  # preprocess_image function defined below
    image = image.to(device)

    with torch.no_grad():
        predicted_keypoints = model(image)
        predicted_keypoints = predicted_keypoints.cpu().numpy().flatten()

    # Use the plot_keypoints function to visualize the predicted vs. ground truth keypoints
    image_pil = Image.open(image_path).convert("RGB")
    plot_keypoints(
        np.array(image_pil),
        predicted_keypoints,
        ground_truth_keypoints,
        filename="predicted_keypoints.png",
    )


# Preprocess image function (resize and apply transforms)
def preprocess_image(image_path, transform):
    image = Image.open(image_path).convert("RGB")
    return transform(image)


# Main function to run the training
def main():
    # Set the device (use GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset paths
    image_dir = "data/external/Img"  # Path to your images
    landmarks_file = (
        "data/external/list_landmarks_align_celeba.txt"  # Path to your landmarks file
    )

    # Initialize data loader
    transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ]
    )
    train_loader = get_data_loader(image_dir, landmarks_file, transform=transform)

    # Initialize model, loss function, and optimizer
    model = initialize_model(device)
    criterion = torch.nn.MSELoss()  # You can experiment with different loss functions
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    num_epochs = 10
    losses = train_model(model, train_loader, criterion, optimizer, num_epochs, device)

    # Save the trained model
    save_model(model)

    # Optionally visualize predictions (after training is done)
    test_image_path = "path_to_test_image.jpg"  # Replace with actual image path
    ground_truth_keypoints = np.array(
        [10, 20, 30, 40]
    )  # Replace with actual ground truth keypoints for testing
    visualize_predictions(
        model, device, test_image_path, ground_truth_keypoints, transform
    )  # Pass transform here

    # Optionally plot the final training loss
    plot_loss(losses, filename="final_loss_plot.png")


if __name__ == "__main__":
    main()
