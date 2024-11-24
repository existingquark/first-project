import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
import torch.optim as optim
from modeling.model import KeypointModel  # Assuming correct folder name here
from plots import plot_loss, plot_keypoints  # For visualization
from PIL import Image
import numpy as np


# Define a function to initialize the dataset and data loader
def get_data_loader(data_path, batch_size=32):
    transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ]
    )
    dataset = datasets.ImageFolder(data_path, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader


# Define a function to initialize the model
def initialize_model(device):
    model = KeypointModel(
        num_landmarks=68
    )  # Adjust the number of landmarks based on your dataset
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
        for batch_idx, (images, _) in enumerate(train_loader):
            images = images.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Compute loss (using dummy labels for now, replace with actual keypoint labels)
            labels = torch.zeros_like(outputs).to(device)  # Replace with actual labels
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        losses.append(avg_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

        # Plot loss every epoch
        plot_loss(losses, filename=f"loss_epoch_{epoch+1}.png")

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

    # Initialize data loader
    data_path = "path_to_your_dataset"  # Replace with actual dataset path
    transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ]
    )
    train_loader = get_data_loader(data_path)

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