import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from modeling.model import KeypointModel
from plots import plot_loss, plot_keypoints
from src.dataset import CelebADataset  # Import your dataset from dataset.py


# Dataset loading
def get_data_loader(image_dir, landmarks_file, batch_size=32):
    from torchvision import transforms

    transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ]
    )
    dataset = CelebADataset(
        image_dir=image_dir, landmarks_file=landmarks_file, transform=transform
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def train_model(
    model, train_loader, criterion, optimizer, num_epochs=10, device="cuda"
):
    model.train()
    epoch_losses = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_idx, (images, keypoints) in enumerate(train_loader):
            images, keypoints = images.to(device), keypoints.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Compute loss
            loss = criterion(outputs, keypoints)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Track and display epoch loss
        avg_loss = running_loss / len(train_loader)
        epoch_losses.append(avg_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

        # Save loss plot
        plot_loss(epoch_losses, filename=f"loss_epoch_{epoch + 1}.png")

        # Visualize predictions periodically
        if (epoch + 1) % 2 == 0:
            visualize_predictions(model, train_loader, device, epoch + 1)

    return epoch_losses


def visualize_predictions(model, loader, device, epoch):
    model.eval()
    images, keypoints = next(iter(loader))
    images, keypoints = images.to(device), keypoints.to(device)

    with torch.no_grad():
        outputs = model(images)

    # Convert to numpy for visualization
    images = images.cpu().numpy()
    outputs = outputs.cpu().numpy()
    keypoints = keypoints.cpu().numpy()

    # Plot first image
    plot_keypoints(
        images[0].transpose((1, 2, 0)),  # CHW -> HWC
        outputs[0],  # Predicted keypoints
        keypoints[0],  # Ground truth keypoints
        filename=f"keypoints_epoch_{epoch}.png",
    )


def main():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset paths
    image_dir = "data/external/Img"  # Adjust to your actual path
    landmarks_file = "data/external/list_landmarks_align_celeba.txt"

    # Initialize data loader
    train_loader = get_data_loader(image_dir, landmarks_file)

    # Model, loss function, and optimizer
    model = KeypointModel(num_landmarks=136).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    num_epochs = 10
    train_model(model, train_loader, criterion, optimizer, num_epochs, device)

    # Save the model
    torch.save(model.state_dict(), "keypoint_model.pth")
    print("Model training complete and saved!")


if __name__ == "__main__":
    main()
