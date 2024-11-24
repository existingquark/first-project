import matplotlib.pyplot as plt
import numpy as np


# Function to plot training loss over time
def plot_loss(losses, filename="loss_plot.png"):
    if losses:
        plt.figure(figsize=(10, 5))
        plt.plot(np.arange(len(losses)), losses, label="Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss over Epochs")
        plt.legend()
        plt.grid(True)
        plt.savefig(filename)
        plt.close()
    else:
        print("Warning: Losses list is empty. No plot will be generated.")


# Function to plot keypoints (could be useful for monitoring predictions)
def plot_keypoints(
    image, predicted_keypoints, ground_truth_keypoints, filename="keypoints_plot.png"
):
    # Convert image from Tensor (C, H, W) to (H, W, C) for displaying
    image = image.transpose((1, 2, 0))  # Convert from CHW to HWC format
    image = np.clip(image, 0, 1)  # Ensure values are within [0, 1] range for display

    plt.figure(figsize=(10, 10))
    plt.imshow(image)

    # Plot the predicted keypoints
    plt.scatter(
        predicted_keypoints[0::2],
        predicted_keypoints[1::2],
        color="red",
        label="Predicted",
        s=10,
    )

    # Plot the ground truth keypoints
    plt.scatter(
        ground_truth_keypoints[0::2],
        ground_truth_keypoints[1::2],
        color="blue",
        label="Ground Truth",
        s=10,
    )

    plt.title("Keypoint Prediction vs Ground Truth")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.legend()
    plt.savefig(filename)
    plt.close()
