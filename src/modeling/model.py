import torch
import torch.nn as nn
import torch.nn.functional as F


class KeypointModel(nn.Module):
    def __init__(self, num_landmarks=136):
        super(KeypointModel, self).__init__()

        # Define the convolutional layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

        # Calculate the input size to fc1 (128 * 128 * 128)
        self.fc1 = nn.Linear(128 * 128 * 128, 1024)  # Adjusted input size for fc1
        self.fc2 = nn.Linear(
            1024, num_landmarks
        )  # Output 136 points (68 keypoints x 2 coordinates)

    def forward(self, x):
        # Pass through convolutional layers with ReLU activation
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        print(f"Shape after conv layers: {x.shape}") #check output shape after second conv layer

        # Flatten the output for the fully connected layer
        x = x.view(x.size(0), -1)

        # Pass through fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
