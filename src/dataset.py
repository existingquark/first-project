import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import numpy as np
import torchvision.transforms as transforms


class CelebADataset(Dataset):
    """
    PyTorch Dataset for the CelebA dataset.
    Reads images, keypoints, and supports train/val/test splits.
    """

    def __init__(
        self, img_dir, keypoint_file, partition_file, split="train", transform=None
    ):
        """
        Args:
            img_dir (str): Path to the images folder.
            keypoint_file (str): Path to the landmarks file (list_landmarks_align_celeba.txt).
            partition_file (str): Path to the partition file (list_eval_partition.txt).
            split (str): One of 'train', 'val', or 'test'.
            transform (callable, optional): A function/transform to apply to the images.
        """
        self.img_dir = img_dir
        self.split = split
        self.transform = transform

        # Load landmarks and partitions
        self.landmarks = pd.read_csv(keypoint_file, sep="\s+", skiprows=1, index_col=0)
        self.partitions = pd.read_csv(
            partition_file, sep=" ", header=None, index_col=0, names=["partition"]
        )

        # Filter images based on the split
        partition_map = {"train": 0, "val": 1, "test": 2}
        self.filtered_images = self.partitions[
            self.partitions["partition"] == partition_map[split]
        ].index.tolist()

    def __len__(self):
        return len(self.filtered_images)

    def __getitem__(self, idx):
        # Get the image filename and corresponding landmarks
        img_name = self.filtered_images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        # Get the keypoints
        keypoints = self.landmarks.loc[img_name].values.astype("float32")
        keypoints = keypoints.reshape(-1)  # Flatten the landmarks into a 1D array

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        # Normalize keypoints (if required, depending on the model input size)
        keypoints = keypoints / 128.0  # Normalize to range [-1, 1] for 128x128 images
        return image, keypoints


def get_data_loaders(
    img_dir, keypoint_file, partition_file, batch_size=32, transform=None
):
    """
    Create PyTorch DataLoaders for train, val, and test splits.

    Args:
        img_dir (str): Path to the images folder.
        keypoint_file (str): Path to the landmarks file.
        partition_file (str): Path to the partition file.
        batch_size (int): Batch size for the data loaders.
        transform (callable, optional): A function/transform to apply to the images.

    Returns:
        dict: Dictionary containing 'train', 'val', and 'test' DataLoaders.
    """
    splits = ["train", "val", "test"]
    loaders = {}

    for split in splits:
        dataset = CelebADataset(
            img_dir, keypoint_file, partition_file, split, transform
        )
        loaders[split] = DataLoader(
            dataset, batch_size=batch_size, shuffle=(split == "train")
        )

    return loaders


# Example usage
if __name__ == "__main__":
    # Paths to required files
    img_dir = "data/raw/celeba/Img/img_align_celeba"
    keypoint_file = "data/external/list_landmarks_align_celeba.txt"
    partition_file = "data/external/list_eval_partition.txt"

    # Define transforms
    transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ]
    )

    # Get DataLoaders
    loaders = get_data_loaders(
        img_dir, keypoint_file, partition_file, batch_size=32, transform=transform
    )

    # Example: Print sizes of datasets
    for split, loader in loaders.items():
        print(f"{split.capitalize()} dataset: {len(loader.dataset)} samples")
