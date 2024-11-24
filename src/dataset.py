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
        self, img_dir, landmarks_file, partition_file, split="train", transform=None
    ):
        """
        Args:
            img_dir (str): Path to the images folder.
            landmarks_file (str): Path to the landmarks file (list_landmarks_align_celeba.txt).
            partition_file (str): Path to the partition file (list_eval_partition.txt).
            split (str): One of 'train', 'val', or 'test'.
            transform (callable, optional): A function/transform to apply to the images.
        """
        self.img_dir = img_dir
        self.split = split
        self.transform = transform

        # Load landmarks and partitions
        self.landmarks = pd.read_csv(
            landmarks_file, sep=r"\s+", skiprows=2, index_col=0
        )
        self.partitions = pd.read_csv(
            partition_file, sep=" ", header=None, index_col=0, names=["partition"]
        )

        # Filter images based on the split
        partition_map = {"train": 0, "val": 1, "test": 2}
        self.filtered_images = self.partitions[
            self.partitions["partition"] == partition_map[split]
        ].index.tolist()

    def __len__(self):
        print(f"Dataset length: {len(self.filtered_images)}")
        return len(self.filtered_images)


def __getitem__(self, idx):
    """
    Retrieve an image and its corresponding keypoints based on the index.

    Args:
        idx (int): Index of the data point.

    Returns:
        Tuple[Tensor, Tensor]: Transformed image and normalized keypoints.
    """
    # Get the image filename
    img_name = self.filtered_images[idx]

    # Remove the extension for correct matching with the landmarks DataFrame
    img_name_base = img_name.split(".")[0]  # Remove '.jpg' extension if needed

    # Ensure we're using the correct directory and extension
    img_path = os.path.join(self.img_dir, img_name_base + ".png")  # Use .png extension

    print(f"Loading image: {img_path}")  # Debugging line

    try:
        # Load the image
        image = Image.open(img_path).convert("RGB")
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Image file not found: {img_path}") from e

    # Get the keypoints by using the base name of the image (without extension)
    try:
        keypoints = self.landmarks.loc[img_name_base].values.astype("float32")
        keypoints = keypoints.reshape(-1, 2)  # Reshape into (num_landmarks, 2)
    except KeyError as e:
        print(f"KeyError for image: {img_name_base}")  # Debugging line
        raise KeyError(f"Keypoints not found for image: {img_name_base}") from e

    # Apply transformations
    if self.transform:
        image = self.transform(image)

    # Normalize keypoints for [-1, 1] range if working with 128x128 images
    keypoints = keypoints / 128.0

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
    img_dir = r"data/raw/celeba/img_align_celeba_png"  # Corrected path
    keypoint_file = "data/external/list_landmarks_align_celeba.txt"
    partition_file = "data/external/list_eval_partition.txt"

    # Define transforms
    transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ]
    )

    # Get DataLoaders for train, validation, and test
    loaders = get_data_loaders(
        img_dir, keypoint_file, partition_file, batch_size=32, transform=transform
    )

    # Example: Print sizes of datasets
    for split, loader in loaders.items():
        print(f"{split.capitalize()} dataset: {len(loader.dataset)} samples")

    # Debug: Print a batch of data
    train_loader = loaders["train"]
    for batch_idx, (images, keypoints) in enumerate(train_loader):
        print(
            f"Batch {batch_idx}: Image shape {images.shape}, Keypoints shape {keypoints.shape}"
        )
        if batch_idx == 0:
            break  # Only show the first batch
