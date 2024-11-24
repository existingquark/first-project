from dataset import CelebADataset
from PIL import Image
import torchvision.transforms as transforms
import os

# Paths to your dataset
img_dir = "data/raw/img_align_celeba_png"
landmarks_file = "data/external/list_landmarks_align_celeba.txt"
partition_file = "data/external/list_eval_partition.txt"

# Create the dataset
dataset = CelebADataset(
    img_dir=img_dir,
    landmarks_file=landmarks_file,
    partition_file=partition_file,
    split="train",  # You can test "val" or "test" too
    transform=None,  # Add transformations if desired
)

# Test 1: Check dataset length
print(f"Dataset length: {len(dataset)}")

# Test 2: Check first few samples
for i in range(5):  # Test first 5 samples
    try:
        image, keypoints = dataset[i]
        print(
            f"Sample {i}: Image size: {image.size}, Keypoints shape: {keypoints.shape}"
        )
    except Exception as e:
        print(f"Error accessing dataset[{i}]: {e}")

# Test 3: Check transform pipeline
img_path = "data/raw/img_align_celeba_png/000001.png"
image = Image.open(img_path).convert("RGB")
transform = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ]
)
try:
    transformed_image = transform(image)
    print(f"Transformed image shape: {transformed_image.shape}")
except Exception as e:
    print(f"Error in transform pipeline: {e}")

# Test 4: Check missing image files
missing_files = []
for img_name in dataset.filtered_images:
    img_path = os.path.join(dataset.img_dir, img_name.replace(".jpg", ".png"))
    if not os.path.exists(img_path):
        missing_files.append(img_name)

if missing_files:
    print(f"Missing files: {len(missing_files)}")
    print("Examples:", missing_files[:5])  # Print first 5 missing files
else:
    print("All image files are present.")

# Test 5: Check missing landmarks
missing_landmarks = []
for img_name in dataset.filtered_images:
    try:
        keypoints = dataset.landmarks.loc[img_name].values
    except KeyError:
        missing_landmarks.append(img_name)

if missing_landmarks:
    print(f"Missing landmarks: {len(missing_landmarks)}")
    print("Examples:", missing_landmarks[:5])  # Print first 5 missing landmarks
else:
    print("All landmarks are present.")
