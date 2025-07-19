"""
    The aim of this script is to extract patches from large images, compute their features using a pre-trained ResNet model, and save the features and simulated multi-labels for further processing. The script uses PyTorch and torchvision libraries for image processing and feature extraction.
    The script is divided into two main stages:
    1. **Patch Extraction**: It extracts 256x256 patches from images in a specified folder.
    2. **Feature Extraction**: It uses a pre-trained ResNet model to compute features for each patch and simulates multi-labels for the patches.
    The script also includes configuration settings for the image folder, patch size, and device (CPU or GPU) to be used for processing. The extracted features and labels are saved as NumPy arrays for later use.
    This script is part of a larger project that aims to augment a dataset using synthetic samples generated from the extracted features.
"""

import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models

# === Stage 1: Patch Extraction & Feature Extraction ===

# Configuration
# path to the folder containing the unrepresented images
original_image_folder = '/Users/mustafamohammadi/Documents/GitHub/3DVNN_202400801/BIOMEDIC_PROJECT/dataset/dataverse_files-2/Combined/underrepresented_Images/BlueRedWhite/'
patch_size = 256
resnet_feature_dim = 512
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ResNet feature extractor
resnet = models.resnet18(pretrained=True)
resnet = nn.Sequential(*list(resnet.children())[:-1])  # Remove the classifier
resnet.to(device)
resnet.eval()

# Image transformation
transform = transforms.Compose([
    transforms.Resize((patch_size, patch_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Function to crop image into patches
def crop_image_to_patches(image_path, patch_size=256):
    image = Image.open(image_path).convert("RGB")
    W, H = image.size
    patches = []

    for i in range(0, W, patch_size):
        for j in range(0, H, patch_size):
            box = (i, j, min(i + patch_size, W), min(j + patch_size, H))
            patch = image.crop(box)
            if patch.size == (patch_size, patch_size):
                patches.append(patch)
    return patches

# Extract features and simulate multi-labels
all_features = []
all_labels = []

for filename in os.listdir(original_image_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(original_image_folder, filename)
        patches = crop_image_to_patches(image_path, patch_size=patch_size)

        for idx, patch in enumerate(patches):
            input_tensor = transform(patch).unsqueeze(0).to(device)
            with torch.no_grad():
                feature = resnet(input_tensor).squeeze().cpu().numpy()
            all_features.append(feature)

            # Simulate 5-class multi-label for demo [white, green, blue, yellow, red]
            label = np.random.randint(0, 2, size=(5,))
            all_labels.append(label)

# Save features and labels
np.save("X_features.npy", np.array(all_features))
np.save("Y_labels.npy", np.array(all_labels))

print(f"✅ Stage 1 complete: {len(all_features)} patch features extracted and saved.")
