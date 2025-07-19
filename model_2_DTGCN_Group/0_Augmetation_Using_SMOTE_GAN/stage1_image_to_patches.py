"""
    This file is used to just extract patches from the images and save them in a folder.
    The patches are then used to train the autoencoder on stage 2. The aim of training the autoencoder is for reconstruction the images on stage 5.
"""
from PIL import Image
import os
import numpy as np


#  === Configuration ===
original_image_folder = "/Users/mustafamohammadi/Documents/GitHub/3DVNN_202400801/BIOMEDIC_PROJECT/dataset/dataverse_files-2/Combined/underrepresented_Images/BlueRedWhite/"  # Folder containing original images
patch_size = 256  # Size of the patches to be extracted
#  === Create Patch Directory ===
patch_dir = "/Users/mustafamohammadi/Documents/GitHub/3DVNN_202400801/BIOMEDIC_PROJECT/dataset/dataverse_files-2/Combined/underrepresented_Images/BlueRedWhite/patch_folder/"  # Directory to save patches
os.makedirs(patch_dir, exist_ok=True)
#  === Function to Save Patches ===

#  === Image Patching ===
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



for filename in os.listdir(original_image_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(original_image_folder, filename)
        patches = crop_image_to_patches(image_path, patch_size=patch_size)
        # Save patches
        for idx, patch in enumerate(patches):
            patch_filename = os.path.join(patch_dir, f"{os.path.splitext(filename)[0]}_patch_{idx}.png")
            patch.save(patch_filename)
            print(f"Saved {patch_filename}")
print(f"✅ Stage 1 complete: {len(os.listdir(original_image_folder))} images processed and patches saved in {patch_dir}.")
# This code extracts patches from images in the specified folder and saves them in a new directory.
# The patches are of size 256x256 pixels.
# The patches are saved with filenames indicating their original image and patch index.
# This is useful for training models that require smaller input sizes or for data augmentation.