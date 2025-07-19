# Author: Mustafa Mohammadi
# GitHub UiD: @mmg63
# Website: http://mohamm6m.myweb.cs.uwindsor.ca/

# Date: 2025-01-09
# Description: This script performs image augmentation using multiprocessing.
# It applies various transformations to images in a specified folder and saves the augmented images in another folder.
# License: MIT License


# === Imports ===
import os
from PIL import Image
import albumentations as A
import numpy as np
from multiprocessing import Pool, cpu_count

# === Configurations ===
input_image_folder = '20250109-BioImages_VoronoiDiagramSimplified_FN/dataset/dataverse_files-2/Combined/Combined_images/'               
output_image_folder = '20250109-BioImages_VoronoiDiagramSimplified_FN/dataset/dataverse_files-2/Combined/Augmented_images_for_whole_dataset/'    
augmentations_per_image = 5                    

os.makedirs(output_image_folder, exist_ok=True)

# === Define the augmentation pipeline ===
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.6),
    A.GaussNoise(p=0.3),
    A.ElasticTransform(p=0.2),
    A.HueSaturationValue(p=0.3),
])

# === Image Augmentation Worker ===
def augment_image(filename):
    try:
        base_name = os.path.splitext(filename)[0]
        image_path = os.path.join(input_image_folder, filename)

        if not os.path.exists(image_path):
            print(f"⚠️ Warning: {image_path} not found.")
            return

        image = np.array(Image.open(image_path).convert('RGB'))

        for i in range(augmentations_per_image):
            augmented = transform(image=image)['image']
            aug_image = Image.fromarray(augmented)
            aug_filename = f"{base_name}_aug{i+1}.png"
            aug_image.save(os.path.join(output_image_folder, aug_filename))

        # print(f"✅ Processed {filename}")
    
    except Exception as e:
        print(f"❌ Error processing {filename}: {e}")

# === Main Function ===
if __name__ == '__main__':
    # Get all .jpg and .png files from the input directory
    valid_extensions = ('.jpg', '.jpeg', '.png')
    filenames = [f for f in os.listdir(input_image_folder) if f.lower().endswith(valid_extensions)]

    print(f"🔍 Found {len(filenames)} images to process."
    )
    # Use as many processes as there are CPU cores (or less if fewer images)
    num_processes = min(cpu_count(), len(filenames))
    with Pool(num_processes) as pool:
        pool.map(augment_image, filenames)

    print("🎉 All augmentations completed using multiprocessing!")
