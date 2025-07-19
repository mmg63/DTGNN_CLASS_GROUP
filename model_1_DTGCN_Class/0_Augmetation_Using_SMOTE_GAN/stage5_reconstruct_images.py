# import torch
# import torch.nn as nn
# import numpy as np
# import torchvision.utils as vutils
# import os

# # === Stage 3: Reconstruct Synthetic Images from Features using Pretrained Decoder ===

# # Configuration
# latent_dim = 512

# X_aug_file = "/Users/mustafamohammadi/Documents/GitHub/3DVNN_202400801/BIOMEDIC_PROJECT/dataset/dataverse_files-2/Combined/underrepresented_Images/BlueRedWhite/X_augmented.npy"
# pretrained_path = "/Users/mustafamohammadi/Documents/GitHub/3DVNN_202400801/BIOMEDIC_PROJECT/dataset/dataverse_files-2/Combined/Combined_images/TMA_patches/autoencoder_checkpoints/decoder.pth"
# output_dir = "/Users/mustafamohammadi/Documents/GitHub/3DVNN_202400801/BIOMEDIC_PROJECT/dataset/dataverse_files-2/Combined/underrepresented_Images/BlueRedWhite/reconstructed_images"
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)

# output_path = os.path.join(output_dir, "synthetic_batch.png")

# # Define decoder model (pretrained decoder from autoencoder or GAN)
# # class Decoder(nn.Module):
# #     def __init__(self, latent_dim=512):
# #         super(Decoder, self).__init__()
# #         self.fc = nn.Sequential(
# #             nn.Linear(latent_dim, 8 * 8 * 128),
# #             nn.ReLU()
# #         )
# #         self.deconv = nn.Sequential(
# #             nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 8x8 → 16x16
# #             nn.ReLU(),
# #             nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),   # 16x16 → 32x32
# #             nn.ReLU(),
# #             nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),    # 32x32 → 64x64
# #             nn.Tanh()
# #         )

# #     def forward(self, z):
# #         x = self.fc(z)
# #         x = x.view(-1, 128, 8, 8)
# #         return self.deconv(x)
# class Decoder(nn.Module):
#     def __init__(self, latent_dim=512):
#         super().__init__()
#         self.fc = nn.Sequential(
#             nn.Linear(latent_dim, 8 * 8 * 128),
#             nn.ReLU()
#         )
#         self.deconv = nn.Sequential(
#             nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 8 → 16
#             nn.ReLU(),
#             nn.ConvTranspose2d(64, 32, 4, 2, 1),   # 16 → 32
#             nn.ReLU(),
#             nn.ConvTranspose2d(32, 16, 4, 2, 1),   # 32 → 64
#             nn.ReLU(),
#             nn.ConvTranspose2d(16, 3, 4, 4, 0),    # 64 → 256
#             nn.Tanh()
#         )

#     def forward(self, z):
#         x = self.fc(z)
#         x = x.view(-1, 128, 8, 8)
#         return self.deconv(x)
    
# # Load model and weights
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# decoder = Decoder(latent_dim=latent_dim).to(device)



# # Load your trained decoder weights (make sure to provide the correct path!)
# if os.path.exists(pretrained_path):
#     decoder.load_state_dict(torch.load(pretrained_path, map_location=device))
#     decoder.eval()
#     print("✅ Pretrained decoder loaded.")
# else:
#     print("❌ Error: Could not find pretrained decoder weights at 'decoder.pth'")
#     exit()

# # Load augmented feature vectors (from Stage 2)
# X_aug = np.load(X_aug_file)

# # Use last 100 synthetic features as an example
# X_tensor = torch.tensor(X_aug[-100:], dtype=torch.float32).to(device)

# # Generate images
# with torch.no_grad():
#     reconstructed_images = decoder(X_tensor)

# # Save image grid

# vutils.save_image(reconstructed_images, output_path, normalize=True)

# print(f"✅ Stage 3 complete. Reconstructed {len(reconstructed_images)} images saved to: {output_path}")





# from PIL import Image
# import math
# import os

# def reconstruct_core_from_patches(patch_folder, output_path, core_size=(3100, 3100), patch_size=256):
#     """
#     Reconstruct a full circular TMA core by stitching 256x256 patches together.
    
#     Args:
#         patch_folder: path to folder with 256x256 patch images (sorted by crop order)
#         output_path: full path to save the reconstructed image
#         core_size: final stitched image size, default 3100x3100
#         patch_size: size of each square patch (default: 256)
#     """
#     core_w, core_h = core_size
#     n_cols = core_w // patch_size
#     n_rows = core_h // patch_size

#     # Create a blank canvas
#     stitched = Image.new("RGB", core_size, (255, 255, 255))

#     # Load patch filenames (sort by name to ensure order)
#     patch_files = sorted([
#         f for f in os.listdir(patch_folder)
#         if f.lower().endswith(('.png', '.jpg', '.jpeg'))
#     ])

#     # Check if we have enough patches
#     expected_patches = n_cols * n_rows
#     if len(patch_files) < expected_patches:
#         print(f"⚠️ Only {len(patch_files)} patches found. Expected: {expected_patches}. Missing parts will be blank.")

#     # Place each patch
#     for idx, patch_file in enumerate(patch_files):
#         row = idx // n_cols
#         col = idx % n_cols

#         if row >= n_rows:
#             break  # ignore extra patches

#         patch_path = os.path.join(patch_folder, patch_file)
#         patch = Image.open(patch_path)

#         x = col * patch_size
#         y = row * patch_size
#         stitched.paste(patch, (x, y))

#     # Save output
#     stitched.save(output_path)
#     print(f"✅ Reconstructed core saved at: {output_path}")


# # Example usage
# patch_folder = "/Users/mustafamohammadi/Documents/GitHub/3DVNN_202400801/BIOMEDIC_PROJECT/dataset/dataverse_files-2/Combined/underrepresented_Images/BlueRedWhite/patch_folder"
# output_path = "/Users/mustafamohammadi/Documents/GitHub/3DVNN_202400801/BIOMEDIC_PROJECT/dataset/dataverse_files-2/Combined/underrepresented_Images/BlueRedWhite/reconstructed_images/"

# reconstruct_core_from_patches(
#     patch_folder=patch_folder,
#     output_path=f"{output_path}stitched_core_ZT111.png",
#     core_size=(3100, 3100),
#     patch_size=256
# )


import os
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from torchvision.utils import save_image
import torchvision.transforms.functional as TF
from PIL import Image


# === Configuration ===
latent_dim = 512
X_aug_path = "/Users/mustafamohammadi/Documents/GitHub/3DVNN_202400801/BIOMEDIC_PROJECT/dataset/dataverse_files-2/Combined/underrepresented_Images/BlueRedWhite/X_augmented.npy"
decoder_weights_path = "/Users/mustafamohammadi/Documents/GitHub/3DVNN_202400801/BIOMEDIC_PROJECT/dataset/dataverse_files-2/Combined/Combined_images/TMA_patches/autoencoder_checkpoints/decoder.pth"
output_dir = "/Users/mustafamohammadi/Documents/GitHub/3DVNN_202400801/BIOMEDIC_PROJECT/dataset/dataverse_files-2/Combined/underrepresented_Images/BlueRedWhite/reconstructed_images"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# === Decoder Architecture ===
class Decoder(nn.Module):
    def __init__(self, latent_dim=512):
        super(Decoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 8 * 8 * 128),
            nn.ReLU()
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 8x8 → 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),   # 16x16 → 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),   # 32x32 → 64x64
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 4, stride=4, padding=0),    # 64x64 → 256x256
            nn.Tanh()
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 128, 8, 8)
        return self.deconv(x)

# === Load Model ===
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
decoder = Decoder(latent_dim=latent_dim).to(device)

if os.path.exists(decoder_weights_path):
    decoder.load_state_dict(torch.load(decoder_weights_path, map_location=device))
    decoder.eval()
    print("✅ Decoder loaded.")
else:
    raise FileNotFoundError(f"❌ Decoder weights not found at: {decoder_weights_path}")

# === Load Synthetic Feature Vectors ===
X_aug = np.load(X_aug_path)
print(f"📦 Loaded {X_aug.shape[0]} synthetic feature vectors.")

# === Inference and Saving ===
batch_size = 32
X_tensor = torch.tensor(X_aug, dtype=torch.float32)
n_total = X_tensor.shape[0]

for start in range(0, n_total, batch_size):
    end = min(start + batch_size, n_total)
    batch = X_tensor[start:end].to(device)

    with torch.no_grad():
        outputs = decoder(batch)

    for i, img in enumerate(outputs):
        patch_idx = start + i
        out_path = os.path.join(output_dir, f"synthetic_patch_{patch_idx:04d}.png")
        TF.to_pil_image(img.cpu()).save(out_path)

print(f"✅ All {n_total} synthetic images saved to: {output_dir}")
