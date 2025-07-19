""" 
    This script it traning the autoencoder(256 X 256) on the pathces exgtarcted from the original images.
    The aim is training this autoencoder on the patches and then using the decoder part of the autoencoder to reconstruct the images.
 """
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm


# === Configuration ===
patch_dir = "/Users/mustafamohammadi/Documents/GitHub/3DVNN_202400801/BIOMEDIC_PROJECT/dataset/dataverse_files-2/Combined/Combined_images/TMA_patches/"       # Folder of 256x256 patches (generated earlier)
save_dir = "/Users/mustafamohammadi/Documents/GitHub/3DVNN_202400801/BIOMEDIC_PROJECT/dataset/dataverse_files-2/Combined/Combined_images/TMA_patches/autoencoder_checkpoints"
os.makedirs(save_dir, exist_ok=True)

batch_size = 64
latent_dim = 512
epochs = 20
lr = 1e-3
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

# === Data Loader ===
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(
    root=os.path.join(patch_dir, ".."),  # expects class subfolder, so hack with parent
    transform=transform
)
# Fake labels so all go in a single folder
dataset.samples = [(os.path.join(patch_dir, f), 0) for f in os.listdir(patch_dir) if f.endswith(('.jpg', '.png'))]

loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

# === Autoencoder ===
class Encoder(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),  # 256 -> 128
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),  # 128 -> 64
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),  # 64 -> 32
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 32 * 32, latent_dim)
        )

    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 8 * 8 * 128),
            nn.ReLU()
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 8 → 16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),   # 16 → 32
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),   # 32 → 64
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 4, 4, 0),    # 64 → 256
            nn.Tanh()
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 128, 8, 8)
        return self.deconv(x)

class Autoencoder(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

if __name__ == "__main__":

    # === Training ===
    model = Autoencoder(latent_dim=latent_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

        for imgs, _ in pbar:
            imgs = imgs.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, imgs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({"loss": running_loss / (pbar.n + 1)})

        print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss / len(loader):.4f}")
        save_image(outputs[:16], os.path.join(save_dir, f"reconstructed_epoch{epoch+1}.png"))

    # Save decoder separately for Stage 3
    torch.save(model.decoder.state_dict(), os.path.join(save_dir, "decoder.pth"))
    print(f"✅ Training complete. Decoder saved to: {os.path.join(save_dir, 'decoder.pth')}")
