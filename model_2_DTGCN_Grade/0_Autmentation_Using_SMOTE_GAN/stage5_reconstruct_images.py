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
X_aug_path = "/Users/*****MYUSERNAME****/Documents/GitHub/3DVNN_202400801/BIOMEDIC_PROJECT/dataset/dataverse_files-2/Combined/underrepresented_Images/BlueRedWhite/X_augmented.npy"
decoder_weights_path = "/Users/*****MYUSERNAME****/Documents/GitHub/3DVNN_202400801/BIOMEDIC_PROJECT/dataset/dataverse_files-2/Combined/Combined_images/TMA_patches/autoencoder_checkpoints/decoder.pth"
output_dir = "/Users/*****MYUSERNAME****/Documents/GitHub/3DVNN_202400801/BIOMEDIC_PROJECT/dataset/dataverse_files-2/Combined/underrepresented_Images/BlueRedWhite/reconstructed_images"
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
