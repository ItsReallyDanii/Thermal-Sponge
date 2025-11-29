"""
model.py
----------------------------------------
Defines the XylemAutoencoder architecture used for
synthetic tree vascular structure encoding and reconstruction.

This model learns a compact latent representation ("genetic code")
for microvascular patterns, which can be decoded into
2D images and analyzed through the physics simulator.
"""

import torch
from torch import nn
import torch.nn.functional as F


class XylemAutoencoder(nn.Module):
    def __init__(self, latent_dim=32):
        super(XylemAutoencoder, self).__init__()
        self.latent_dim = latent_dim

        # ==========================
        # Encoder — compress geometry
        # ==========================
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # 256 → 128
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 128 → 64
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 64 → 32
            nn.ReLU(inplace=True),
        )

        # Flatten to latent vector
        self.fc_enc = nn.Linear(128 * 32 * 32, latent_dim)

        # ==========================
        # Decoder — reconstruct geometry
        # ==========================
        self.fc_dec = nn.Linear(latent_dim, 128 * 16 * 16)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 16 → 32
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # 32 → 64
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),   # 64 → 128
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),    # 128 → 256
            nn.Sigmoid()
        )

    def encode(self, x):
        """Encodes input structure → latent vector."""
        x = self.encoder_conv(x)
        x = x.view(x.size(0), -1)
        z = self.fc_enc(x)
        return z

    def decode(self, z):
        """Decodes latent vector → reconstructed structure."""
        x = self.fc_dec(z)
        x = x.view(-1, 128, 16, 16)
        x = self.decoder_conv(x)
        return x

    def forward(self, x):
        """Full forward pass: encode → decode."""
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z


# ===============================================================
# Optional quick test to verify correct output shape
# ===============================================================
if __name__ == "__main__":
    model = XylemAutoencoder(latent_dim=32)
    dummy = torch.randn(1, 1, 256, 256)
    recon, z = model(dummy)
    print(f"✅ Model forward test successful.")
    print(f"Input shape: {dummy.shape}")
    print(f"Latent shape: {z.shape}")
    print(f"Reconstruction shape: {recon.shape}")