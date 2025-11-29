"""
analyze_latent.py
Visualize the learned latent space of the XylemAutoencoder.
Works with dynamic-layer model version (auto-initializing architecture).
"""

import os, sys, subprocess

# --- Auto-install missing dependencies ---
REQUIRED = ["torch", "torchvision", "numpy", "matplotlib",
            "Pillow", "scikit-learn", "umap-learn"]
for pkg in REQUIRED:
    try:
        __import__(pkg.replace("-", "_"))
    except ImportError:
        print(f"Installing missing package: {pkg}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

# --- Imports ---
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap

# --- Path setup ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# --- Import model definition ---
try:
    from src.model import XylemAutoencoder
except ModuleNotFoundError:
    import model
    XylemAutoencoder = model.XylemAutoencoder

# --- Config ---
DATA_DIR = os.path.join(ROOT_DIR, "data", "generated_microtubes")
MODEL_PATH = os.path.join(ROOT_DIR, "results", "xylem_autoencoder.pt")
RESULTS_DIR = os.path.join(ROOT_DIR, "results", "latent_analysis")
os.makedirs(RESULTS_DIR, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Dataset loader ---
class XylemDataset(Dataset):
    def __init__(self, path):
        self.files = sorted([f for f in os.listdir(path) if f.endswith(".png")])
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        img = Image.open(os.path.join(DATA_DIR, self.files[idx]))
        return self.transform(img), self.files[idx]

# --- Load model + initialize layers before loading weights ---
model = XylemAutoencoder(latent_dim=32).to(DEVICE)

# Force layer initialization with dummy input
with torch.no_grad():
    _ = model(torch.zeros(1, 1, 256, 256).to(DEVICE))

# Load trained weights (allow partial mismatch)
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
missing, unexpected = model.load_state_dict(state_dict, strict=False)
print(f"Loaded weights. Missing keys: {missing}, Unexpected keys: {unexpected}")

model.eval()

# --- Load data ---
dataset = XylemDataset(DATA_DIR)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

# --- Encode all images into latent vectors ---
latents, names = [], []
with torch.no_grad():
    for imgs, fns in loader:
        imgs = imgs.to(DEVICE)
        _, z = model(imgs)
        latents.append(z.cpu().numpy())
        names += fns
latents = np.vstack(latents)
print(f"Extracted latent vectors: {latents.shape}")

# --- Compute 2D projections ---
tsne = TSNE(n_components=2, perplexity=5, random_state=42).fit_transform(latents)
umap_proj = umap.UMAP(n_neighbors=5, min_dist=0.3,
                      metric="euclidean", random_state=42).fit_transform(latents)

# --- Plot t-SNE ---
plt.figure(figsize=(6, 5))
plt.scatter(tsne[:, 0], tsne[:, 1],
            c=np.arange(len(latents)), cmap="viridis", s=60)
for i, name in enumerate(names):
    plt.text(tsne[i, 0], tsne[i, 1],
             name.replace("xylem_", "").replace(".png", ""),
             fontsize=8)
plt.title("t-SNE of latent space")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "latent_tsne.png"))
plt.close()

# --- Plot UMAP ---
plt.figure(figsize=(6, 5))
plt.scatter(umap_proj[:, 0], umap_proj[:, 1],
            c=np.arange(len(latents)), cmap="plasma", s=60)
for i, name in enumerate(names):
    plt.text(umap_proj[i, 0], umap_proj[i, 1],
             name.replace("xylem_", "").replace(".png", ""),
             fontsize=8)
plt.title("UMAP of latent space")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "latent_umap.png"))
plt.close()

print(f"✅ Latent analysis complete. Plots saved to {RESULTS_DIR}")
