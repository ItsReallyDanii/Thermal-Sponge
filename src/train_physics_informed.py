"""
train_physics_informed.py  (v0.4 – physics-dominant, porosity-weighted)

Fine-tune the XylemAutoencoder using a learned physics surrogate.

Pipeline:
  - Load autoencoder (results/model_hybrid.pth)
  - Load physics surrogate CNN (results/physics_surrogate.pth)
  - Load generated microtubes as training images
  - Load REAL xylem solver stats as physics targets from
    results/flow_metrics/flow_metrics.csv  (Type == "Real")
  - For each epoch:
      recon = AE(imgs)
      pred_metrics = surrogate(recon)
      physics_loss = weighted sum of per-metric squared errors, with
                     porosity error upweighted
      total_loss = RECON_WEIGHT * recon_loss + λ_phys * physics_loss
  - Save tuned AE to results/model_physics_tuned.pth
  - Save full training log to results/physics_training_log.csv
"""

import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim

from src.model import XylemAutoencoder
from src.train_surrogate import PhysicsSurrogateCNN  # single source of truth

DEVICE = torch.device("cpu")
TARGET_SIZE = (256, 256)

# --------------------
# Loss weights
# --------------------
# How much we care about pixel-wise reconstruction vs physics.
# Physics-dominant: recon is down-weighted, physics is up-weighted.
RECON_WEIGHT = 0.1          # was 1.0 before

# Global ramp for physics influence over epochs
LAMBDA_PHYS_START = 3.0
LAMBDA_PHYS_END   = 20.0

NUM_EPOCHS = 100

# Per-metric physics weights
# (Names correspond to columns: Mean_K, Mean_dP/dy, FlowRate, Porosity, Anisotropy)
K_WEIGHT        = 1.0   # permeability
DPDY_WEIGHT     = 1.0   # pressure gradient
FLOW_WEIGHT     = 1.0   # flowrate
POROSITY_WEIGHT = 10.0  # aggressively push porosity toward target
ANISO_WEIGHT    = 1.0   # anisotropy


# ----------------------------------------------------
# Data loading
# ----------------------------------------------------
def load_and_preprocess_images(path: str) -> torch.Tensor:
    """Load and resize all grayscale images from a folder to consistent size."""
    imgs = []
    for f in sorted(os.listdir(path)):
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif")):
            img = Image.open(os.path.join(path, f)).convert("L")
            img = img.resize(TARGET_SIZE, Image.BILINEAR)
            arr = np.array(img, dtype=np.float32) / 255.0
            imgs.append(torch.from_numpy(arr).unsqueeze(0))  # [1, H, W]
    if not imgs:
        raise RuntimeError(f"No images found in {path}")
    return torch.stack(imgs)  # [N, 1, H, W]


# ----------------------------------------------------
# Load real-physics targets from flow_metrics.csv
# ----------------------------------------------------
def load_real_targets(flow_metrics_path: str = "results/flow_metrics/flow_metrics.csv"):
    """
    Reads solver stats and returns mean targets for:
    [Mean_K, Mean_dP/dy, FlowRate, Porosity, Anisotropy] for REAL samples.
    """
    if not os.path.exists(flow_metrics_path):
        raise FileNotFoundError(
            f"Flow metrics file not found at {flow_metrics_path}. "
            "Run flow_simulation.py first."
        )

    df = pd.read_csv(flow_metrics_path)
    if "Type" in df.columns:
        df_real = df[df["Type"].str.lower() == "real"]
    else:
        df_real = df

    cols = ["Mean_K", "Mean_dP/dy", "FlowRate", "Porosity", "Anisotropy"]
    missing = [c for c in cols if c not in df_real.columns]
    if missing:
        raise ValueError(f"Missing expected columns in flow_metrics.csv: {missing}")

    targets_np = df_real[cols].mean(axis=0).to_numpy(dtype=np.float32)
    print("🎯 Real-physics targets from solver:")
    for name, val in zip(cols, targets_np):
        print(f"   {name:12s} ≈ {val:.6f}")

    targets = torch.from_numpy(targets_np).to(DEVICE)
    return targets, cols


# ----------------------------------------------------
# Main training loop
# ----------------------------------------------------
def main():
    print("🌱 Surrogate-based physics fine-tuning started on cpu")

    # 1) Load autoencoder
    ae = XylemAutoencoder().to(DEVICE)
    ae.load_state_dict(torch.load("results/model_hybrid.pth", map_location=DEVICE))
    ae.train()

    # 2) Load fixed surrogate (same arch as in train_surrogate.py)
    surrogate = PhysicsSurrogateCNN().to(DEVICE)
    surrogate.load_state_dict(torch.load("results/physics_surrogate.pth", map_location=DEVICE))
    surrogate.eval()
    for p in surrogate.parameters():
        p.requires_grad = False  # freeze weights; we only backprop through AE

    # 3) Load physics targets from REAL xylem solver metrics
    target_vec, metric_names = load_real_targets("results/flow_metrics/flow_metrics.csv")

    # Map metric names to indices for clarity
    metric_index = {name: i for i, name in enumerate(metric_names)}

    # 4) Load training images (synthetic microtubes)
    data_path = "data/generated_microtubes"
    imgs = load_and_preprocess_images(data_path).to(DEVICE)
    print(f"🧩 Loaded {imgs.shape[0]} generated structures → resized to {TARGET_SIZE}")

    recon_loss_fn = nn.MSELoss()
    optimizer = optim.Adam(ae.parameters(), lr=1e-4)

    logs = []

    for epoch in range(1, NUM_EPOCHS + 1):
        optimizer.zero_grad()

        # Forward through AE
        recon, _ = ae(imgs)
        recon_loss = recon_loss_fn(recon, imgs)

        # Surrogate predictions on reconstructions: [B, 5]
        pred_metrics = surrogate(recon)
        pred_mean = pred_metrics.mean(dim=0)  # [5]

        # Physics error vector
        diff = pred_mean - target_vec  # [5]

        # Extract per-metric squared errors by name
        loss_k    = diff[metric_index["Mean_K"]]      ** 2
        loss_dpdy = diff[metric_index["Mean_dP/dy"]]  ** 2
        loss_flow = diff[metric_index["FlowRate"]]    ** 2
        loss_por  = diff[metric_index["Porosity"]]    ** 2
        loss_aniso= diff[metric_index["Anisotropy"]]  ** 2

        # Weighted physics loss (porosity emphasized)
        phys_loss = (
            K_WEIGHT        * loss_k
            + DPDY_WEIGHT   * loss_dpdy
            + FLOW_WEIGHT   * loss_flow
            + POROSITY_WEIGHT * loss_por
            + ANISO_WEIGHT  * loss_aniso
        )

        # Gradually ramp global physics weight
        t = epoch / NUM_EPOCHS
        lambda_phys = LAMBDA_PHYS_START + (LAMBDA_PHYS_END - LAMBDA_PHYS_START) * t

        # Total loss: recon is down-weighted, physics dominates
        total_loss = RECON_WEIGHT * recon_loss + lambda_phys * phys_loss
        total_loss.backward()
        optimizer.step()

        # Gradient norm for monitoring
        grad_norm = 0.0
        for p in ae.parameters():
            if p.grad is not None:
                grad_norm += p.grad.norm().item()
        grad_norm = float(grad_norm)

        # Log row
        log_row = {
            "epoch": epoch,
            "total": float(total_loss.item()),
            # log the *unscaled* recon + weighted physics for interpretability
            "recon": float(recon_loss.item()),
            "phys": float(phys_loss.item()),
            "lambda_phys": float(lambda_phys),
            "recon_weight": float(RECON_WEIGHT),
            "loss_K": float(loss_k.item()),
            "loss_dPdy": float(loss_dpdy.item()),
            "loss_Flow": float(loss_flow.item()),
            "loss_Porosity": float(loss_por.item()),
            "loss_Anisotropy": float(loss_aniso.item()),
            "GradNorm": grad_norm,
        }
        # Also stash the current mean metrics (detached to CPU)
        for name, val in zip(metric_names, pred_mean.detach().cpu().numpy()):
            log_row[f"pred_{name}"] = float(val)

        logs.append(log_row)

        if epoch == 1 or epoch % 5 == 0:
            metrics_str = " | ".join(
                f"{name}: {log_row[f'pred_{name}']:.5f}" for name in metric_names
            )
            print(
                f"Epoch {epoch:3d}/{NUM_EPOCHS} | "
                f"Total: {log_row['total']:.5f} | "
                f"Recon(unscaled): {log_row['recon']:.5f} | "
                f"Phys(weighted): {log_row['phys']:.5f} | "
                f"λ_phys: {log_row['lambda_phys']:.2f} | "
                f"{metrics_str} | "
                f"GradNorm: {grad_norm:.2e}"
            )

    # 5) Save tuned model + log
    os.makedirs("results", exist_ok=True)
    torch.save(ae.state_dict(), "results/model_physics_tuned.pth")
    pd.DataFrame(logs).to_csv("results/physics_training_log.csv", index=False)

    print("✅ Surrogate-based physics fine-tuning complete.")
    print("💾 Model saved → results/model_physics_tuned.pth")
    print("🧾 Training log saved → results/physics_training_log.csv")


if __name__ == "__main__":
    main()
