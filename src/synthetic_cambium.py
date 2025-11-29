"""
synthetic_cambium.py
----------------------------------------
Simulates adaptive vascular growth (synthetic cambium), where a trained
autoencoder "grows" vascular-like structures based on feedback from
a fluid-flow simulation.

This models how real trees reinforce xylem tissue in response
to pressure gradients — closing the loop between structure and flow.
"""

import os, sys, torch, numpy as np, matplotlib.pyplot as plt

# --- Ensure imports work both in Camber and locally ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# --- Project modules ---
from src.model import XylemAutoencoder
from src.simulate_flow import simulate_pressure_field, compute_conductivity

# --- Config ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LATENT_DIM = 32
ITERATIONS = 20
ALPHA = 0.1  # growth learning rate (like cambial sensitivity)
SAVE_DIR = os.path.join(ROOT_DIR, "results", "cambium_growth")
os.makedirs(SAVE_DIR, exist_ok=True)


# ======================================================================
# Helper functions
# ======================================================================

def decode_structure(model, z):
    """
    Decode latent vector z → vascular structure image.
    Automatically detects which decoder attribute the model uses.
    Ensures correct reshaping before ConvTranspose2d layers.
    """
    with torch.no_grad():
        # Fully connected expansion from latent to feature map
        x = model.fc_dec(z)

        # Reshape for convolutional decoder (matches model.py)
        x = x.view(-1, 128, 16, 16)

        # Choose appropriate decoder
        if hasattr(model, "decoder_conv"):
            decoded = model.decoder_conv(x)
        elif hasattr(model, "decoder"):
            decoded = model.decoder(x)
        elif hasattr(model, "deconv"):
            decoded = model.deconv(x)
        else:
            raise AttributeError("❌ Model missing decoder attribute (decoder / decoder_conv / deconv).")

        # Ensure correct dimensionality (B,C,H,W)
        if len(decoded.shape) != 4:
            raise RuntimeError(f"Unexpected decoded shape: {decoded.shape}")

        return decoded


def growth_cycle(model, z):
    """
    Perform one iteration of adaptive growth:
      1. Decode latent z into a vascular structure.
      2. Simulate pressure and flow field.
      3. Compute conductivity and gradient feedback.
      4. Update z based on the feedback ("growth rule").
    """
    recon = decode_structure(model, z)
    img = recon.detach().cpu().numpy()[0, 0]

    # Simulate pressure field and compute effective conductivity
    p_field, mask = simulate_pressure_field(img)
    conductivity = compute_conductivity(p_field, mask)

    # Compute feedback — mean gradient of pressure field
    grad_proxy = torch.tensor(np.mean(np.gradient(p_field)), dtype=torch.float32).to(DEVICE)

    # Update latent space ("growth" step)
    z_new = z + ALPHA * grad_proxy
    z_new = z_new / torch.norm(z_new)  # normalize latent vector (stability)

    return z_new, conductivity, img, p_field


# ======================================================================
# Main simulation loop
# ======================================================================

def main():
    print("🌿 Running Synthetic Cambium Growth Loop...")

    # Initialize model
    model = XylemAutoencoder(latent_dim=LATENT_DIM).to(DEVICE)

    # --- Load pretrained model safely ---
    state_path = os.path.join(ROOT_DIR, "results", "xylem_autoencoder.pt")
    if not os.path.exists(state_path):
        raise FileNotFoundError(f"❌ Model weights not found at {state_path}")

    print("⚙️ Loading pretrained weights (ignoring size mismatches)...")
    state_dict = torch.load(state_path, map_location=DEVICE)
    model_state = model.state_dict()

    # Only keep matching layer shapes
    filtered_state = {
        k: v for k, v in state_dict.items()
        if k in model_state and v.shape == model_state[k].shape
    }

    # Load compatible weights
    missing, unexpected = model.load_state_dict(filtered_state, strict=False)
    print(f"✅ Loaded {len(filtered_state)} compatible layers, "
          f"skipped {len(model_state) - len(filtered_state)} mismatched ones.")

    model.eval()

    # --- Initialize random latent vector (the 'cambium seed') ---
    z = torch.randn(1, LATENT_DIM, device=DEVICE)
    conductivity_history = []

    # --- Growth iterations ---
    for i in range(ITERATIONS):
        z, cond, img, field = growth_cycle(model, z)
        conductivity_history.append(cond)

        # Save visualization
        plt.imshow(img, cmap="gray")
        plt.axis("off")
        plt.title(f"Iteration {i+1} | Conductivity: {cond:.4f}")
        plt.savefig(os.path.join(SAVE_DIR, f"growth_{i:03d}.png"),
                    bbox_inches="tight", pad_inches=0)
        plt.close()

        print(f"Iteration {i+1}/{ITERATIONS} | Conductivity: {cond:.5f}")

    # Save final metrics
    np.savetxt(os.path.join(SAVE_DIR, "conductivity_history.txt"), conductivity_history)
    print(f"✅ Synthetic cambium growth complete. Results saved to: {SAVE_DIR}")


# ======================================================================
# Entrypoint
# ======================================================================

if __name__ == "__main__":
    main()
