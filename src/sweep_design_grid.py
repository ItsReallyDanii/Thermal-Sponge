import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from src.model import XylemAutoencoder
from src.train_thermal_surrogate import ThermalSurrogate
from src.optimize_latent_thermal import inverse_design_thermal

# CONFIG
MODEL_PATH = "results/model_physics_tuned.pth"
SURROGATE_PATH = "results/thermal_surrogate.pth"
OUTPUT_DIR = "results/thermal_design/"
GRID_SIZE = 5  # 5x5 grid

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Starting Design Sweep ({GRID_SIZE}x{GRID_SIZE})...")
    
    # 1. Load Models
    # Geometry
    ckpt_ae = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    ae = XylemAutoencoder().to(device)
    if isinstance(ckpt_ae, dict) and 'state_dict' in ckpt_ae:
        ae.load_state_dict(ckpt_ae['state_dict'])
    else:
        ae.load_state_dict(ckpt_ae)
    ae.eval()
    
    # Physics
    surrogate = ThermalSurrogate().to(device)
    ckpt_surr = torch.load(SURROGATE_PATH, map_location=device, weights_only=False)
    surrogate.load_state_dict(ckpt_surr['state_dict'])
    
    # Metadata for normalization
    meta = {k: v for k, v in ckpt_surr.items() if k != 'state_dict'}
    surrogate.eval()
    
    # 2. Define Targets
    # Flux (Q) range: 0.06 to 0.14
    # Density (Rho) range: 0.20 to 0.60
    target_qs = np.linspace(0.06, 0.14, GRID_SIZE)
    target_rhos = np.linspace(0.20, 0.60, GRID_SIZE)
    
    # 3. The Grid Loop
    # We will stitch images into a large canvas
    canvas = np.zeros((256 * GRID_SIZE, 256 * GRID_SIZE))
    
    print("   Generating grid...")
    
    for i, q in enumerate(target_qs):      # Rows (Flux)
        for j, r in enumerate(target_rhos): # Cols (Density)
            print(f"   [{i},{j}] Target: Q={q:.3f}, Rho={r:.3f}...", end="\r")
            
            # Run Inverse Design (Fast mode: fewer steps)
            z_opt, img_opt, _ = inverse_design_thermal(
                ae, surrogate, meta, 
                target_q=q, target_rho=r, 
                steps=100
            )
            
            # Convert to numpy
            tile = img_opt.cpu().detach().squeeze().numpy()
            
            # Insert into canvas
            # Note: We flip i to make Flux increase UPWARDS in the plot if desired, 
            # or just standard matrix order. Let's do standard matrix:
            # Row i, Col j
            r_start, r_end = i*256, (i+1)*256
            c_start, c_end = j*256, (j+1)*256
            canvas[r_start:r_end, c_start:c_end] = tile
            
    print("\n✅ Grid generation complete.")
    
    # 4. Save
    plt.figure(figsize=(10, 10))
    plt.imshow(canvas, cmap='inferno')
    plt.title(f"Generative Design Manifold ({GRID_SIZE}x{GRID_SIZE})\nX-axis: Density (0.2->0.6) | Y-axis: Cooling (0.06->0.14)")
    plt.axis('off')
    
    save_path = os.path.join(OUTPUT_DIR, "design_manifold.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"🖼️  Manifold saved to: {save_path}")

if __name__ == "__main__":
    main()