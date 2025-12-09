import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. THE BRAIN
# ==========================================
class BeamNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# ==========================================
# 2. DATA GENERATION
# ==========================================
def get_normalized_beam():
    slice_centers = [-0.5, 0.5, 1.5]
    x_list, y_list, z_list = [], [], []
    for x_c in slice_centers:
        y_vals = np.random.uniform(0, 250, 2000)
        z_tilt = y_vals * 2
        z_vals = np.random.uniform(0, 2500, 2000) + z_tilt
        x_vals = np.random.normal(x_c, 0.05, 2000)
        x_list.append(x_vals); y_list.append(y_vals); z_list.append(z_vals)

    x, y, z = np.concatenate(x_list), np.concatenate(y_list), np.concatenate(z_list)
    x = (x - x.min()) / (x.max() - x.min())
    y = (y - y.min()) / (y.max() - y.min())
    z = (z - z.min()) / (z.max() - z.min())
    return np.vstack((x, y, z)).T.astype(np.float32)

# ==========================================
# 3. PHYSICS LOSS (Aggressive Connectivity)
# ==========================================
def calculate_physics_loss(density, coords, target_mass=0.50):
    # A. Mass Target (Increased slightly to allow for the web)
    current_mass = torch.mean(density)
    mass_loss = (current_mass - target_mass) ** 2
    
    # B. Stiffness
    z_coords = coords[:, 2].unsqueeze(1)
    total_mass = torch.sum(density) + 1e-6
    z_centroid = torch.sum(density * z_coords) / total_mass
    distance_sq = (z_coords - z_centroid) ** 2
    I = torch.sum(density * distance_sq)
    stiffness_loss = 1.0 / (I + 1e-6)
    
    # C. SHEAR / CONNECTIVITY (THE FIX)
    # Target Zone: The absolute center strip (Height 0.4 to 0.6)
    z = coords[:, 2]
    mid_mask = (z > 0.4) & (z < 0.6)
    
    if mid_mask.sum() > 0:
        mid_density = density[mid_mask]
        # FORCE: Mid-section must be > 0.8 density (Solid)
        # We penalize any point in the middle that is less than 0.8
        shear_penalty = torch.mean(torch.relu(0.8 - mid_density) ** 2)
    else:
        shear_penalty = 0.0
    
    return stiffness_loss, mass_loss, shear_penalty

# ==========================================
# 4. TRAINING ROUTINE
# ==========================================
def train_and_export():
    points_np = get_normalized_beam()
    coords = torch.tensor(points_np)
    model = BeamNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    
    print("Starting Optimization (Forcing SOLID Web)...")
    
    for epoch in range(800): 
        optimizer.zero_grad()
        predicted_density = model(coords)
        
        stiff, mass, shear = calculate_physics_loss(predicted_density, coords)
        
        # WEIGHT ADJUSTMENT:
        # Stiffness: 20.0
        # Shear (Web): 100.0 (The AI cannot ignore this now)
        # Mass: 5.0 (Lowest priority, mass will rise to accommodate the web)
        total_loss = (20.0 * stiff) + (5.0 * mass) + (100.0 * shear)
        
        total_loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss {total_loss.item():.4f} | Mass: {torch.mean(predicted_density).item():.2f} | Web Penalty: {shear.item():.4f}")

    return model

# ==========================================
# 5. EXPORT (Corrected Threshold)
# ==========================================
def export_to_obj(model, filename="final_beam_design_fixed.obj", resolution=60):
    print(f"Voxelizing and exporting to {filename}...")
    
    x = np.linspace(0, 1, resolution)
    y = np.linspace(0, 1, resolution)
    z = np.linspace(0, 1, resolution)
    xx, yy, zz = np.meshgrid(x, y, z)
    
    grid_points = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T
    grid_tensor = torch.tensor(grid_points, dtype=torch.float32)
    
    with torch.no_grad():
        density = model(grid_tensor).numpy().reshape(resolution, resolution, resolution)
    
    # FILTER FIX:
    # Lowered from 0.5 to 0.4 to catch the web structure
    solid_indices = np.argwhere(density > 0.4)
    
    with open(filename, 'w') as f:
        f.write("# Inverse Design Beam Export (Fixed Web)\n")
        for idx in solid_indices:
            vx, vy, vz = idx / resolution
            f.write(f"v {vx:.4f} {vy:.4f} {vz:.4f}\n")
            
    print("Export Complete.")

if __name__ == "__main__":
    trained_model = train_and_export()
    export_to_obj(trained_model)