import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. THE BRAIN (Same MLP)
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
    # Standard geometry
    slice_centers = [-0.5, 0.5, 1.5]
    x_list, y_list, z_list = [], [], []
    for x_c in slice_centers:
        y_vals = np.random.uniform(0, 250, 1500)
        z_tilt = y_vals * 2
        z_vals = np.random.uniform(0, 2500, 1500) + z_tilt
        x_vals = np.random.normal(x_c, 0.05, 1500)
        x_list.append(x_vals); y_list.append(y_vals); z_list.append(z_vals)

    x, y, z = np.concatenate(x_list), np.concatenate(y_list), np.concatenate(z_list)
    x = (x - x.min()) / (x.max() - x.min())
    y = (y - y.min()) / (y.max() - y.min())
    z = (z - z.min()) / (z.max() - z.min())
    return np.vstack((x, y, z)).T.astype(np.float32)

# ==========================================
# 3. ADVANCED PHYSICS LOSS (Bending + Shear)
# ==========================================
def calculate_physics_loss(density, coords, target_mass=0.45):
    # A. Mass Target
    current_mass = torch.mean(density)
    mass_loss = (current_mass - target_mass) ** 2
    
    # B. Stiffness (Moment of Inertia)
    z_coords = coords[:, 2].unsqueeze(1)
    total_mass = torch.sum(density) + 1e-6
    z_centroid = torch.sum(density * z_coords) / total_mass
    distance_sq = (z_coords - z_centroid) ** 2
    I = torch.sum(density * distance_sq)
    stiffness_loss = 1.0 / (I + 1e-6)
    
    # C. SHEAR / CONNECTIVITY CONSTRAINT (New)
    # We look at the middle of the beam (Height ~ 0.5)
    # If the network tries to delete the web (density < 0.2), we penalize it.
    z = coords[:, 2]
    # Select points in the middle 20% of height
    mid_mask = (z > 0.4) & (z < 0.6)
    if mid_mask.sum() > 0:
        mid_density = density[mid_mask]
        # We want mid_density to be at least 0.2
        # Relu logic: penalty only if density < 0.2
        shear_penalty = torch.mean(torch.relu(0.2 - mid_density) ** 2)
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
    
    print("Starting Final Optimization (Stiffness + Web Connectivity)...")
    
    for epoch in range(600): # 600 epochs for high detail
        optimizer.zero_grad()
        predicted_density = model(coords)
        
        stiff, mass, shear = calculate_physics_loss(predicted_density, coords)
        
        # Weighted Loss:
        # Stiffness is king (20.0)
        # Shear ensures it doesn't break (10.0)
        # Mass keeps it light (5.0)
        total_loss = (20.0 * stiff) + (5.0 * mass) + (10.0 * shear)
        
        total_loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss {total_loss.item():.4f} | Mass: {torch.mean(predicted_density).item():.2f} | Web Penalty: {shear.item():.4f}")

    return model

# ==========================================
# 5. VOXELIZER & OBJ EXPORTER (No Dependencies)
# ==========================================
def export_to_obj(model, filename="final_beam_design.obj", resolution=50):
    print(f"Voxelizing and exporting to {filename}...")
    
    # 1. Create a dense 3D Grid
    x = np.linspace(0, 1, resolution)
    y = np.linspace(0, 1, resolution)
    z = np.linspace(0, 1, resolution)
    xx, yy, zz = np.meshgrid(x, y, z)
    
    grid_points = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T
    grid_tensor = torch.tensor(grid_points, dtype=torch.float32)
    
    # 2. Query the Brain
    with torch.no_grad():
        density = model(grid_tensor).numpy().reshape(resolution, resolution, resolution)
    
    # 3. Threshold (The "Surface")
    # Anything > 0.5 is solid material
    solid_indices = np.argwhere(density > 0.5)
    
    # 4. Write OBJ file (Point Cloud / Voxel approach for simplicity)
    # Standard OBJ format: v x y z
    with open(filename, 'w') as f:
        f.write("# Inverse Design Beam Export\n")
        f.write(f"# Resolution: {resolution}\n")
        for idx in solid_indices:
            # Scale indices back to 0-1
            vx, vy, vz = idx / resolution
            f.write(f"v {vx:.4f} {vy:.4f} {vz:.4f}\n")
            
    print("Export Complete. File saved.")

if __name__ == "__main__":
    trained_model = train_and_export()
    
    # Visualize preview
    points_np = get_normalized_beam()
    with torch.no_grad():
        final_d = trained_model(torch.tensor(points_np)).numpy()
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    mask = final_d.flatten() > 0.4
    p = points_np[mask]
    ax.scatter(p[:,0], p[:,1], p[:,2], c=final_d[mask].flatten(), cmap='magma', s=2)
    ax.set_title("Final Design (With Web)")
    ax.set_box_aspect((1,1,1))
    plt.show()
    
    # Export
    export_to_obj(trained_model, resolution=60)