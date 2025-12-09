import pandas as pd
import matplotlib.pyplot as plt
import os
from PIL import Image

# 1. Load the FRESH Verified Metrics (Thermal)
csv_path = "results/thermal_metrics/thermal_metrics.csv"
if not os.path.exists(csv_path):
    print(f"❌ Metrics file not found at {csv_path}")
    exit()

df = pd.read_csv(csv_path)

# 2. Find the Best "Titanium Wood" Candidate
# We look for the trade-off: High Heat Flux (Q_total) AND High Density (rho_solid)
# Score = Normalized_Flux * (Density^2)
df['flux_norm'] = df['Q_total'] / df['Q_total'].max()
df['stiff_proxy'] = df['rho_solid'] ** 2
df['score'] = df['flux_norm'] * (df['stiff_proxy'] / df['stiff_proxy'].max())

# Sort by score
winner = df.sort_values('score', ascending=False).iloc[0]

print(f"🦖 FOUND TRUE CANDIDATE (Audited):")
print(f"   Filename: {winner['filename']}")
print(f"   Density:  {winner['rho_solid']:.4f} (Solid %)")
print(f"   Flux (Q): {winner['Q_total']:.6f}")
print(f"   Score:    {winner['score']:.4f}")

# 3. Visualize it
img_path = os.path.join("data/generated_microtubes", winner['filename'])
if os.path.exists(img_path):
    img = Image.open(img_path)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(img, cmap='gray') # Gray map: Dark=Solid, Bright=Void
    plt.title(f"The Candidate Structure\n{winner['filename']}", fontsize=14)
    plt.axis('off')
    
    save_path = "results/titanium_wood_reveal.png"
    plt.savefig(save_path, dpi=150)
    print(f"\n📸 Snapshot saved to: {save_path}")
    print("   Open this image to see if it's a lattice or a block!")
else:
    print(f"❌ Image file {winner['filename']} not found in data folder.")