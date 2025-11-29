"""
morphology_dashboard.py
-------------------------------------
Creates a summary dashboard comparing
Pre- vs Post-Hybrid morphological convergence.
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import json
import os

# Paths
PRE_PATH = "results/morphology_metrics/latent_overlap_Pre-Hybrid.png"
POST_PATH = "results/morphology_metrics/latent_overlap_Post-Hybrid.png"
METRICS_PATH = "results/morphology_metrics/metrics_summary.json"
SAVE_PATH = "results/morphology_metrics/morphology_dashboard.png"

# Load images
img_pre = mpimg.imread(PRE_PATH)
img_post = mpimg.imread(POST_PATH)

# Load metrics (optional)
metrics = {}
if os.path.exists(METRICS_PATH):
    with open(METRICS_PATH, "r") as f:
        metrics = json.load(f)

# Build figure
fig, axes = plt.subplots(1, 3, figsize=(14, 5))

# Pre-hybrid map
axes[0].imshow(img_pre)
axes[0].axis("off")
axes[0].set_title("Latent Space: Pre-Hybrid")

# Post-hybrid map
axes[1].imshow(img_post)
axes[1].axis("off")
axes[1].set_title("Latent Space: Post-Hybrid")

# Metrics bar chart (if available)
if metrics:
    labels = list(metrics.keys())
    pre_vals = [metrics[k]["pre"] for k in labels]
    post_vals = [metrics[k]["post"] for k in labels]

    axes[2].barh(labels, pre_vals, color="gray", alpha=0.6, label="Pre")
    axes[2].barh(labels, post_vals, color="green", alpha=0.6, label="Post")
    axes[2].invert_yaxis()
    axes[2].set_title("Morphological Metrics")
    axes[2].legend()
else:
    axes[2].text(0.5, 0.5, "No metrics JSON found", ha="center", va="center")
    axes[2].axis("off")

plt.tight_layout()
plt.savefig(SAVE_PATH, dpi=300)
plt.close()

print(f"✅ Dashboard saved to {SAVE_PATH}")
