# Project: Thermal Sponge Bio-Audit (Batch 1)

## 🎯 Objective
We are performing a "Bio-Audit" on the physics engine to resolve a "Threshold Trap" error where the solver was using inconsistent porosity definitions (0.80 vs 0.50).

## 🛠️ Task: Centralize Physics Constants
You are the Lead Architect. Your goal is to establish a "Single Source of Truth" for physical constants and refactor the thermal solver to use it.

### Step 1: Create `src/constants.py`
Create a new file `src/constants.py` containing exactly:
```python
# src/constants.py
# GLOBAL PHYSICS TRUTH
# Normalized Images: 0.0 (Black/Solid) to 1.0 (White/Void)

# The critical threshold: Pixel > 0.60 is VOID (Fluid/Air).
# Pixel <= 0.60 is SOLID (Material).
VOID_THRESHOLD = 0.60

# Conductivity (W/mK)
K_SOLID = 1.0
K_VOID = 0.05