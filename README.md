# Inverse Design of Functionally Graded Porous Media
### *A Physics-Informed Generative Approach to Biological & Thermal Transport*

![Status](https://img.shields.io/badge/Status-Research_Artifact-blue)
![Domain](https://img.shields.io/badge/Domain-SciML-green)
![Framework](https://img.shields.io/badge/Framework-PyTorch-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

**Key Finding:** Validated a generative design engine that creates microstructures with **3x hydraulic stiffness** compared to biological xylem and **Pareto-optimal cooling efficiency** compared to standard engineering baselines.

---

## 📄 Abstract

Transport microstructures—whether in biological xylem or electronic cooling plates—are often limited by a trade-off between **efficiency** (flow rate/heat flux) and **material cost** (density/pressure drop). Traditional topology optimization is computationally expensive, while standard biomimicry often copies evolutionary constraints (like growth and self-repair) that are irrelevant to engineering.

This project introduces a **"Bio-Audited" Generative Design Framework**. By training a surrogate-assisted autoencoder on biological data and fine-tuning it via differentiable physics solvers, we created a **"Material Compiler"** capable of inverse-designing microstructures for specific multi-physics targets.

We demonstrate the framework's generalizability across two distinct domains:
1.  **Biological Hydraulics:** Identifying a "minimal viable xylem" that matches biological flow rates with only **15% porosity** (vs. ~43% in nature), quantifying the "biological safety factor."
2.  **Thermal Management:** Generating "organic fin" topologies that achieve the cooling performance of chaotic foams while maintaining the low hydraulic resistance of straight fins.

---

## 📊 Key Results

### 1. The Biological Efficiency Gap
We mapped the trade-off between **Flow Rate** (Simulated via Darcy solver) and **Stiffness Potential** (Heuristic $E \propto \rho^2$).
* **Finding:** The AI identified a Pareto front of designs that are significantly stiffer than biological xylem for the equivalent hydraulic conductivity.
* **Implication:** Much of the void space in real xylem is hydraulically redundant in steady-state conditions, optimized instead for cavitation resistance and repair—constraints we can remove for synthetic engineering.

![Trade-off Plot](results/flow_stiffness_tradeoff.png)
*(Figure 1: AI-optimized microstructures [circled] vs. biological baselines.)*

### 2. Thermal Generalization (The "Cooling Coral")
We retrained the physics engine to solve the **Steady-State Heat Diffusion Equation** ($\nabla \cdot (k \nabla T) = 0$) to design heat sinks for high-performance electronics.
* **Benchmark:** We compared AI designs against Straight Fins, Grids, and Random Noise (Foam).
* **The Win:** While Random Noise achieved high cooling flux, it suffered from massive pressure drop. The AI design occupied the optimal "High Flux / Low Resistance" quadrant.

### 3. Manufacturability & Control
Unlike generative models that produce "pixel dust," this framework enforces structural connectivity.
* **Functionally Graded Materials (FGM):** We successfully generated continuous beams transitioning from **Dense ($E_{high}$)** to **Porous ($E_{low}$)**, validated for SLA 3D printing.
* **Design Manifold:** We demonstrated control by sweeping the latent space to generate a smooth transition of morphologies.

![Gradient Beam](results/gradient_beam/gradient_beam_render.png)
*(Figure 2: 3D-printable functionally graded beam generated via latent interpolation.)*

---

## 🧠 System Architecture

The framework consists of three coupled modules forming a closed-loop design engine:

```mermaid
graph TD
    A[Geometry Generator] -->|Latent Code z| B(Decoder)
    B -->|Microstructure| C{Physics Surrogate}
    C -->|Predict Flow/Heat| D[Optimizer]
    D -->|Gradient Update| A
    B -->|Validation| E[FEM/FDM Solver]
