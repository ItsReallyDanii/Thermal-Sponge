import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from src.benchmark_baselines import generate_vertical_fins, generate_grid, generate_random_noise, solve_steady_heat
from src.audit_efficiency import solve_flow_resistance, load_ai_design

OUTPUT_DIR = "results/baselines/"
AI_IMAGE_PATH = "results/thermal_design/MaxCooling_Heavy_structure.png"

def get_metrics(img, name):
    # 1. Thermal
    k_map = np.where(img < 0.5, 1.0, 0.05) # Solid=1, Void=0.05
    T = solve_steady_heat(k_map)
    dTdx = T[:, -1] - T[:, -2]
    flux = np.sum(-k_map[:, -1] * dTdx)
    
    # 2. Flow Resistance
    # Invert for flow: Void(1) is high permeability
    # Binary: 1=Void, 0=Solid
    binary_void = (img > 0.5).astype(float)
    res, _ = solve_flow_resistance(binary_void)
    
    return {"name": name, "flux": flux, "resistance": res, "type": "baseline"}

def main():
    print("⚔️  Running Multi-Physics Showdown...")
    data = []
    
    # 1. Baselines
    print("   Simulating Fins...")
    for n in [10, 20, 40]:
        img = generate_vertical_fins(num_fins=n, thickness=4)
        data.append(get_metrics(img, f"Fins_{n}"))
        
    print("   Simulating Grids...")
    for n in [8, 16, 32]:
        img = generate_grid(num_cells=n, thickness=2)
        data.append(get_metrics(img, f"Grid_{n}"))
        
    print("   Simulating Random...")
    for d in [0.4, 0.6]:
        img = generate_random_noise(density=d)
        data.append(get_metrics(img, f"Random_{d}"))
        
    # 2. Your AI Design
    print("   Simulating AI...")
    ai_img = load_ai_design(AI_IMAGE_PATH)
    if ai_img is not None:
        # ai_img is already 1=Void, 0=Solid? 
        # Check load_ai_design in audit_efficiency. It returns Binary Void=1.
        # We need to invert it for get_metrics which expects Image (White=Void, Black=Solid)?
        # Actually get_metrics logic: img < 0.5 is SOLID. 
        # load_ai_design returns binary where 1=VOID. 
        # So we pass it directly? No, let's just run solvers directly to be safe.
        
        # Thermal (Needs K map)
        k_map_ai = np.where(ai_img < 0.5, 1.0, 0.05) # If 1=Void, <0.5 is False -> Solid? 
        # Wait, if ai_img is 1=Void, then ai_img < 0.5 selects 0 (Solid). Correct.
        T_ai = solve_steady_heat(k_map_ai)
        dTdx_ai = T_ai[:, -1] - T_ai[:, -2]
        flux_ai = np.sum(-k_map_ai[:, -1] * dTdx_ai)
        
        # Flow (Needs Binary Void)
        res_ai, _ = solve_flow_resistance(ai_img)
        
        data.append({"name": "AI_Coral", "flux": flux_ai, "resistance": res_ai, "type": "AI"})

    # 3. Plot
    df = pd.DataFrame(data)
    print(df)
    
    plt.figure(figsize=(8,6))
    
    # Plot Baselines
    bases = df[df['type'] == 'baseline']
    plt.scatter(bases['resistance'], bases['flux'], c='gray', label='Baselines')
    
    # Label Baselines
    for _, row in bases.iterrows():
        plt.text(row['resistance'], row['flux'], "  "+row['name'], fontsize=8)
        
    # Plot AI
    ai = df[df['type'] == 'AI']
    plt.scatter(ai['resistance'], ai['flux'], c='red', s=100, label='AI Generative')
    
    plt.xlabel("Flow Resistance (Lower is Better)")
    plt.ylabel("Heat Flux (Higher is Better)")
    plt.title("The Multi-Physics Pareto Front")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "multiphysics_frontier.png"))
    print("\n✅ Chart saved: results/baselines/multiphysics_frontier.png")

if __name__ == "__main__":
    main()