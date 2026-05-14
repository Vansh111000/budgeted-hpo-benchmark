import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import numpy as np

os.makedirs('aggregation/new_results/v2_analysis_plots', exist_ok=True)

df = pd.read_csv('aggregation/new_results/summary_table.csv')

# 1. Compare vs TPE and Random
methods_base = ['catsplus_v2', 'tpe', 'random']
df_base = df[df['method'].isin(methods_base)]

print("### CATS+ V2 vs Baselines (TPE & Random Search)\n")
print("| Instance | Seed | CATS+ V2 | TPE | Random | Winner |")
print("|---|---|---|---|---|---|")

instances = df['instance'].unique()
seeds = df['seed'].unique()

for inst in instances:
    for seed in seeds:
        sub = df_base[(df_base['instance'] == inst) & (df_base['seed'] == seed)]
        scores = {row['method']: row['best_score'] for _, row in sub.iterrows()}
        if len(scores) < 3: continue
        
        v2_score = scores.get('catsplus_v2', np.nan)
        tpe_score = scores.get('tpe', np.nan)
        rand_score = scores.get('random', np.nan)
        
        best = min(scores.values())
        winner = [k for k, v in scores.items() if v == best][0]
        if winner == 'catsplus_v2': winner = '**CATS+ V2**'
        elif winner == 'tpe': winner = 'TPE'
        else: winner = 'Random'
        
        print(f"| {inst} | {seed} | {v2_score:.4f} | {tpe_score:.4f} | {rand_score:.4f} | {winner} |")

print("\n### CATS+ V2 vs Previous CATS Iterations\n")
methods_cats = ['catsplus_v2', 'catsplus', 'cats']
df_cats = df[df['method'].isin(methods_cats)]

print("| Instance | Seed | CATS+ V2 | CATS+ | CATS | Winner |")
print("|---|---|---|---|---|---|")

for inst in instances:
    for seed in seeds:
        sub = df_cats[(df_cats['instance'] == inst) & (df_cats['seed'] == seed)]
        scores = {row['method']: row['best_score'] for _, row in sub.iterrows()}
        if len(scores) < 3: continue
        
        v2_score = scores.get('catsplus_v2', np.nan)
        cp_score = scores.get('catsplus', np.nan)
        c_score = scores.get('cats', np.nan)
        
        best = min(scores.values())
        winner = [k for k, v in scores.items() if v == best][0]
        if winner == 'catsplus_v2': winner = '**CATS+ V2**'
        elif winner == 'catsplus': winner = 'CATS+'
        else: winner = 'CATS'
        
        print(f"| {inst} | {seed} | {v2_score:.4f} | {cp_score:.4f} | {c_score:.4f} | {winner} |")


# Generate Individual Convergence Graphs
print("\nGenerating graphs...")
plt.style.use('dark_background')

# Let's plot the convergence for 3 representative instances (seed 0)
for inst in [3945, 146212, 168329]:
    plt.figure(figsize=(10, 6))
    
    for method, color in [('catsplus_v2', '#00ffcc'), ('tpe', '#ff3366'), ('random', '#aaaaaa')]:
        file_path = f'results/{method}_lcbench_inst{inst}_seed0.csv'
        if os.path.exists(file_path):
            trace = pd.read_csv(file_path)
            cummin = trace['score'].cummin()
            plt.plot(range(1, len(cummin) + 1), cummin, label=method, color=color, linewidth=2)
            
    plt.title(f'Convergence Comparison: Instance {inst} (Seed 0)', fontsize=14, color='white')
    plt.xlabel('Evaluations', fontsize=12, color='white')
    plt.ylabel('Best Score (Lower is Better)', fontsize=12, color='white')
    plt.legend()
    plt.grid(True, alpha=0.2)
    
    out_path = f'aggregation/new_results/v2_analysis_plots/convergence_inst{inst}.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='black')
    plt.close()

print("Graphs generated in aggregation/new_results/v2_analysis_plots/")
