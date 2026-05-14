import os
import csv
import math
from collections import defaultdict
import matplotlib.pyplot as plt

RESULTS_DIR = "results"
FIG_OUT = "figures/new_results/budget_analysis.png"
TABLE_OUT = "aggregation/new_results/budget_analysis.md"
METHODS = ["cats", "catsplus", "catsplus_v2", "random", "tpe"]
BUDGETS = [10, 20, 50, 100, 200, 500, 1000]

def get_best_so_far(scores):
    best = []
    cur = None
    for s in scores:
        cur = s if cur is None else min(cur, s)
        best.append(cur)
    return best

def main():
    data = defaultdict(lambda: defaultdict(dict))
    
    for fname in os.listdir(RESULTS_DIR):
        if not fname.endswith(".csv"): continue
        path = os.path.join(RESULTS_DIR, fname)
        
        parts = fname.replace(".csv", "").split("_")
        if "lcbench" not in parts: continue
        
        idx = parts.index("lcbench")
        method = "_".join(parts[:idx])
        if method not in METHODS: continue
        
        inst_str = parts[idx+1].replace("inst", "")
        seed_str = parts[idx+2].replace("seed", "")
        
        scores = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                scores.append(float(row["score"]))
        
        data[method][inst_str][seed_str] = get_best_so_far(scores)
        
    instances = set()
    seeds = set()
    for m in METHODS:
        for i in data[m]:
            instances.add(i)
            for s in data[m][i]:
                seeds.add(s)
                
    instances = sorted(list(instances))
    seeds = sorted(list(seeds))
    
    ranks_at_budget = defaultdict(list)
    
    for b in BUDGETS:
        rank_sums = defaultdict(float)
        rank_counts = defaultdict(int)
        
        for inst in instances:
            for seed in seeds:
                cell_scores = {}
                valid = True
                for m in METHODS:
                    if inst not in data[m] or seed not in data[m][inst]:
                        valid = False
                        break
                    
                    curve = data[m][inst][seed]
                    idx = min(b - 1, len(curve) - 1)
                    if idx < 0:
                        valid = False
                        break
                    cell_scores[m] = curve[idx]
                    
                if not valid: continue
                
                sorted_methods = sorted(cell_scores.keys(), key=lambda x: cell_scores[x])
                scores_sorted = [cell_scores[m] for m in sorted_methods]
                
                ranks = {}
                i = 0
                while i < len(sorted_methods):
                    j = i
                    while j < len(sorted_methods) and scores_sorted[j] == scores_sorted[i]:
                        j += 1
                    avg_r = (i + 1 + j) / 2.0
                    for k in range(i, j):
                        ranks[sorted_methods[k]] = avg_r
                    i = j
                    
                for m in METHODS:
                    rank_sums[m] += ranks[m]
                    rank_counts[m] += 1
                    
        for m in METHODS:
            if rank_counts[m] > 0:
                ranks_at_budget[m].append(rank_sums[m] / rank_counts[m])
            else:
                ranks_at_budget[m].append(math.nan)
                
    os.makedirs(os.path.dirname(FIG_OUT), exist_ok=True)
    plt.figure(figsize=(10, 6))
    for m in METHODS:
        plt.plot(BUDGETS, ranks_at_budget[m], marker='o', label=m)
    
    plt.xlabel("Evaluation Budget (n_evals)")
    plt.ylabel("Average Rank (lower is better)")
    plt.title("Average Rank vs. Evaluation Budget")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(FIG_OUT, dpi=200)
    print(f"Saved plot to {FIG_OUT}")
    
    os.makedirs(os.path.dirname(TABLE_OUT), exist_ok=True)
    with open(TABLE_OUT, "w", encoding="utf-8") as f:
        header = "| Method | " + " | ".join([f"B={b}" for b in BUDGETS]) + " |"
        sep = "|---|" + "|".join(["---:" for _ in BUDGETS]) + "|"
        f.write(header + "\n")
        f.write(sep + "\n")
        
        # Sort methods by their rank at the final budget
        sorted_m = sorted(METHODS, key=lambda x: ranks_at_budget[x][-1] if ranks_at_budget[x] else 0)
        
        for m in sorted_m:
            row = f"| {m} | " + " | ".join([f"{r:.3f}" if not math.isnan(r) else "N/A" for r in ranks_at_budget[m]]) + " |"
            f.write(row + "\n")
            
    print(f"Saved table to {TABLE_OUT}")

if __name__ == "__main__":
    main()
