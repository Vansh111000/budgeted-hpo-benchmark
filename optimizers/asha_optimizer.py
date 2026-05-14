import os
import csv
import json
import argparse
import random
from datetime import datetime
from yahpo_gym import local_config, benchmark_set
from runner.evaluate import BENCH_NAME

# Standard project CSV header
HEADER = ["timestamp","optimizer","bench","instance","seed","trial_id","score","cfg_json","out_json"]

# Global to hold our data path
DATA_PATH = os.path.abspath("yahpo_data")

def get_bench(instance):
    local_config.init_config()
    local_config.set_data_path(DATA_PATH)
    bench = benchmark_set.BenchmarkSet(BENCH_NAME)
    bench.set_instance(instance)
    return bench

def run_asha_manual(instance="3945", seed=0, n_evals=20, out_dir="results"):
    """
    Manual implementation of ASHA (Asynchronous Successive Halving Algorithm).
    In this sequential version, we follow the Successive Halving logic:
    1. Sample N configurations (n_evals).
    2. Evaluate all at rung 0.
    3. Promote top 1/eta to rung 1, then top 1/eta to rung 2, etc.
    """
    random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, f"asha_{BENCH_NAME}_inst{instance}_seed{seed}.csv")
    
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(HEADER)

    bench = get_bench(instance)
    cs = bench.config_space
    
    # ASHA Hyperparameters
    eta = 2 # Reduction factor
    max_budget = 40
    min_budget = 1
    
    # Calculate rungs: 1, 2, 4, 8, 16, 32, then 40 (clamped)
    rungs = []
    b = min_budget
    while b < max_budget:
        rungs.append(b)
        b *= eta
    if rungs[-1] != max_budget:
        rungs.append(max_budget)

    print(f"Rungs: {rungs}")

    # Active trials: list of (config, current_rung_idx, last_score)
    # We start by sampling n_evals configs
    trials = []
    for i in range(n_evals):
        cfg = cs.sample_configuration(1).get_dictionary()
        trials.append({
            "id": f"trial_{i}",
            "cfg": cfg,
            "rung_idx": 0,
            "score": float('inf')
        })

    def log_result(trial_id, score, cfg, result, rung_val):
        with open(out_csv, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                datetime.now().isoformat(),
                "asha",
                BENCH_NAME,
                instance,
                seed,
                f"{trial_id}_rung{rung_val}",
                score,
                json.dumps(cfg),
                json.dumps({k: (float(v[0]) if isinstance(v, list) else float(v)) if hasattr(v, '__iter__') or isinstance(v, (float, int, getattr(__import__('numpy'), 'floating', float))) else v for k, v in result.items()})
            ])

    def evaluate_at_rung(trial, rung_val):
        cfg = trial["cfg"].copy()
        cfg["epoch"] = rung_val # Fidelity key for lcbench
        res = bench.objective_function([cfg])[0]
        
        # Robust score extraction (matching project logic)
        score = None
        for k in ["loss", "val_cross_entropy"]:
            if k in res:
                val = res[k]
                score = float(val[0]) if isinstance(val, list) else float(val)
                break
        if score is None:
            k = list(res.keys())[0]
            val = res[k]
            score = float(val[0]) if isinstance(val, list) else float(val)
            
        trial["score"] = score
        log_result(trial["id"], score, trial["cfg"], res, rung_val)
        return score

    # Successive Halving Loop
    current_pool = trials
    for ridx, rung_val in enumerate(rungs):
        print(f"\n--- Rung {ridx} (budget={rung_val}), Pool Size={len(current_pool)} ---")
        
        # Evaluate everyone in current pool at this rung
        for trial in current_pool:
            evaluate_at_rung(trial, rung_val)
        
        # Sort by score (minimization)
        current_pool.sort(key=lambda x: x["score"])
        
        # Keep top 1/eta for the next rung
        if ridx < len(rungs) - 1:
            next_size = max(1, len(current_pool) // eta)
            current_pool = current_pool[:next_size]
        
    print("\nFINAL BEST SCORE (manual ASHA):", current_pool[0]["score"])
    print("CSV SAVED:", out_csv)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--instance", default="3945")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_evals", type=int, default=20, help="Number of base configurations to sample")
    p.add_argument("--out_dir", default="results")
    args = p.parse_args()
    
    run_asha_manual(instance=args.instance, seed=args.seed, n_evals=args.n_evals, out_dir=args.out_dir)
