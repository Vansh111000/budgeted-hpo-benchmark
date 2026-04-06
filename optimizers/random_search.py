import csv
import json
import os
import random
from datetime import datetime

from runner.evaluate import make_benchmark, run_one, BENCH_NAME

def random_search(instance="3945", seed=2, n_trials=200, out_dir="results"):
    random.seed(seed)

    bench = make_benchmark(instance)
    os.makedirs(out_dir, exist_ok=True)

    out_csv = os.path.join(out_dir, f"random_{BENCH_NAME}_inst{instance}_seed{seed}.csv")

    best_score = None
    best_cfg = None

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "optimizer", "bench", "instance", "seed",
            "trial_id", "score", "cfg_json", "out_json"
        ])

        for t in range(n_trials):
            cfg = bench.config_space.sample_configuration(1).get_dictionary()
            score, out = run_one(bench, cfg, seed=seed)
            out = {k : float(v) for k,v in out.items()}#Changed float32 to float

            if best_score is None or score < best_score:
                best_score = score
                best_cfg = cfg

            writer.writerow([
                datetime.now().isoformat(),
                "random",
                BENCH_NAME,
                instance,
                seed,
                t,
                score,
                json.dumps(cfg),
                json.dumps(out),
            ])

            print(f"[trial {t}] score={score:.6f} best={best_score:.6f}")

    print("\nFINAL BEST SCORE:", best_score)
    print("FINAL BEST CFG:", best_cfg)
    print("CSV SAVED:", out_csv)

if __name__ == "__main__":
    random_search(instance="3945", seed=2, n_trials=200)