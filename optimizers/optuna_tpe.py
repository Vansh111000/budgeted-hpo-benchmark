import csv, json, os
from datetime import datetime
import optuna
from optuna.samplers import TPESampler
from runner.evaluate import make_benchmark, run_one, BENCH_NAME
from runner.optuna_space import suggest_from_configspace
HEADER = ["timestamp","optimizer","bench","instance","seed","trial_id","score","cfg_json","out_json"]
def run_tpe(instance="3945", seed=0, n_evals=200, out_dir="results"):
    bench = make_benchmark(instance)
    cs = bench.config_space
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, f"tpe_{BENCH_NAME}_inst{instance}_seed{seed}.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(HEADER)
    def objective(trial):
        cfg = suggest_from_configspace(trial, cs)
        score, out = run_one(bench, cfg, seed=seed)
        score = float(score)
        out = {k : float(v) for k,v in out.items()}
        with open(out_csv, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                datetime.now().isoformat(),
                "tpe",
                BENCH_NAME,
                instance,
                seed,
                trial.number,
                score,
                json.dumps(cfg),
                json.dumps(out),
            ])
        return score
    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_evals)
    print("BEST SCORE:", study.best_value)
    print("CSV SAVED:", out_csv)
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--instance", default="3945")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_evals", type=int, default=200)
    args = p.parse_args()
    run_tpe(instance=args.instance, seed=args.seed, n_evals=args.n_evals)