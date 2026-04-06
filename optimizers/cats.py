# optimizers/cats.py
import csv, json, os, random, copy
from datetime import datetime
from runner.evaluate import make_benchmark, run_one, BENCH_NAME
HEADER = ["timestamp","optimizer","bench","instance","seed","trial_id","score","cfg_json","out_json"]
def perturb_cfg(cfg, cs, strength=0.15):
    new_cfg = copy.deepcopy(cfg)
    for hp in cs.get_hyperparameters():
        name = hp.name
        cls = hp.__class__.__name__
        # categorical: rarely change
        if hasattr(hp, "choices"):
            if random.random() < 0.15:
                      new_cfg[name] = random.choice(list(hp.choices))
            continue
        # float
        if hasattr(hp, "lower") and hasattr(hp, "upper") and "Float" in cls:
            lo, hi = float(hp.lower), float(hp.upper)
            x = float(new_cfg[name])
            step = (hi - lo) * strength
            x2 = x + random.uniform(-step, step)
            x2 = min(max(x2, lo), hi)
            new_cfg[name] = x2
            continue
        # int
        if hasattr(hp, "lower") and hasattr(hp, "upper") and "Integer" in cls:
            lo, hi = int(hp.lower), int(hp.upper)
            x = int(new_cfg[name])
            step = max(1, int((hi - lo) * strength))
            x2 = x + random.randint(-step, step)
            x2 = min(max(x2, lo), hi)
            new_cfg[name] = int(x2)
            continue
    return new_cfg
def run_cats(instance="3945", seed=0, n_evals=200, out_dir="results", stage1_frac=0.6, k=10):
    random.seed(seed)
    bench = make_benchmark(instance)
    cs = bench.config_space
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, f"cats_{BENCH_NAME}_inst{instance}_seed{seed}.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(HEADER)
    n1 = int(n_evals * stage1_frac)
    n2 = n_evals - n1
    records = []  # (score, cfg)
    # Stage 1: broad screening
    for t in range(n1):
        cfg = cs.sample_configuration(1).get_dictionary()
        score, out = run_one(bench, cfg, seed=seed)
        score = float(score)
        out = {k : float(v) for k,v in out.items()}
        records.append((score, cfg))
        with open(out_csv, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([datetime.now().isoformat(),"cats",BENCH_NAME,instance,seed,t,score,json.dumps(cfg),json.dumps(out)])
    records.sort(key=lambda x: x[0])  # minimise
    elites = [cfg for (_, cfg) in records[:k]]
    # Stage 2: refine around elites
    for i in range(n2):
        base = random.choice(elites)
        cfg2 = perturb_cfg(base, cs, strength=0.15)
        eval_id = n1 + i
        score, out = run_one(bench, cfg2, seed=seed)
        score = float(score) 
        out = {k : float(v) for k,v in out.items()}
        with open(out_csv, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([datetime.now().isoformat(),"cats",BENCH_NAME,instance,seed,eval_id,score,json.dumps(cfg2),json.dumps(out)])
    print("CSV SAVED:", out_csv)
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--instance", default="3945")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_evals", type=int, default=200)
    args = p.parse_args()
    run_cats(instance=args.instance, seed=args.seed, n_evals=args.n_evals)