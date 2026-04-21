# optimizers/catsplus_v2.py
#
# CATS+ V2: Focused instance HPO with Adaptive Sigma Decay
#
# Improvements over V1:
# 1. Strict instance binding: Overwrites OpenML_task_id to ensure focus.
# 2. Sigma Decay: Reduces exploration noise over time.
# 3. Weighted Elites: Samples mu centered closer to the best trial.

import argparse
import csv
import json
import math
import os
import random
from datetime import datetime
from statistics import mean, pstdev

import sys
import os
sys.path.append(os.getcwd())

from runner.evaluate import make_benchmark, run_one, BENCH_NAME

try:
    from ConfigSpace import Configuration
except Exception:
    Configuration = None

HEADER = ["timestamp","optimizer","bench","instance","seed","trial_id","score","cfg_json","out_json"]

def _is_categorical(hp) -> bool:
    return hasattr(hp, "choices")

def _is_numeric(hp) -> bool:
    return hasattr(hp, "lower") and hasattr(hp, "upper")

def _is_log_scaled(hp) -> bool:
    return bool(getattr(hp, "log", False)) or ("Log" in hp.__class__.__name__)

def _clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x

def validate_cfg(cs, cfg: dict, target_instance: str):
    # Ensure we are always on the correct instance
    if "OpenML_task_id" in cfg:
        cfg["OpenML_task_id"] = target_instance
    
    if Configuration is None:
        return cfg
    try:
        conf = Configuration(cs, values=cfg)
        cs.check_configuration(conf)
        return conf.get_dictionary()
    except Exception:
        # If invalid, just return the dict anyway but stick to instance
        return cfg

def sample_from_elite_model(cs, elites, rng, *, 
                           prior_alpha, sigma_scale, min_sigma, 
                           p_uniform_numeric, target_instance):
    cfg = {}
    elite_vals = {}
    for hp in cs.get_hyperparameters():
        elite_vals[hp.name] = [e[hp.name] for e in elites if hp.name in e]

    for hp in cs.get_hyperparameters():
        name = hp.name
        
        # CATEGORICAL
        if _is_categorical(hp):
            choices = list(hp.choices)
            if name == "OpenML_task_id":
                cfg[name] = target_instance
                continue
                
            counts = {c: 0.0 for c in choices}
            for v in elite_vals[name]:
                if v in counts:
                    counts[v] += 1.0

            total = sum(counts.values()) + prior_alpha * len(choices)
            probs = [(counts[c] + prior_alpha) / total for c in choices]

            r = rng.random()
            acc = 0.0
            for c, p in zip(choices, probs):
                acc += p
                if r <= acc:
                    cfg[name] = c
                    break
            else:
                cfg[name] = choices[-1]
            continue

        # NUMERIC
        if _is_numeric(hp):
            lo = float(hp.lower)
            hi = float(hp.upper)

            if rng.random() < p_uniform_numeric:
                if "Integer" in hp.__class__.__name__:
                    cfg[name] = int(rng.randint(int(lo), int(hi)))
                else:
                    cfg[name] = float(rng.uniform(lo, hi))
                continue

            vals = [float(v) for v in elite_vals[name]]
            if not vals:
                if "Integer" in hp.__class__.__name__:
                    cfg[name] = int(rng.randint(int(lo), int(hi)))
                else:
                    cfg[name] = float(rng.uniform(lo, hi))
                continue

            log_scale = _is_log_scaled(hp)
            
            # Weighted mu: weight the top 3 trials more
            best_vals = vals[:3]
            mu_base = (mean(vals) + mean(best_vals)) / 2.0
            
            if log_scale:
                safe_lo = max(lo, 1e-12)
                vals_t = [math.log(_clamp(v, safe_lo, hi)) for v in vals]
                best_t = [math.log(_clamp(v, safe_lo, hi)) for v in best_vals]
                mu = (mean(vals_t) + mean(best_t)) / 2.0
                sigma = pstdev(vals_t) if len(vals_t) > 1 else min_sigma
                sigma = max(sigma, min_sigma) * sigma_scale
                x = rng.gauss(mu, sigma)
                x = _clamp(x, math.log(safe_lo), math.log(hi))
                v = math.exp(x)
            else:
                mu = mu_base
                min_abs = (hi - lo) * min_sigma
                sigma = pstdev(vals) if len(vals) > 1 else min_abs
                sigma = max(sigma, min_abs) * sigma_scale
                v = rng.gauss(mu, sigma)
                v = _clamp(v, lo, hi)

            if "Integer" in hp.__class__.__name__:
                v = int(round(v))
                v = int(_clamp(v, int(lo), int(hi)))
            else:
                v = float(v)
            cfg[name] = v
            continue

    return cfg

def catsplus_v2(instance, seed, n_evals, out_dir,
               elite_frac, warmup_frac, k_min,
               p_global_start, p_global_end,
               p_uniform_numeric,
               prior_alpha, sigma_scale_start, sigma_scale_end, 
               min_sigma, light_log):

    rng = random.Random(seed)
    bench = make_benchmark(instance)
    cs = bench.config_space

    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, f"catsplus_v2_{BENCH_NAME}_inst{instance}_seed{seed}.csv")

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)

        obs = []
        best_score = None

        def log_row(eval_id, score, cfg, out):
            nonlocal best_score
            if best_score is None or score < best_score:
                best_score = score
            writer.writerow([
                datetime.now().isoformat(),
                "catsplus_v2",
                BENCH_NAME,
                instance,
                seed,
                eval_id,
                float(score),
                json.dumps(cfg),
                json.dumps({} if light_log else out),
            ])

        n_warmup = max(10, int(warmup_frac * n_evals))
        
        # Warmup
        for t in range(n_warmup):
            cfg = cs.sample_configuration(1).get_dictionary()
            cfg["OpenML_task_id"] = instance # Force focus
            score, out = run_one(bench, cfg, seed=seed)
            obs.append((float(score), cfg))
            log_row(t, score, cfg, out)

        # Main Loop
        for eval_id in range(n_warmup, n_evals):
            # Calculate dynamic schedules
            progress = (eval_id - n_warmup) / (n_evals - n_warmup)
            curr_sigma_scale = sigma_scale_start + progress * (sigma_scale_end - sigma_scale_start)
            curr_p_global = p_global_start + progress * (p_global_end - p_global_start)

            obs_sorted = sorted(obs, key=lambda x: x[0])
            elite_n = max(k_min, int(elite_frac * len(obs_sorted)))
            elites = [cfg for (_, cfg) in obs_sorted[:elite_n]]

            if rng.random() < curr_p_global:
                cfg_try = cs.sample_configuration(1).get_dictionary()
            else:
                cfg_try = sample_from_elite_model(
                    cs, elites, rng,
                    prior_alpha=prior_alpha,
                    sigma_scale=curr_sigma_scale,
                    min_sigma=min_sigma,
                    p_uniform_numeric=p_uniform_numeric,
                    target_instance=instance
                )

            cfg_valid = validate_cfg(cs, cfg_try, instance)
            score, out = run_one(bench, cfg_valid, seed=seed)
            
            obs.append((float(score), cfg_valid))
            log_row(eval_id, score, cfg_valid, out)

            if (eval_id + 1) % 50 == 0:
                print(f"[catsplus_v2 {eval_id+1}/{n_evals}] best={best_score:.6f} sigma={curr_sigma_scale:.2f}")

    print("CSV SAVED:", out_csv)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--instance", default="3945")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_evals", type=int, default=200)
    p.add_argument("--out_dir", default="results")
    args = p.parse_args()

    catsplus_v2(
        instance=str(args.instance),
        seed=args.seed,
        n_evals=args.n_evals,
        out_dir=args.out_dir,
        elite_frac=0.10, # Narrower elite focus
        warmup_frac=0.10,
        k_min=8,
        p_global_start=0.20,
        p_global_end=0.05, # Shrink global search over time
        p_uniform_numeric=0.10,
        prior_alpha=1.0,
        sigma_scale_start=1.2, # Start wide
        sigma_scale_end=0.2,   # End narrow (Exploitation)
        min_sigma=0.10,
        light_log=True
    )
