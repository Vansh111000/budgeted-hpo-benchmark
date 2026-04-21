# optimizers/catsplus.py
#
# CATS+ (this work): Elite-Distribution Adaptive Sampling
#
# Summary:
#   CATS+ improves the original CATS by replacing "local perturbation" with an
#   elite-distribution update (Estimation-of-Distribution / Cross-Entropy style).
#
# Run (from repo root):
#   python optimizers/catsplus.py --instance 3945 --seed 0 --n_evals 200
#
# Use --light_log to store out_json as {} and reduce CSV size.

import argparse
import csv
import json
import math
import os
import random
from datetime import datetime
from statistics import mean, pstdev

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

def validate_cfg(cs, cfg: dict):
    # Validate cfg against ConfigSpace constraints/conditions.
    # Return a clean dict if valid, else None.
    if Configuration is None:
        return cfg
    try:
        conf = Configuration(cs, values=cfg)
        cs.check_configuration(conf)
        return conf.get_dictionary()
    except Exception:
        return None

def sample_from_elite_model(cs, elites, rng, *, prior_alpha, sigma_scale, min_sigma, p_uniform_numeric):
    # Build a per-hyperparameter sampling distribution from elites and sample one cfg.
    cfg = {}

    elite_vals = {}
    for hp in cs.get_hyperparameters():
        elite_vals[hp.name] = [e[hp.name] for e in elites if hp.name in e]

    for hp in cs.get_hyperparameters():
        name = hp.name

        # categorical
        if _is_categorical(hp):
            choices = list(hp.choices)
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

        # numeric
        if _is_numeric(hp):
            lo = float(hp.lower)
            hi = float(hp.upper)

            # numeric exploration even in model mode
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
            if log_scale:
                safe_lo = max(lo, 1e-12)
                vals_t = [math.log(_clamp(v, safe_lo, hi)) for v in vals]
                mu = mean(vals_t)
                sigma = pstdev(vals_t) if len(vals_t) > 1 else min_sigma
                sigma = max(sigma, min_sigma) * sigma_scale
                x = rng.gauss(mu, sigma)
                x = _clamp(x, math.log(safe_lo), math.log(hi))
                v = math.exp(x)
            else:
                mu = mean(vals)
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

        raise ValueError(f"Unsupported hyperparameter type: {hp.__class__.__name__} ({hp})")

    return cfg

def catsplus(instance, seed, n_evals, out_dir,
            elite_frac, warmup_frac, k_min,
            p_global, p_uniform_numeric,
            prior_alpha, sigma_scale, min_sigma,
            light_log):

    rng = random.Random(seed)

    bench = make_benchmark(instance)
    cs = bench.config_space

    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, f"catsplus_{BENCH_NAME}_inst{instance}_seed{seed}.csv")

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)

        obs = []  # list of (score, cfg_dict)
        best_score = None

        def log_row(trial_id, score, cfg, out):
            nonlocal best_score
            if best_score is None or score < best_score:
                best_score = score
            writer.writerow([
                datetime.now().isoformat(),
                "catsplus",
                BENCH_NAME,
                instance,
                seed,
                trial_id,
                float(score),
                json.dumps(cfg),
                json.dumps({} if light_log else out),
            ])

        n_warmup = max(10, int(warmup_frac * n_evals))
        n_warmup = min(n_warmup, n_evals)

        # Warmup: random trials
        for t in range(n_warmup):
            cfg = cs.sample_configuration(1).get_dictionary()
            score, out = run_one(bench, cfg, seed=seed)
            score = float(score)
            obs.append((score, cfg))
            log_row(t, score, cfg, out)

            if (t + 1) % 10 == 0 or t == n_warmup - 1:
                f.flush()
                print(f"[catsplus warmup {t+1}/{n_warmup}] score={score:.6f} best={best_score:.6f}")

        # Main: elite-distribution sampling
        for trial_id in range(n_warmup, n_evals):
            obs_sorted = sorted(obs, key=lambda x: x[0])
            elite_n = max(k_min, int(elite_frac * len(obs_sorted)))
            elite_n = min(elite_n, len(obs_sorted))
            elites = [cfg for (_, cfg) in obs_sorted[:elite_n]]

            if rng.random() < p_global:
                cfg_try = cs.sample_configuration(1).get_dictionary()
            else:
                cfg_try = sample_from_elite_model(
                    cs, elites, rng,
                    prior_alpha=prior_alpha,
                    sigma_scale=sigma_scale,
                    min_sigma=min_sigma,
                    p_uniform_numeric=p_uniform_numeric,
                )

            cfg_valid = validate_cfg(cs, cfg_try)
            if cfg_valid is None:
                cfg_valid = cs.sample_configuration(1).get_dictionary()

            score, out = run_one(bench, cfg_valid, seed=seed)
            score = float(score)

            obs.append((score, cfg_valid))
            log_row(trial_id, score, cfg_valid, out)

            if (trial_id + 1) % 25 == 0 or trial_id == n_evals - 1:
                f.flush()
                print(f"[catsplus {trial_id+1}/{n_evals}] score={score:.6f} best={best_score:.6f} elites={elite_n}")

    print("CSV SAVED:", out_csv)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--instance", default="3945")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_evals", type=int, default=200)
    p.add_argument("--out_dir", default="results")

    p.add_argument("--elite_frac", type=float, default=0.15)
    p.add_argument("--warmup_frac", type=float, default=0.15)
    p.add_argument("--k_min", type=int, default=10)

    p.add_argument("--p_global", type=float, default=0.20)
    p.add_argument("--p_uniform_numeric", type=float, default=0.10)

    p.add_argument("--prior_alpha", type=float, default=1.0)
    p.add_argument("--sigma_scale", type=float, default=1.0)
    p.add_argument("--min_sigma", type=float, default=0.15)

    p.add_argument("--light_log", action="store_true")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    catsplus(
        instance=str(args.instance),
        seed=int(args.seed),
        n_evals=int(args.n_evals),
        out_dir=str(args.out_dir),
        elite_frac=float(args.elite_frac),
        warmup_frac=float(args.warmup_frac),
        k_min=int(args.k_min),
        p_global=float(args.p_global),
        p_uniform_numeric=float(args.p_uniform_numeric),
        prior_alpha=float(args.prior_alpha),
        sigma_scale=float(args.sigma_scale),
        min_sigma=float(args.min_sigma),
        light_log=bool(args.light_log),
    )
