# aggregation/aggregate_runs.py
#
# Purpose:
#   Read raw run CSV files from collected_results/ and produce a single summary table:
#     aggregation/summary_table.csv
#
# Assumptions:
#   - You have extracted all run ZIPs into a folder named collected_results/
#   - CSV filenames follow:
#       <method>_lcbench_inst<instance>_seed<seed>.csv
#     Example:
#       random_lcbench_inst3945_seed0.csv
#   - CSV header matches the project logging contract:
#       timestamp,optimizer,bench,instance,seed,eval_id,score,cfg_json,out_json
#
# How to run (from repo root):
#   python aggregation/aggregate_runs.py
#
# Output:
#   aggregation/summary_table.csv

import csv
import glob
import os
import re
from statistics import mean

# Expected CSV header (must match protocol)
EXPECTED_HEADER = ["timestamp","optimizer","bench","instance","seed","trial_id","score","cfg_json","out_json"]

# Filename pattern:
# <method>_lcbench_inst<instance>_seed<seed>.csv
PATTERN = re.compile(r"^(?P<method>[a-zA-Z0-9_]+)_lcbench_inst(?P<instance>\d+)_seed(?P<seed>\d+)\.csv$")

def read_scores(csv_path: str):
    '''Read and validate one run CSV; return list of float scores.'''
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)

        if header != EXPECTED_HEADER and header != ["timestamp","optimizer","bench","instance","seed","eval_id","score","cfg_json","out_json"]:
            raise ValueError(
                f"Header mismatch in {csv_path}\n"
                f"Expected: {EXPECTED_HEADER}\n"
                f"Got:      {header}"
            )

        scores = []
        for row in reader:
            # score is column index 6
            scores.append(float(row[6]))
        return scores

def best_and_anytime(scores):
    '''Return (best_score, anytime_metric) where lower is better for both.'''
    best = min(scores)

    best_so_far = []
    cur = None
    for s in scores:
        cur = s if cur is None else min(cur, s)
        best_so_far.append(cur)

    # Normalized anytime metric: mean of best-so-far curve
    anytime = mean(best_so_far)
    return best, anytime

def main(input_dir: str, out_csv: str):
    rows = []
    for path in sorted(glob.glob(os.path.join(input_dir, "*.csv"))):
        base = os.path.basename(path)
        m = PATTERN.match(base)
        if not m:
            # Ignore unexpected filenames
            continue

        method = m.group("method")
        instance = m.group("instance")
        seed = m.group("seed")

        scores = read_scores(path)
        best, anytime = best_and_anytime(scores)

        rows.append([method, "lcbench", instance, seed, len(scores), best, anytime, base])

    # Sort by instance, seed, method (stable)
    rows.sort(key=lambda r: (int(r[2]), int(r[3]), r[0]))

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method","bench","instance","seed","n_evals","best_score","anytime_mean_best_so_far","file"])
        w.writerows(rows)

    print("Wrote:", out_csv)
    print("Rows:", len(rows))

if __name__ == "__main__":
    # Read CSVs from colleceted_results/collected-results_1000_new/
    main(input_dir="colleceted_results/collected_results_1000_combined", out_csv="aggregation/summary_table.csv")
