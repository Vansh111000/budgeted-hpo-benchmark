# aggregation/paper_figures.py
#
# Purpose:
#   Read aggregation/summary_table.csv and the raw run CSV logs in collected_results/
#   and generate paper-ready outputs:
#     1) aggregation/ranking_table.csv
#     2) aggregation/ranking_table.md   (easy to paste into the paper)
#     3) figures/overlay_inst<id>.png   (one plot per instance; mean best-so-far over seeds)
#
# Assumptions:
#   - You already ran:
#       python aggregation/aggregate_runs.py
#     so that aggregation/summary_table.csv exists.
#   - collected_results/ contains the raw run CSVs referenced by the "file" column.
#
# How to run (from repo root):
#   python aggregation/paper_figures.py

import csv
import os
from collections import defaultdict
import math

import matplotlib.pyplot as plt

SUMMARY_PATH = "aggregation/summary_table.csv"
CSV_DIR = "colleceted_results/collected_results_1000_combined"
FIG_DIR = "figures"

# Keep method order stable for plots and tables
METHODS = ["random", "tpe", "cats"]

def read_summary():
    rows = []
    with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if row["method"] in METHODS:
                rows.append(row)
    return rows

def best_so_far_curve(csv_path):
    scores = []
    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            scores.append(float(row["score"]))

    curve = []
    cur = None
    for s in scores:
        cur = s if cur is None else min(cur, s)
        curve.append(cur)
    return curve

def mean_curves(curves):
    '''Average multiple curves pointwise (truncate to shortest length).'''
    if not curves:
        return []
    L = min(len(c) for c in curves)
    curves = [c[:L] for c in curves]
    out = []
    for i in range(L):
        out.append(sum(c[i] for c in curves) / len(curves))
    return out

def write_ranking_table(summary_rows):
    '''Compute average ranks across (instance, seed) cells using best_score.'''
    # Group by (instance, seed)
    grouped = defaultdict(list)
    for row in summary_rows:
        key = (row["instance"], row["seed"])
        grouped[key].append(row)

    rank_sums = defaultdict(float)
    rank_counts = defaultdict(int)
    win_counts = defaultdict(int)
    win_instances = defaultdict(list)

    for key, rows in grouped.items():
        present = {r["method"] for r in rows}
        # Require all methods to be present for a fair cell
        if not all(m in present for m in METHODS):
            continue

        rows_sorted = sorted(rows, key=lambda r: float(r["best_score"]))  # minimization
        scores = [float(r["best_score"]) for r in rows_sorted]

        # Tie-aware average rank
        ranks = {}
        i = 0
        while i < len(rows_sorted):
            j = i
            while j < len(rows_sorted) and scores[j] == scores[i]:
                j += 1
            avg_rank = (i + 1 + j) / 2.0
            for k in range(i, j):
                ranks[rows_sorted[k]["method"]] = avg_rank
            i = j

        # Winner count: all methods tied for best count as wins
        best_val = scores[0]
        for r in rows_sorted:
            if float(r["best_score"]) == best_val:
                win_counts[r["method"]] += 1
                win_instances[r["method"]].append(f"Inst {r['instance']} Seed {r['seed']}")
            else:
                break

        for m in METHODS:
            rank_sums[m] += ranks[m]
            rank_counts[m] += 1

    out_rows = []
    for m in METHODS:
        avg_rank = (rank_sums[m] / rank_counts[m]) if rank_counts[m] else math.nan
        out_rows.append({
            "method": m,
            "avg_rank_lower_is_better": avg_rank,
            "cells_counted": rank_counts[m],
            "win_count": win_counts[m],
            "win_pairs": ", ".join(win_instances[m])
        })

    out_rows.sort(key=lambda r: (r["avg_rank_lower_is_better"] if not math.isnan(r["avg_rank_lower_is_better"]) else 1e9))

    os.makedirs("aggregation", exist_ok=True)

    with open("aggregation/ranking_table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["method","avg_rank_lower_is_better","cells_counted","win_count","win_pairs"])
        w.writeheader()
        w.writerows(out_rows)

    # Markdown version for easy copy-paste into the paper
    with open("aggregation/ranking_table.md", "w", encoding="utf-8") as f:
        f.write("| Method | Avg Rank (↓) | Cells | Wins | Winning Pairs |\n")
        f.write("|---|---:|---:|---:|---|\n")
        for r in out_rows:
            f.write(f"| {r['method']} | {r['avg_rank_lower_is_better']:.3f} | {r['cells_counted']} | {r['win_count']} | {r['win_pairs']} |\n")

    print("Wrote aggregation/ranking_table.csv and aggregation/ranking_table.md")

def make_overlay_plots(summary_rows):
    '''One overlay plot per instance: mean best-so-far over seeds for each method.'''
    files = defaultdict(lambda: defaultdict(list))
    for row in summary_rows:
        inst = row["instance"]
        m = row["method"]
        fname = row["file"]
        files[inst][m].append(os.path.join(CSV_DIR, fname))

    os.makedirs(FIG_DIR, exist_ok=True)

    for inst, per_method in sorted(files.items(), key=lambda kv: int(kv[0])):
        if not all(m in per_method for m in METHODS):
            continue

        plt.figure()

        for m in METHODS:
            curves = []
            for path in per_method[m]:
                if os.path.exists(path):
                    curves.append(best_so_far_curve(path))
            mean_curve = mean_curves(curves)

            if not mean_curve:
                continue

            x = list(range(len(mean_curve)))
            plt.plot(x, mean_curve, label=m)

        plt.xlabel("evaluation index")
        plt.ylabel("best loss so far (lower is better)")
        plt.title(f"lcbench instance {inst}: mean best-so-far over seeds")
        plt.legend()
        plt.tight_layout()

        out_path = os.path.join(FIG_DIR, f"overlay_inst{inst}.png")
        plt.savefig(out_path, dpi=200)
        plt.close()
        print("Wrote", out_path)

def main():
    summary_rows = read_summary()
    write_ranking_table(summary_rows)
    make_overlay_plots(summary_rows)

if __name__ == "__main__":
    main()
