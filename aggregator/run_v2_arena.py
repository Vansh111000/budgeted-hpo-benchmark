import os
import sys
import csv
import shutil
from collections import defaultdict
import math

# Add the current directory to sys.path
sys.path.append(os.path.join(os.getcwd(), 'aggregator'))

import aggregate_runs
import paper_figures

# Flexible reader found earlier
def patched_read_scores(csv_path: str):
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        # score is column index 6 usually. check header
        idx = header.index("score") if "score" in header else 6
        scores = []
        for row in reader:
            if not row: continue
            scores.append(float(row[idx]))
        return scores

def run_arena():
    arena_dir = "colleceted_results/v2_arena"
    os.makedirs(arena_dir, exist_ok=True)
    
    # 1. Gather files
    # Methods from previous collection
    source_dir = "colleceted_results/catplus"
    for f in os.listdir(source_dir):
        if f.startswith(("tpe_", "random_", "cats_", "catsplus_")) and f.endswith(".csv"):
            shutil.copy(os.path.join(source_dir, f), os.path.join(arena_dir, f))
    
    # Files from latest runs
    results_dir = "results"
    for f in os.listdir(results_dir):
        if f.startswith("catsplus_v2_") and f.endswith(".csv"):
            shutil.copy(os.path.join(results_dir, f), os.path.join(arena_dir, f))
            
    # 2. Setup Aggregator
    output_dir = "aggregation/v2_vs_tpe"
    fig_dir = "figures/v2_vs_tpe"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    
    summary_csv = os.path.join(output_dir, "summary_table.csv")
    
    # Fix regex to allow underscores in method name
    import re
    aggregate_runs.PATTERN = re.compile(r"^(?P<method>[a-zA-Z0-9_]+)_lcbench_inst(?P<instance>\d+)_seed(?P<seed>\d+)\.csv$")
    
    aggregate_runs.read_scores = patched_read_scores
    paper_figures.SUMMARY_PATH = summary_csv
    paper_figures.CSV_DIR = arena_dir
    paper_figures.FIG_DIR = fig_dir
    paper_figures.METHODS = ["random", "tpe", "cats", "catsplus", "catsplus_v2"]

    print("--- Running Aggregation ---")
    aggregate_runs.main(input_dir=arena_dir, out_csv=summary_csv)
    
    print("--- Running Figures and Rankings ---")
    summary_rows = paper_figures.read_summary()
    
    # Ranking logic
    grouped = defaultdict(list)
    for row in summary_rows:
        key = (row["instance"], row["seed"])
        grouped[key].append(row)

    rank_sums = defaultdict(float)
    rank_counts = defaultdict(int)
    win_counts = defaultdict(int)

    count = 0
    for key, rows in grouped.items():
        present = {r["method"] for r in rows}
        if not all(m in present for m in paper_figures.METHODS):
            continue
        count += 1
        rows_sorted = sorted(rows, key=lambda r: float(r["best_score"]))
        scores = [float(r["best_score"]) for r in rows_sorted]
        
        ranks = {}
        i = 0
        while i < len(rows_sorted):
            j = i
            while j < len(rows_sorted) and scores[j] == scores[i]:
                j += 1
            avg_rank = (i + 1 + j) / 2.0
            for k in range(i, j): ranks[rows_sorted[k]["method"]] = avg_rank
            i = j

        best_val = scores[0]
        for r in rows_sorted:
            if float(r["best_score"]) == best_val: win_counts[r["method"]] += 1
            else: break

        for m in paper_figures.METHODS:
            rank_sums[m] += ranks[m]
            rank_counts[m] += 1

    out_rows = []
    for m in paper_figures.METHODS:
        avg_rank = (rank_sums[m] / rank_counts[m]) if rank_counts[m] else math.nan
        out_rows.append({"method": m, "avg_rank": avg_rank, "wins": win_counts[m]})
    
    out_rows.sort(key=lambda x: x["avg_rank"])
    
    ranking_md = os.path.join(output_dir, "ranking_table.md")
    with open(ranking_md, "w") as f:
        f.write("# CATS+ V2 VS TPE\n\n")
        f.write(f"Counted {count} instance-seed pairs.\n\n")
        f.write("| Method | Avg Rank | Wins |\n")
        f.write("|---|---|---:|\n")
        for r in out_rows:
            f.write(f"| {r['method']} | {r['avg_rank']:.3f} | {r['wins']} |\n")

    print(f"Ranking results saved to {ranking_md}")
    paper_figures.make_overlay_plots(summary_rows)

if __name__ == "__main__":
    run_arena()
