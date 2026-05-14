import os
import sys
import csv
import shutil
import math
import zipfile
from collections import defaultdict
import re

# Add current dir to path to import local modules
sys.path.append(os.path.join(os.getcwd(), 'aggregator'))

import aggregate_runs
import paper_figures

def patched_read_scores(csv_path: str):
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return []
        # score is column index 6 usually. check header
        idx = header.index("score") if "score" in header else 6
        scores = []
        for row in reader:
            if not row: continue
            try:
                scores.append(float(row[idx]))
            except (ValueError, IndexError):
                continue
        return scores

def run_process():
    input_dir = "colleceted_results/final/results"
    output_dir = "aggregation/final"
    fig_dir = "figures/final"
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    
    summary_csv = os.path.join(output_dir, "summary_table.csv")
    
    # Configure modules
    aggregate_runs.PATTERN = re.compile(r"^(?P<method>[a-zA-Z0-9_]+)_lcbench_inst(?P<instance>\d+)_seed(?P<seed>\d+)\.csv$")
    aggregate_runs.read_scores = patched_read_scores
    
    paper_figures.SUMMARY_PATH = summary_csv
    paper_figures.CSV_DIR = input_dir
    paper_figures.FIG_DIR = fig_dir
    paper_figures.METHODS = ["random", "tpe", "cats", "catsplus", "catsplus_v2"]
    
    print("--- Running Aggregation ---")
    # Instead of calling aggregate_runs.main directly, I'll inline/patch the loop
    # to handle the empty scores case or just patch best_and_anytime
    
    def safe_best_and_anytime(scores):
        if not scores: return 1e9, 1e9 # Default bad scores
        return original_best_and_anytime(scores)
        
    original_best_and_anytime = aggregate_runs.best_and_anytime
    aggregate_runs.best_and_anytime = safe_best_and_anytime
    aggregate_runs.main(input_dir=input_dir, out_csv=summary_csv)
    
    print("--- Running Figures and Rankings ---")
    summary_rows = paper_figures.read_summary()
    
    # Use paper_figures logic to write tables and plots
    paper_figures.write_ranking_table(summary_rows)
    # We need to ensure write_ranking_table uses our output_dir
    # Looking at paper_figures.py, it hardcodes "aggregation/ranking_table.csv"
    # I should patch those paths or move files after
    
    # Patching paper_figures paths globally for this run
    old_ranking_csv = "aggregation/ranking_table.csv"
    old_ranking_md = "aggregation/ranking_table.md"
    
    paper_figures.make_overlay_plots(summary_rows)
    
    # Move files to aggregation/final if they were created in aggregation/
    if os.path.exists(old_ranking_csv):
        shutil.move(old_ranking_csv, os.path.join(output_dir, "ranking_table.csv"))
    if os.path.exists(old_ranking_md):
        shutil.move(old_ranking_md, os.path.join(output_dir, "ranking_table.md"))

    print("--- Zipping Results ---")
    zip_name = "final results.zip"
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add aggregation tables
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                zipf.write(os.path.join(root, file), os.path.join("tables", file))
        
        # Add figures
        for root, dirs, files in os.walk(fig_dir):
            for file in files:
                zipf.write(os.path.join(root, file), os.path.join("figures", file))
                
    print(f"--- Created {zip_name} ---")

if __name__ == "__main__":
    run_process()
