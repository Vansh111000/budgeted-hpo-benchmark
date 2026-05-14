import os
import sys

# Add current dir to path
sys.path.append(os.path.join(os.getcwd(), 'aggregator'))

import paper_figures

def main():
    # Configure paper_figures for the new results
    paper_figures.SUMMARY_PATH = "aggregation/new_results/summary_table.csv"
    paper_figures.CSV_DIR = "results"
    paper_figures.FIG_DIR = "figures/new_results"
    paper_figures.METHODS = ["cats", "catsplus", "catsplus_v2", "random", "tpe"]
    
    print("--- Generating Figures and Ranking Tables ---")
    summary_rows = paper_figures.read_summary()
    
    # Generate ranking tables
    paper_figures.write_ranking_table(summary_rows)
    
    # Move them to the correct folder
    import shutil
    os.makedirs("aggregation/new_results", exist_ok=True)
    if os.path.exists("aggregation/ranking_table.csv"):
        shutil.move("aggregation/ranking_table.csv", "aggregation/new_results/ranking_table.csv")
    if os.path.exists("aggregation/ranking_table.md"):
        shutil.move("aggregation/ranking_table.md", "aggregation/new_results/ranking_table.md")
        
    # Generate figures
    paper_figures.make_overlay_plots(summary_rows)
    
    print("Done! Ranking tables are in aggregation/new_results/")
    print("Figures are in figures/new_results/")

if __name__ == "__main__":
    main()
