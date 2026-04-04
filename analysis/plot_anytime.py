import csv
import os
import sys
import matplotlib.pyplot as plt

def plot_anytime(csv_path: str):
    trials = []
    scores = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            trials.append(int(row["trial_id"]))
            scores.append(float(row["score"]))

    best_so_far = []
    cur_best = None
    for s in scores:
        if cur_best is None or s < cur_best:
            cur_best = s
        best_so_far.append(cur_best)

    out_png = csv_path.replace(".csv", ".png")

    plt.figure()
    plt.plot(trials, best_so_far)
    plt.xlabel("trial")
    plt.ylabel("best loss so far (lower is better)")
    plt.title(os.path.basename(csv_path))
    plt.tight_layout()
    plt.savefig(out_png)
    print("SAVED:", out_png)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analysis/plot_anytime.py results/<csv_file>")
        sys.exit(1)
    plot_anytime(sys.argv[1])