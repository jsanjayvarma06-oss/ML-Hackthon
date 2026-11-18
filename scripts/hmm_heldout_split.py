"""Held-out check of the HMM parameter choice (E5f). OWNER: Sanjay.

The HMM parameters in hmm_params.json are picked on the same out-of-fold train points they
are scored on, which flatters the number a little. This script measures how much: split the
80 train trips into two halves (even / odd position), pick the best (beta, emission weight)
on one half, and score that choice on the other half.

Input: results/pipeline/e5_log/e19_k16_per_trip.json, written by
scripts/experiments/e5/e19.py (K=16 candidates, OOF tuned LightGBM, floor 50, uturn 100):
{"(beta, w, floor, uturn)": [[n_correct, n_labelled] per trip, in trip order]}.
Output: results/pipeline/hmm_heldout_split.csv. Train data only; student_test is not used.
"""
import ast
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "results/pipeline/e5_log/e19_k16_per_trip.json"
OUT = ROOT / "results/pipeline/hmm_heldout_split.csv"


def top1(counts, idx):
    return sum(counts[i][0] for i in idx) / sum(counts[i][1] for i in idx)


def main():
    runs = {ast.literal_eval(k): v for k, v in json.loads(SRC.read_text()).items()}
    n = len(next(iter(runs.values())))
    halves = {"even": [i for i in range(n) if i % 2 == 0], "odd": [i for i in range(n) if i % 2 == 1]}
    rows = []
    for pick, score in [("even", "odd"), ("odd", "even")]:
        best = max(runs, key=lambda k: top1(runs[k], halves[pick]))
        rows.append({"chosen_on": pick, "scored_on": score, "beta": best[0], "emission_weight": best[1],
                     "floor": best[2], "uturn": best[3],
                     "top1_chosen_half": round(top1(runs[best], halves[pick]), 4),
                     "top1_held_out_half": round(top1(runs[best], halves[score]), 4)})
    best = max(runs, key=lambda k: top1(runs[k], range(n)))
    rows.append({"chosen_on": "all", "scored_on": "all", "beta": best[0], "emission_weight": best[1],
                 "floor": best[2], "uturn": best[3], "top1_chosen_half": round(top1(runs[best], range(n)), 4),
                 "top1_held_out_half": ""})
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
