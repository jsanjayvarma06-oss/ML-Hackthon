"""Feature selection evidence on the train table. OWNER: Sanjay.

Run from the repo root (after build_dataset.py):  python scripts/feature_report.py
Train only. Writes to results/pipeline/:
    feature_summary.csv     per feature: means by label, ROC AUC, mutual information,
                            Top-1 when candidates are ranked by that feature alone
    plots/feature_corr.png  correlation between features
    plots/feature_hist.png  distributions for correct vs wrong candidates
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.feature_selection import mutual_info_classif  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from pathsnap.config import PROCESSED_DIR, RESULTS_DIR, SEED  # noqa: E402
from pathsnap.features import FEATURE_COLUMNS  # noqa: E402
from pathsnap.metrics import UNLABELLED, top1_from_table  # noqa: E402

HIST_FEATURES = ["dist_m", "heading_diff_deg", "dist_gap_to_best", "dist_prev_m"]
# Same value for every candidate of a point, so they cannot rank candidates on their own;
# they only help in combination (e.g. heading matters less when speed is low).
POINT_LEVEL = {"speed_mps", "n_candidates", "heading_valid"}


def main():
    out = RESULTS_DIR / "pipeline"
    (out / "plots").mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    df = df[df["true_arc"] != UNLABELLED].reset_index(drop=True)
    y = df["label"].to_numpy()

    sample = df.sample(n=min(200_000, len(df)), random_state=SEED)
    mi = mutual_info_classif(sample[FEATURE_COLUMNS], sample["label"], random_state=SEED,
                             discrete_features=[c in ("dist_rank", "heading_valid", "n_candidates",
                                                      "has_reverse_twin") for c in FEATURE_COLUMNS])

    rows = []
    for col, mi_val in zip(FEATURE_COLUMNS, mi):
        x = df[col].to_numpy()
        auc = roc_auc_score(y, x)
        higher_is_better = auc >= 0.5
        rows.append({
            "feature": col,
            "mean_correct": float(x[y == 1].mean()),
            "mean_wrong": float(x[y == 0].mean()),
            "roc_auc": round(max(auc, 1 - auc), 4),
            "direction": "higher = more likely correct" if higher_is_better else "lower = more likely correct",
            "mutual_info": round(float(mi_val), 4),
            "top1_alone": (np.nan if col in POINT_LEVEL
                           else round(top1_from_table(df, x if higher_is_better else -x), 4)),
        })
    summary = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
    summary.to_csv(out / "feature_summary.csv", index=False)
    print(summary.to_string(index=False))

    corr = df[FEATURE_COLUMNS].corr()
    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(corr)), corr.columns, fontsize=8)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Feature correlation (train, labelled candidates)")
    fig.tight_layout()
    fig.savefig(out / "plots" / "feature_corr.png", dpi=130)

    fig, axes = plt.subplots(1, len(HIST_FEATURES), figsize=(4 * len(HIST_FEATURES), 3.2))
    for ax, col in zip(axes, HIST_FEATURES):
        hi = np.percentile(df[col], 99)
        bins = np.linspace(df[col].min(), hi, 50)
        ax.hist(df.loc[y == 0, col].clip(upper=hi), bins=bins, alpha=0.6, density=True, label="wrong arc")
        ax.hist(df.loc[y == 1, col].clip(upper=hi), bins=bins, alpha=0.6, density=True, label="correct arc")
        ax.set_title(col)
    axes[0].legend()
    fig.suptitle("Correct vs wrong candidates (train)")
    fig.tight_layout()
    fig.savefig(out / "plots" / "feature_hist.png", dpi=130)
    print(f"\nwrote {out / 'feature_summary.csv'} and plots")


if __name__ == "__main__":
    main()
