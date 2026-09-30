"""Bootstrap confidence intervals for a saved heldout scores.csv."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scores_csv", type=Path)
    parser.add_argument("--rounds", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    rows = list(csv.DictReader(args.scores_csv.open(encoding="utf-8")))
    labels = np.asarray([int(row["label"]) for row in rows])
    names = [key for key in rows[0] if key not in {"name", "label"}]
    rng = np.random.default_rng(args.seed)
    result = {}
    for name in names:
        values = np.asarray([float(row[name]) for row in rows])
        aucs, aps = [], []
        normal = np.flatnonzero(labels == 0)
        abnormal = np.flatnonzero(labels == 1)
        for _ in range(args.rounds):
            indices = np.concatenate((rng.choice(normal, len(normal), replace=True), rng.choice(abnormal, len(abnormal), replace=True)))
            sampled_labels = labels[indices]
            sampled_values = values[indices]
            aucs.append(roc_auc_score(sampled_labels, sampled_values))
            aps.append(average_precision_score(sampled_labels, sampled_values))
        result[name] = {
            "auroc": float(roc_auc_score(labels, values)),
            "auroc_ci95": [float(np.quantile(aucs, 0.025)), float(np.quantile(aucs, 0.975))],
            "average_precision": float(average_precision_score(labels, values)),
            "average_precision_ci95": [float(np.quantile(aps, 0.025)), float(np.quantile(aps, 0.975))],
        }
    output = args.scores_csv.with_name("bootstrap_ci.json")
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print("saved", output)


if __name__ == "__main__":
    main()
