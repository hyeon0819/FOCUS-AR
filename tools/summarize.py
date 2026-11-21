"""Collect results.json files of finished runs into the rows of Tables 2-4."""

import argparse
import json
import os

import numpy as np


def load(path):
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def base2novel(root):
    rows = {}
    for dataset in ["k400", "hmdb51", "ucf101"]:
        runs = [load(f"{root}/base2novel/{dataset}_split{s}/results.json") for s in (1, 2, 3)]
        runs = [r for r in runs if r]
        if runs:
            rows[dataset] = {k: np.mean([r[k] for r in runs]) for k in ("base", "novel", "hm")}
            print(f"{dataset:7s} ({len(runs)} splits)  Base {rows[dataset]['base']:.1f}  "
                  f"Novel {rows[dataset]['novel']:.1f}  HM {rows[dataset]['hm']:.1f}")
    if len(rows) == 3:
        avg = {k: np.mean([r[k] for r in rows.values()]) for k in ("base", "novel", "hm")}
        print(f"{'average':7s}             Base {avg['base']:.1f}  Novel {avg['novel']:.1f}  HM {avg['hm']:.1f}")


def fewshot(root):
    accs = []
    for dataset in ["hmdb51", "ucf101"]:
        row = []
        for shot in (2, 4, 8, 16):
            r = load(f"{root}/fewshot/{dataset}_k{shot}/results.json")
            row.append(r["top1"] if r else float("nan"))
        accs += row
        print(f"{dataset:7s} " + "  ".join(f"K={k}: {a:.1f}" for k, a in zip((2, 4, 8, 16), row)))
    print(f"average {np.mean(accs):.1f}")


def zeroshot(root):
    r = load(f"{root}/zeroshot/k400/results.json")
    if r:
        for dataset in ["hmdb51", "ucf101", "k600"]:
            print(f"{dataset:7s} {r[dataset]['mean']:.1f} +/- {r[dataset]['std']:.1f}")
        print(f"average {r['average']:.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["base2novel", "fewshot", "zeroshot"], required=True)
    parser.add_argument("--work-dirs", default="work_dirs")
    args = parser.parse_args()
    globals()[args.task](args.work_dirs)
