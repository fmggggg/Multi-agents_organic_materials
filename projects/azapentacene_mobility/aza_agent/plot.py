"""Best-so-far plot for one or more campaigns: per-evaluation mu_lowestE (dots) and running maximum (line).
usage: python -m aza_agent.plot OUT.png DIR[:LABEL] [DIR[:LABEL] ...]   (linear y axis; AZA_PLOT_LOG=1 for log)"""
import os
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # validated default categorical order (dataviz palette)


def main():
    out = Path(sys.argv[1])
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
                         "font.size": 9})
    fig, ax = plt.subplots(figsize=(4.2, 3.0), dpi=200)
    for k, spec in enumerate(sys.argv[2:]):
        d, _, label = spec.partition(":")
        rows = list(csv.DictReader(open(Path(d) / "molecules.csv")))
        xs, ys = [], []
        for i, r in enumerate(rows, 1):
            if r["status"] == "evaluated":
                xs.append(i)
                ys.append(float(r["mu_lowestE"]))
        best, run = [], -1
        for y in ys:
            run = max(run, y)
            best.append(run)
        c = COLORS[k]
        ax.scatter(xs, ys, s=12, color=c, alpha=0.45, linewidths=0)
        ax.step(xs, best, where="post", color=c, lw=2, label=label or Path(d).name)
    if os.environ.get("AZA_PLOT_LOG") == "1":
        ax.set_yscale("log")
    else:
        ax.set_ylim(bottom=0)
    ax.set_xlabel("Molecules evaluated")
    ax.set_ylabel("MCRT mobility of lowest-energy crystal\n(cm$^2$ V$^{-1}$ s$^{-1}$)")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#e6e5e0", lw=0.6)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=8, handlelength=1.5)
    fig.tight_layout()
    fig.savefig(out)
    print(out)


if __name__ == "__main__":
    main()
