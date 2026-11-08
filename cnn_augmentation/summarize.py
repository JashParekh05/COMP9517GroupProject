"""Aggregate results/*.json into tables and figures for the report.

Example:
    python summarize.py --split val
    python summarize.py --split test --models baseline_s0 geo_photo_s0 full_s0 full_seen_s0

Outputs (in --out, default figures/):
    summary_<split>.csv / .md     one row per model: clean mAP, mean corrupted mAP,
                                  relative robustness, per-corruption mean, train/infer time
    severity_<split>.png          small multiples: mAP@0.5 vs severity, one panel per corruption
    heatmap_<split>.png           models x corruptions, mean mAP@0.5 over severities

If several seeds of a config exist (e.g. full_s0, full_s1), add --group-seeds to report
mean +- std over seeds.
"""
import argparse
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from experiments import SEEN

# Reference palette (light mode, for print). Fixed order: a model keeps its colour
# across every figure because colour is assigned by position in --models, not by rank.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 9, "axes.titlesize": 10, "legend.frameon": False,
    })


def load(results_dir, split):
    out = {}
    for p in sorted(glob.glob(os.path.join(results_dir, f"*_{split}.json"))):
        with open(p) as f:
            r = json.load(f)
        out[r["name"]] = r
    return out


def group_seeds(res):
    """Merge runs named <cfg>_s<k>[...] into one entry with mean/std over seeds."""
    groups = {}
    for name, r in res.items():
        key = re.sub(r"_s\d+", "", name)
        groups.setdefault(key, []).append(r)
    merged = {}
    for key, rs in groups.items():
        m = {"name": key, "n_seeds": len(rs), "clean": {"ap": float(np.mean([r["clean"]["ap"] for r in rs]))},
             "clean_std": float(np.std([r["clean"]["ap"] for r in rs])),
             "infer_ms_per_img": float(np.mean([r["infer_ms_per_img"] for r in rs])),
             "train_time_s": np.mean([r["train_time_s"] for r in rs if r.get("train_time_s")] or [np.nan]),
             "corrupted": {}}
        if "mean_corrupted_ap" in rs[0]:
            for c in rs[0]["corrupted"]:
                m["corrupted"][c] = {s: float(np.mean([r["corrupted"][c][s] for r in rs]))
                                     for s in rs[0]["corrupted"][c] if s.isdigit()}
            m["mean_ap_per_corruption"] = {c: float(np.mean([r["mean_ap_per_corruption"][c] for r in rs]))
                                           for c in rs[0]["mean_ap_per_corruption"]}
            mc = [r["mean_corrupted_ap"] for r in rs]
            m["mean_corrupted_ap"], m["mean_corrupted_std"] = float(np.mean(mc)), float(np.std(mc))
            m["relative_robustness"] = (m["mean_corrupted_ap"] / m["clean"]["ap"]
                                        if m["clean"]["ap"] >= 0.01 else None)
        merged[key] = m
    return merged


def table(res, models, out_base):
    corrs = list(res[models[0]].get("mean_ap_per_corruption", {}))
    unseen = [c for c in corrs if c not in SEEN]
    head = ["model", "clean mAP", "mean corr. mAP", "unseen-only mAP", "rel. robustness",
            *corrs, "train min", "infer ms/img"]
    rows = []
    for m in models:
        r = res[m]
        pc = r.get("mean_ap_per_corruption", {})
        fmt = lambda v, sd=None: "-" if v is None else (f"{v:.3f}" if sd is None else f"{v:.3f}±{sd:.3f}")
        tt = r.get("train_time_s")
        rows.append([m, fmt(r["clean"]["ap"], r.get("clean_std")),
                     fmt(r.get("mean_corrupted_ap"), r.get("mean_corrupted_std")),
                     fmt(float(np.mean([pc[c] for c in unseen])) if pc else None),
                     fmt(r.get("relative_robustness")),
                     *[fmt(pc.get(c)) for c in corrs],
                     "-" if tt is None or np.isnan(tt) else f"{tt / 60:.1f}",
                     f"{r['infer_ms_per_img']:.1f}"])
    with open(out_base + ".csv", "w") as f:
        f.write(",".join(head) + "\n")
        for row in rows:
            f.write(",".join(row) + "\n")
    with open(out_base + ".md", "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(head) + " |\n|" + "---|" * len(head) + "\n")
        for row in rows:
            f.write("| " + " | ".join(row) + " |\n")
    print(open(out_base + ".md", encoding="utf-8").read())
    print(f"'unseen-only' = mean over corruptions NOT used by the *_seen configs: {unseen}")


def severity_plot(res, models, path):
    corrs = list(res[models[0]]["corrupted"])
    ncol = 4
    nrow = int(np.ceil(len(corrs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(10, 2.4 * nrow + 0.6), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, c in zip(axes, corrs):
        for k, m in enumerate(models):
            sev = sorted(int(s) for s in res[m]["corrupted"][c] if s.isdigit())
            ys = [res[m]["clean"]["ap"]] + [res[m]["corrupted"][c][str(s)] for s in sev]
            ax.plot([0] + sev, ys, color=SERIES[k], lw=2, marker="o", ms=4,
                    mec=SURFACE, mew=1, label=m, zorder=3)
        ax.set_title(c.replace("_", " ") + (" *" if c not in SEEN else ""), loc="left")
        ax.set_xticks(range(0, 6))
        ax.set_ylim(bottom=0)
    for ax in axes[len(corrs):]:
        ax.axis("off")
    for ax in axes[::ncol]:
        ax.set_ylabel("mAP@0.5")
    for ax in axes[-ncol:]:
        ax.set_xlabel("severity (0 = clean)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(len(models), 4),
               bbox_to_anchor=(0.5, 1.0), labelcolor=INK2)
    fig.text(0.01, 0.005, "* held out: never seen in training by the *_seen configs",
             fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def heatmap(res, models, path):
    corrs = list(res[models[0]]["mean_ap_per_corruption"])
    cols = ["clean"] + corrs + ["mean corr."]
    M = np.array([[res[m]["clean"]["ap"]] + [res[m]["mean_ap_per_corruption"][c] for c in corrs]
                  + [res[m]["mean_corrupted_ap"]] for m in models])
    fig, ax = plt.subplots(figsize=(1.0 + 0.85 * len(cols), 0.9 + 0.38 * len(models)))
    vmax = max(M.max(), 1e-6)
    ax.imshow(M, cmap=BLUES, vmin=0, vmax=vmax, aspect="auto")
    ax.grid(False)
    ax.set_xticks(range(len(cols)), [c.replace("_", "\n") for c in cols], fontsize=8)
    ax.set_yticks(range(len(models)), models, fontsize=8, color=INK2)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7.5,
                    color="#ffffff" if M[i, j] > 0.55 * vmax else INK)
    # thin separators around the "clean" and "mean" columns
    for x in (0.5, len(cols) - 1.5):
        ax.axvline(x, color=SURFACE, lw=3)
    ax.set_title("mAP@0.5 (corruptions: mean over severities 1-5)", loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--models", nargs="*", help="which runs and in what order (max 8 for plots)")
    ap.add_argument("--group-seeds", action="store_true")
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()

    res = load(args.results, args.split)
    if args.group_seeds:
        res = group_seeds(res)
    if not res:
        raise SystemExit(f"no results/*_{args.split}.json found - run evaluate.py first")
    models = args.models or sorted(res)
    missing = [m for m in models if m not in res]
    if missing:
        raise SystemExit(f"unknown models {missing}; available: {sorted(res)}")
    os.makedirs(args.out, exist_ok=True)
    style()

    table(res, models, os.path.join(args.out, f"summary_{args.split}"))
    has_corr = all(res[m].get("corrupted") for m in models)
    if has_corr:
        plot_models = models[:len(SERIES)]
        if len(models) > len(SERIES):
            print(f"note: plotting only the first {len(SERIES)} models; pass --models to choose")
        severity_plot(res, plot_models, os.path.join(args.out, f"severity_{args.split}.png"))
        heatmap(res, models, os.path.join(args.out, f"heatmap_{args.split}.png"))
        print(f"figures written to {args.out}/")


if __name__ == "__main__":
    main()
