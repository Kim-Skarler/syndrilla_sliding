"""Plot sliding-window LER vs p, one line per window config, with the run_sweep_v3 baseline overlaid.

Reads results_sw/BB_<n>_<schedule>/p_*/sw_result.yaml (written by run_sliding.py) and, for the
cross-check, results/BB_<n>_<schedule>/d_*/<algo>/p_*/result_phy_err_*.yaml (run_sweep_v3.py) when it exists.

Usage (repo root): python plot_sliding.py --code 144
"""
import argparse
import glob
import os
import re

import matplotlib.pyplot as plt
import yaml

from plot_results_v3 import fast_extract_lers

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
# BP stage of the sliding-window decoder -> run_sweep_v3 algo folder
SWEEP_ALGO = {"bp_norm_min_sum": "bposd", "bp_lottery": "lottery_bposd"}

VARIANT_LS = {"merge": (0, (6, 2)), "prior": (0, (1, 1)), "random": (0, (3, 1, 1, 1)),
              "merge+prior": (0, (6, 1, 1, 1)), "merge+random": (0, (6, 1, 1, 1, 1, 1))}
STYLES = {  # label: (color, marker, linestyle)
    "baseline": ("black", "o", "-"),
    "SW(1)": ("#d62728", "v", "--"),
    "SW(2)": ("#ff7f0e", "^", "--"),
    "SW(3)": ("#2ca02c", "s", "--"),
    "SW(4)": ("#1f77b4", "D", "--"),
    "SW(3,1)": ("#9467bd", "v", ":"),
    "SW(3,2)": ("#8c564b", "^", ":"),
    "SW(4,2)": ("#e377c2", "s", ":"),
}


def load_sw(base, code, schedule):
    data = {}  # label -> {p: ler}
    algo = None
    for path in glob.glob(os.path.join(base, f"BB_{code}_{schedule}", "p_*", "sw_result.yaml")):
        with open(path, encoding="utf-8") as f:
            res = yaml.safe_load(f)
        p = res["meta"].get("p")
        p = float(p) if p is not None else float(re.search(r"p_([0-9.]+)", path).group(1).rstrip("."))
        algo = SWEEP_ALGO.get(res["meta"]["algorithm"][0], res["meta"]["algorithm"][0])
        for label, r in res["configs"].items():
            if r["errors"] > 0:
                data.setdefault(label, {})[p] = r["ler"]
    return data, algo


def load_sweep_baseline(base, code, algo, schedule):
    pts = {}
    for path in glob.glob(os.path.join(base, f"BB_{code}_{schedule}", "d_*", algo, "p_*", "result_phy_err_*.yaml")):
        m = re.search(r"p_([0-9.]+)", path)
        ler_osd, _ = fast_extract_lers(path)
        if m and ler_osd:
            pts[float(m.group(1).rstrip("."))] = ler_osd
    return pts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--code", default="144")
    ap.add_argument("--algo", default=None, help="sweep algo for the overlay (default: from sw_result meta)")
    ap.add_argument("--sw-dir", default=os.path.join(REPO_ROOT, "results_sw"))
    ap.add_argument("--sweep-dir", default=os.path.join(REPO_ROOT, "results"))
    ap.add_argument("-o", "--output", default=None, help="default: plots/sliding_BB_<n>_<schedule>.png")
    ap.add_argument("--schedule", default="7layer_x", help="BB circuit schedule of the run (default: 7layer_x)")
    args = ap.parse_args()

    data, algo = load_sw(args.sw_dir, args.code, args.schedule)
    if not data:
        print(f"[SKIP] no sliding-window results for BB_{args.code} in {args.sw_dir}")
        return
    algo = args.algo or algo
    sweep = load_sweep_baseline(args.sweep_dir, args.code, algo, args.schedule)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    def style(label):
        # "SW(2)+merge+prior": color/marker from the window, line style from the variant
        base, _, variant = label.partition("+")
        color, marker, ls = STYLES.get(base, ("gray", "x", "-."))
        return color, marker, VARIANT_LS.get(variant, ls)

    def order(label):
        base = label.partition("+")[0]
        return (base != "baseline", list(STYLES).index(base) if base in STYLES else 99, label)

    for label in sorted(data, key=order):
        pts = data[label]
        color, marker, ls = style(label)
        ps = sorted(pts)
        ax.plot(ps, [pts[p] for p in ps], color=color, marker=marker, ls=ls, lw=1.5, markersize=6,
                markerfacecolor="none", label=label)
    if sweep:
        ps = sorted(sweep)
        ax.plot(ps, [sweep[p] for p in ps], color="gray", marker="x", ls="none", markersize=9,
                label=f"run_sweep_v3 {algo} (cross-check)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Physical Error Rate ($p$)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Logical Error Rate ($p_L$)", fontsize=12, fontweight="bold")
    ax.set_title(f"BB [[{args.code}]] {args.schedule} - DEM sliding window ({algo})", fontsize=13, fontweight="bold")
    ax.tick_params(direction="in", which="both", top=True, right=True)
    ax.grid(True, which="major", ls="-", color="gray", alpha=0.35)
    ax.grid(True, which="minor", ls=":", color="gray", alpha=0.2)
    ax.legend(title="Window (T_BP, T_OSD)", loc="lower right", framealpha=0.9)

    out = args.output or os.path.join(REPO_ROOT, "plots", f"sliding_BB_{args.code}_{args.schedule}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out)
    print(f"[SAVED] {out}")


if __name__ == "__main__":
    main()
