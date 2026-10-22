#!/usr/bin/env python3
"""
plot_compare_schedules.py - Compare BB circuit schedules (7layer vs ibm vs stim) from run_sweep_v3 results

- Scans results/BB_<n>_<schedule>/d_*/<algo>/p_*/result_phy_err_*.yaml (a folder without a
  schedule suffix counts as 7layer)
- One figure per BB code, one panel per decoder, one line per schedule, with binomial 1-sigma
  error bars (low p has few logical errors)
- Prints a table: LER, errors/shots, and the ratio to the reference schedule (default 7layer)

Usage (repo root):
  python plot_compare_schedules.py
  python plot_compare_schedules.py --codes 144 --algos bposd,lottery_bposd --per-round
  python plot_compare_schedules.py --pure-bp
"""

import argparse
import glob
import math
import os
import re

import matplotlib.pyplot as plt

from plot_results_v3 import BB_SCHEDULES, fast_extract_lers

SCHEDULE_STYLES = {
    "7layer": {"color": "#1f77b4", "marker": "o", "label": "7layer (Stim-style, original CNOT order)"},
    "stim": {"color": "#ff7f0e", "marker": "s", "label": "stim (Stim-style, IBM CNOT order)"},
    "ibm": {"color": "#2ca02c", "marker": "^", "label": "ibm (IBM noise, X detectors only)"},
}
ALGO_TITLES = {"bposd": "BP + OSD-0", "lottery_bposd": "Lottery BP + OSD-0", "relay_bp": "Relay BP"}


def read_counts(path):
    """(shots, logical errors) of decoder_full in a result YAML."""
    with open(path, encoding="utf-8") as f:
        txt = f.read()
    block = txt.split("decoder_full:", 1)[1] if "decoder_full:" in txt else ""
    bs = re.search(r"batch size: (\d+)", block)
    bc = re.search(r"batch count: (\d+)", block)
    te = re.search(r"target error reached: (\d+)", block)
    if not (bs and bc and te):
        return None, None
    return int(bs.group(1)) * int(bc.group(1)), int(te.group(1))


def collect(results_dir):
    """data[code][algo][schedule][p] = dict(ler, ler_bp, shots, errors, rounds)"""
    data = {}
    for path in glob.glob(os.path.join(results_dir, "BB_*", "d_*", "*", "p_*", "result_phy_err_*.yaml")):
        parts = os.path.relpath(path, results_dir).replace("\\", "/").split("/")
        tag, d_dir, algo, p_dir = parts[0], parts[1], parts[2], parts[3]
        schedule, code = "7layer", tag[len("BB_"):]
        for s in BB_SCHEDULES:
            if code.endswith(f"_{s}"):
                schedule, code = s, code[:-len(s) - 1]
                break
        p = float(p_dir[len("p_"):].rstrip("."))
        ler, ler_bp = fast_extract_lers(path)
        shots, errors = read_counts(path)
        if ler is None:
            continue
        data.setdefault(code, {}).setdefault(algo, {}).setdefault(schedule, {})[p] = {
            "ler": ler, "ler_bp": ler_bp, "shots": shots, "errors": errors,
            "rounds": int(re.search(r"\d+", d_dir).group(0)),
        }
    return data


def per_round(ler, rounds):
    return 1 - (1 - ler) ** (1 / rounds) if ler < 1 else 1.0


def point(entry, pure_bp, by_round):
    """(value, 1-sigma) for one result point."""
    val = entry["ler_bp"] if pure_bp else entry["ler"]
    if val is None:
        return None, None
    sigma = None
    if not pure_bp and entry["shots"]:
        sigma = math.sqrt(max(val * (1 - val), 0) / entry["shots"])
    if by_round:
        lo = per_round(max(val - (sigma or 0), 0), entry["rounds"])
        val_r = per_round(val, entry["rounds"])
        sigma = (val_r - lo) if sigma is not None else None
        val = val_r
    return val, sigma


def print_table(data, codes, algos, schedules, ref):
    print("=" * 100)
    print(f"LER per shot (logical errors / shots); ratio = schedule / {ref}")
    for code in codes:
        for algo in algos:
            sched_pts = data.get(code, {}).get(algo, {})
            ps = sorted({p for s in schedules for p in sched_pts.get(s, {})})
            if not ps:
                continue
            print(f"\n[BB {code}] {algo}")
            print(f"{'p':>8s} | " + " | ".join(f"{s:>26s}" for s in schedules))
            for p in ps:
                cells = []
                ref_e = sched_pts.get(ref, {}).get(p)
                for s in schedules:
                    e = sched_pts.get(s, {}).get(p)
                    if not e:
                        cells.append(f"{'-':>26s}")
                        continue
                    cnt = f"({e['errors']}/{e['shots']})" if e["shots"] else ""
                    ratio = f" x{e['ler'] / ref_e['ler']:.2f}" if ref_e and ref_e["ler"] and s != ref else ""
                    cells.append(f"{e['ler']:9.2e} {cnt:>11s}{ratio:>6s}")
                print(f"{p:>8g} | " + " | ".join(cells))
    print("=" * 100)
    print("Points with < ~20 logical errors are noisy; compare them with their error bars.")


def plot_code(data, code, algos, schedules, pure_bp, by_round, out_path):
    algos = [a for a in algos if a in data.get(code, {})]
    if not algos:
        print(f"[SKIP] no results for BB {code}")
        return
    fig, axes = plt.subplots(1, len(algos), figsize=(6.5 * len(algos), 5.5), dpi=200, sharey=True, squeeze=False)
    for ax, algo in zip(axes[0], algos):
        for s in schedules:
            pts = data[code][algo].get(s, {})
            xs, ys, es = [], [], []
            for p in sorted(pts):
                v, sig = point(pts[p], pure_bp, by_round)
                if v is None or v <= 0:
                    continue
                xs.append(p); ys.append(v); es.append(sig if sig is not None else 0)
            if not xs:
                continue
            st = SCHEDULE_STYLES.get(s, {"color": "gray", "marker": "x", "label": s})
            # keep the lower error bar above zero on a log axis
            lower = [min(e, y * 0.999) for y, e in zip(ys, es)]
            ax.errorbar(xs, ys, yerr=[lower, es], color=st["color"], marker=st["marker"], markersize=6,
                        markerfacecolor="none", lw=1.5, capsize=3, label=st["label"])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(ALGO_TITLES.get(algo, algo) + (" - BP only" if pure_bp else ""), fontsize=12, fontweight="bold")
        ax.set_xlabel("Physical Error Rate ($p$)", fontsize=11)
        ax.tick_params(direction="in", which="both", top=True, right=True)
        ax.grid(True, which="major", ls="-", color="gray", alpha=0.35)
        ax.grid(True, which="minor", ls=":", color="gray", alpha=0.2)
        ax.legend(fontsize=8, loc="lower right", framealpha=0.9)
    axes[0][0].set_ylabel("Logical Error Rate per round" if by_round else "Logical Error Rate ($p_L$)", fontsize=11)
    fig.suptitle(f"BB [[{code}]] - circuit schedule comparison (STIM DEM)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"\033[92m[SAVED]\033[0m {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Compare BB circuit schedules (7layer / ibm / stim)")
    ap.add_argument("-r", "--results-dir", default="results")
    ap.add_argument("-o", "--output-dir", default="plots")
    ap.add_argument("--codes", default=None, help="comma separated, e.g. '72,144' (default: all found)")
    ap.add_argument("--algos", default="bposd,lottery_bposd")
    ap.add_argument("--schedules", default="7layer,stim,ibm")
    ap.add_argument("--ref", default="7layer", help="schedule the table ratios are relative to")
    ap.add_argument("--pure-bp", action="store_true", help="plot BP-only LER (decoder_0) instead of BP+OSD")
    ap.add_argument("--per-round", action="store_true", help="plot 1-(1-p_L)^(1/rounds)")
    args = ap.parse_args()

    data = collect(args.results_dir)
    codes = args.codes.split(",") if args.codes else sorted(data, key=lambda c: int(re.search(r"\d+", c).group(0)))
    algos = args.algos.split(",")
    schedules = args.schedules.split(",")

    print_table(data, codes, algos, schedules, args.ref)
    suffix = ("_pure_bp" if args.pure_bp else "") + ("_per_round" if args.per_round else "")
    for code in codes:
        plot_code(data, code, algos, schedules, args.pure_bp, args.per_round,
                  os.path.join(args.output_dir, f"compare_BB_{code}_schedules{suffix}.png"))


if __name__ == "__main__":
    main()
