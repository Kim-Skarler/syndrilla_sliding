"""Plot sliding-window LER vs p, one line per window config, with the run_sweep_v3 baseline overlaid.

Reads results_sw/BB_<n>_<schedule>/p_*/sw_result.yaml (written by run_sliding.py) and, for the
cross-check, results/BB_<n>_<schedule>/d_*/<algo>/p_*/result_phy_err_*.yaml (run_sweep_v3.py) when it exists.

By default only baseline and the flushed windows are drawn, unmerged solid and merged dashed;
--all draws every config.

--quant draws only the fixed-point figure (SW(5,1) of lottery_quant_bposd_i*f* against its format).

--ranking draws only the OSD ranking figure (SW(5,1) of lottery_quant_bposd_i*f*[_<ranking>] against p).

Usage (repo root): python plot_sliding.py --code 144 [--all | --quant | --ranking]
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
              "merge+prior": (0, (6, 1, 1, 1)), "merge+random": (0, (6, 1, 1, 1, 1, 1)),
              "flush": (0, (2, 2)), "merge+flush": (0, (8, 2, 2, 2))}
STYLES = {  # label: (color, marker, linestyle)
    "baseline": ("black", "o", "-"),
    "SW(1)": ("#d62728", "v", "--"),
    "SW(2)": ("#ff7f0e", "^", "--"),
    "SW(3)": ("#2ca02c", "s", "--"),
    "SW(4)": ("#1f77b4", "D", "--"),
    "SW(5)": ("#17becf", "P", "--"),
    "SW(6)": ("#7f7f7f", "X", "--"),
    "SW(3,1)": ("#9467bd", "v", ":"),
    "SW(3,2)": ("#8c564b", "^", ":"),
    "SW(4,2)": ("#e377c2", "s", ":"),
    "SW(4,1)": ("#bcbd22", "<", ":"),
    "SW(5,2)": ("#9467bd", ">", ":"),
    "SW(5,3)": ("#8c564b", "h", ":"),
}


def flush_view(data):
    """baseline plus the flushed configs in DEM column order, keyed without "+flush"; solid
    lines unmerged, dashed merged. SW(1) has no "+flush" label since flushing it changes
    nothing, so its plain labels stand in. An unflushed SW(T_BP,T_OSD) with T_OSD < T_BP
    is kept too, marked "round": its tail keeps the OSD at T_OSD rounds, which the flush
    does not. Round configs are dotted in DEM column order, dash-dot for "prior" and
    densely dotted for "random"."""
    out = {}
    order_ls = {"prior": "-.", "random": (0, (1, 1))}
    for label, pts in data.items():
        base, *variant = label.split("+")
        if base == "baseline":
            if not variant:
                out[label] = (pts, "-")
            continue
        flush = "flush" in variant
        variant = [v for v in variant if v != "flush"]
        sizes = [int(x) for x in base[3:-1].split(",")]
        name = "+".join([base] + variant)
        if flush or base == "SW(1)":
            if variant in ([], ["merge"]):
                out[name] = (pts, "--" if variant else "-")
        elif len(sizes) == 2 and sizes[1] < sizes[0] and "merge" not in variant:
            out[name + " round"] = (pts, order_ls.get(variant[0], ":") if variant else ":")
    return out


def load_sw(base, code, schedule):
    """{algo: {label: {p: ler}}} from results_sw/BB_<n>_<schedule>/<algo>/p_*/ (and the older
    layout without the algo folder)."""
    data = {}
    root = os.path.join(base, f"BB_{code}_{schedule}")
    paths = glob.glob(os.path.join(root, "*", "p_*", "sw_result.yaml")) + \
        glob.glob(os.path.join(root, "p_*", "sw_result.yaml"))
    for path in paths:
        with open(path, encoding="utf-8") as f:
            res = yaml.safe_load(f) or {}
        if "meta" not in res:
            continue
        p = res["meta"].get("p")
        p = float(p) if p is not None else float(re.search(r"p_([0-9.]+)", path).group(1).rstrip("."))
        # the algo folder tells lottery_bposd from lottery_nopre_bposd (both run bp_lottery)
        folder = os.path.basename(os.path.dirname(os.path.dirname(path)))
        algo = folder if os.path.dirname(os.path.dirname(path)) != root else \
            SWEEP_ALGO.get(res["meta"]["algorithm"][0], res["meta"]["algorithm"][0])
        for label, r in res["configs"].items():
            if r["errors"] > 0:
                data.setdefault(algo, {}).setdefault(label, {})[p] = r["ler"]
    return data


def plot_quant(args, label="SW(5,1)", float_algo="lottery_bposd", quant_algo="lottery_quant_bposd"):
    """LER of <label> against the fixed-point format, one panel per p: x = frac bits, one line
    per int width (error bar sqrt(errors) / shots), with the float <label> and float baseline
    of <float_algo> as horizontal lines."""
    root = os.path.join(args.sw_dir, f"BB_{args.code}_{args.schedule}")
    pts = {}  # p -> {(iw, fw) or "float" or "baseline": (ler, err)}
    for path in glob.glob(os.path.join(root, "*", "p_*", "sw_result.yaml")):
        folder = os.path.basename(os.path.dirname(os.path.dirname(path)))
        m = re.fullmatch(rf"{quant_algo}_i(\d+)f(\d+)", folder)
        if not m and folder != float_algo:
            continue
        with open(path, encoding="utf-8") as f:
            res = yaml.safe_load(f) or {}
        p = float(res["meta"]["p"])
        keys = [((int(m.group(1)), int(m.group(2))), label)] if m else [("float", label), ("baseline", "baseline")]
        for key, lab in keys:
            r = res.get("configs", {}).get(lab)
            if r and r["errors"] > 0:
                pts.setdefault(p, {})[key] = (r["ler"], r["errors"] ** 0.5 / r["shots"])
    ps = sorted(p for p, v in pts.items() if any(isinstance(k, tuple) for k in v))
    if not ps:
        print(f"[SKIP] no {quant_algo}_i*f* results in {root}")
        return
    fig, axes = plt.subplots(1, len(ps), figsize=(4.2 * len(ps), 4.2), dpi=300, squeeze=False)
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#d62728"]
    for ax, p in zip(axes[0], ps):
        v = pts[p]
        iws = sorted({k[0] for k in v if isinstance(k, tuple)})
        for ci, iw in enumerate(iws):
            fws = sorted(k[1] for k in v if isinstance(k, tuple) and k[0] == iw)
            ax.errorbar(fws, [v[(iw, f)][0] for f in fws], yerr=[v[(iw, f)][1] for f in fws],
                        color=colors[ci % len(colors)], marker="o", ms=5, lw=1.5, capsize=3,
                        label=f"int {iw} bits")
        for key, ls, name in (("float", "--", f"float {label}"), ("baseline", ":", "float baseline")):
            if key in v:
                ax.axhline(v[key][0], color="black", ls=ls, lw=1.2, label=name)
        ax.set_yscale("log")
        ax.set_xlabel("fraction bits (+1 sign, + int bits)", fontsize=10)
        ax.set_title(f"p = {p}", fontsize=11, fontweight="bold")
        ax.tick_params(direction="in", which="both", top=True, right=True)
        ax.grid(True, which="major", ls="-", color="gray", alpha=0.35)
        all_fws = sorted({k[1] for k in v if isinstance(k, tuple)})
        ax.set_xticks(all_fws)
        ax.set_xlim(all_fws[0] - 0.5, all_fws[-1] + 0.5)
    axes[0][0].set_ylabel("Logical Error Rate ($p_L$)", fontsize=11, fontweight="bold")
    axes[0][-1].legend(fontsize=8, framealpha=0.9)
    fig.suptitle(f"BB [[{args.code}]] {args.schedule}\n{label} round, fixed-point lottery+PRE BP",
                 fontsize=11, fontweight="bold")
    out = os.path.join(args.output_dir, f"sliding_BB_{args.code}_{args.schedule}_quant.png")
    os.makedirs(args.output_dir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"[SAVED] {out}")


def plot_ranking(args, label="SW(5,1)", float_algo="lottery_bposd", quant_algo="lottery_quant_bposd"):
    """LER of <label> against p for every fixed-point format and OSD ranking found
    (lottery_quant_bposd_i<int>f<frac>[_<ranking>]): color by ranking, line style by format,
    with the float <label> of <float_algo> in black."""
    root = os.path.join(args.sw_dir, f"BB_{args.code}_{args.schedule}")
    lines = {}  # (width, ranking) -> {p: (ler, err)}
    for path in glob.glob(os.path.join(root, "*", "p_*", "sw_result.yaml")):
        folder = os.path.basename(os.path.dirname(os.path.dirname(path)))
        # only the known rankings, so backup folders such as *_hardware_lzc_old are left out
        m = re.fullmatch(rf"{quant_algo}_i(\d+)f(\d+)(?:_(hardware_user|hardware_lzc))?", folder)
        if not m and folder != float_algo:
            continue
        with open(path, encoding="utf-8") as f:
            res = yaml.safe_load(f) or {}
        r = res.get("configs", {}).get(label)
        if not r or r["errors"] == 0:
            continue
        key = (f"Q{m.group(1)}.{m.group(2)}", m.group(3) or "standard") if m else ("float", "standard")
        lines.setdefault(key, {})[float(res["meta"]["p"])] = (r["ler"], r["errors"] ** 0.5 / r["shots"])
    if not any(k[1] != "standard" for k in lines):
        print(f"[SKIP] no {quant_algo}_i*f*_<ranking> results in {root}")
        return
    # only the formats run with a hardware ranking (the format sweep has standard-only ones)
    ranked = {k[0] for k in lines if k[1] != "standard"}
    lines = {k: v for k, v in lines.items() if k[0] == "float" or k[0] in ranked}
    colors = {"standard": "#2a78d6", "hardware_user": "#eb6834", "hardware_lzc": "#1baf7a"}
    widths = sorted({k[0] for k in lines if k[0] != "float"})
    styles = ["-", "--", ":", "-."]
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    for key in sorted(lines, key=lambda k: (k[0] != "float", k[0], list(colors).index(k[1]))):
        pts = lines[key]
        ps = sorted(pts)
        if key[0] == "float":
            kw = dict(color="black", ls="-", marker="o", label=f"float {label}")
        else:
            kw = dict(color=colors.get(key[1], "gray"), ls=styles[widths.index(key[0]) % len(styles)],
                      marker="os^"[widths.index(key[0]) % 3], markerfacecolor="none", label=f"{key[0]} {key[1]}")
        ax.errorbar(ps, [pts[p][0] for p in ps], yerr=[pts[p][1] for p in ps], lw=1.5, ms=6, capsize=3, **kw)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Physical Error Rate ($p$)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Logical Error Rate ($p_L$)", fontsize=12, fontweight="bold")
    ax.set_title(f"BB [[{args.code}]] {args.schedule} - {label} round\nfixed-point lottery+PRE BP, OSD column ranking",
                 fontsize=12, fontweight="bold")
    ax.tick_params(direction="in", which="both", top=True, right=True)
    ax.grid(True, which="major", ls="-", color="gray", alpha=0.35)
    ax.grid(True, which="minor", ls=":", color="gray", alpha=0.2)
    ax.legend(title="BP format, OSD ranking", loc="lower right", fontsize=9, framealpha=0.9)
    out = os.path.join(args.output_dir, f"sliding_BB_{args.code}_{args.schedule}_ranking.png")
    os.makedirs(args.output_dir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"[SAVED] {out}")


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
    ap.add_argument("--algo", default=None, help="comma separated algos to plot (default: every algo found)")
    ap.add_argument("--sw-dir", default=os.path.join(REPO_ROOT, "results_sw"))
    ap.add_argument("--sweep-dir", default=os.path.join(REPO_ROOT, "results"))
    ap.add_argument("-o", "--output-dir", default=os.path.join(REPO_ROOT, "plots"),
                    help="writes sliding_BB_<n>_<schedule>_<algo>.png here")
    ap.add_argument("--schedule", default="7layer_x", help="BB circuit schedule of the run (default: 7layer_x)")
    ap.add_argument("--all", action="store_true",
                    help="every config; default is baseline plus SW(T)+flush with and without merge")
    ap.add_argument("--quant", action="store_true",
                    help="only the fixed-point figure: SW(5,1) LER against int/frac bits of lottery_quant_bposd")
    ap.add_argument("--ranking", action="store_true",
                    help="only the OSD ranking figure: SW(5,1) LER against p per format and ranking")
    args = ap.parse_args()

    if args.quant:
        plot_quant(args)
        return
    if args.ranking:
        plot_ranking(args)
        return
    all_data = load_sw(args.sw_dir, args.code, args.schedule)
    algos = args.algo.split(",") if args.algo else sorted(all_data)
    if not any(all_data.get(a) for a in algos):
        print(f"[SKIP] no sliding-window results for BB_{args.code}_{args.schedule} in {args.sw_dir}")
        return
    for algo in algos:
        if all_data.get(algo):
            plot_one(args, algo, all_data[algo])


def plot_one(args, algo, data):
    # run_sweep_v3 points of the same circuit, when they exist (none for X-only schedules)
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

    lines = {label: (pts, style(label)[2]) for label, pts in data.items()} if args.all else flush_view(data)
    for label in sorted(lines, key=order):
        pts, ls = lines[label]
        color, marker, _ = style(label)
        ps = sorted(pts)
        ax.plot(ps, [pts[p] for p in ps], color=color, marker=marker, ls=ls, lw=1.5, markersize=6,
                markerfacecolor="none" if "merge" in label else color, label=label)
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
    ax.legend(title="Window (T_BP, T_OSD)" if args.all else "Window, flushed (dashed: merge)",
              loc="lower right", framealpha=0.9)

    suffix = "_all" if args.all else ""
    out = os.path.join(args.output_dir, f"sliding_BB_{args.code}_{args.schedule}_{algo}{suffix}.png")
    os.makedirs(args.output_dir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"[SAVED] {out}")


if __name__ == "__main__":
    main()
