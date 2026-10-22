#!/usr/bin/env python3
"""
plot_results_v3.py - High-Quality QEC Threshold Plotter for Syndrilla v3 Results

Features:
1. Automatically scans results/ directory for result_phy_err_*.yaml
2. Separates plots for Surface code and BB codes
3. Same distance -> Same color
4. Different decoder -> Different marker and linestyle
5. Dual legend design (Code distance legend + Decoder style legend)
6. Log-Log scale with clean academic aesthetic (300 DPI)
"""

import os
import re
import yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# 1. 顏色映射 (同一個 distance 固定顏色)
SURFACE_COLORS = {
    3: "#1f77b4",   # 藍
    9: "#ff7f0e",   # 橘
    15: "#2ca02c",  # 綠
    21: "#d62728",  # 紅
}

BB_COLORS = {
    "BB_72": "#1f77b4",    # d=6
    "BB_144": "#ff7f0e",   # d=12
    "BB_288": "#2ca02c",   # d=18
    "BB_360": "#d62728",   # d=24
    "72": "#1f77b4",
    "144": "#ff7f0e",
    "288": "#2ca02c",
    "360": "#d62728",
    "[[72,12,6]]": "#1f77b4",
    "[[144,12,12]]": "#ff7f0e",
    "[[288,12,18]]": "#2ca02c",
    "[[784,24,24]]": "#d62728"
}

# 2. 解碼器樣式 (Decoder 不同 Marker / 線型)
DECODER_STYLES_BPOSD = {
    "bposd": {"marker": "o", "ls": "-", "label": "BP + OSD-0"},
    "lottery_bposd": {"marker": "s", "ls": "--", "label": "Lottery BP + OSD-0 (PRE)"},
    "bp_norm_min_sum": {"marker": "o", "ls": "-", "label": "BP + OSD-0"},
    "bp_lottery": {"marker": "s", "ls": "--", "label": "Lottery BP + OSD-0"},
    "relay_bp": {"marker": "^", "ls": ":", "label": "Relay BP"},
}

DECODER_STYLES_PURE_BP = {
    "bposd": {"marker": "^", "ls": "-", "label": "BP (NMS)"},
    "lottery_bposd": {"marker": "o", "ls": "--", "label": "Lottery BP"},
    "bp_norm_min_sum": {"marker": "^", "ls": "-", "label": "BP (NMS)"},
    "bp_lottery": {"marker": "o", "ls": "--", "label": "Lottery BP"},
    "relay_bp": {"marker": "s", "ls": ":", "label": "Relay BP"},
}

def fast_extract_lers(filepath: str):
    """
    極速串流解析 YAML (單檔 < 0.5ms，避免 yaml.load 反序列化龐大整數陣列導致 CPU 凍結)
    回傳: (ler_osd, ler_bp)
    """
    ler_osd = None
    ler_bp = None
    current_dec = None
    in_hx = False
    last_dec_num = -1
    last_dec_ler = None

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("decoder_"):
                    m = re.match(r"^(decoder_\w+):", line)
                    if m:
                        current_dec = m.group(1)
                        in_hx = False
                elif line.startswith("  hx:"):
                    in_hx = True
                elif in_hx and "logical error rate:" in line:
                    val_str = line.strip().split(":", 1)[1].strip()
                    val = float(val_str)
                    if current_dec == "decoder_0":
                        ler_bp = val
                    if current_dec == "decoder_full":
                        ler_osd = val
                    elif current_dec and current_dec.startswith("decoder_"):
                        dec_suffix = current_dec[8:]
                        if dec_suffix.isdigit():
                            num = int(dec_suffix)
                            if num > last_dec_num:
                                last_dec_num = num
                                last_dec_ler = val
                elif not line.startswith(" "):
                    current_dec = None
                    in_hx = False
    except Exception:
        pass

    if ler_osd is None:
        ler_osd = last_dec_ler
    return ler_osd, ler_bp

def collect_data(results_dir: str):
    """
    掃描 results 目錄 (只收集 STIM DEM 結果)
    單次遍歷同時擷取 BPOSD 與純 BP 數據。
    BB 依電路版本分組: 資料夾 BB_<n>_ibm -> "ibm"，BB_<n> -> "7layer"
    回傳: (surface_osd, bb_osd), (surface_bp, bb_bp)，其中 bb_* 為 {schedule: {code: {algo: {p: ler}}}}
    """
    surface_osd = {}
    bb_osd = {}
    surface_bp = {}
    bb_bp = {}

    if not os.path.exists(results_dir):
        print(f"[WARN] Results directory '{results_dir}' does not exist.")
        return (surface_osd, bb_osd), (surface_bp, bb_bp)

    for root, dirs, files in os.walk(results_dir):
        for f in files:
            if f.startswith("result_phy_err_") and f.endswith(".yaml"):
                rel_path = os.path.relpath(os.path.join(root, f), results_dir)
                rel_parts = rel_path.split(os.sep)
                if len(rel_parts) < 3:
                    continue

                # 排除 Space-Time Matrix (如包含 SHOT 等)
                if any("shot" in p.lower() for p in rel_parts):
                    continue

                code_tag = rel_parts[0]

                # 動態搜尋 p 與 algo
                p_idx = -1
                for i, part in enumerate(rel_parts):
                    if re.search(r"p_([0-9\.]+)", part):
                        p_idx = i
                        break

                if p_idx != -1:
                    p_str = rel_parts[p_idx]
                    algo = rel_parts[p_idx - 1] if p_idx > 0 else "unknown"
                else:
                    p_str = f
                    algo = rel_parts[-2] if len(rel_parts) >= 2 else "unknown"

                m_p = re.search(r"p_([0-9\.]+)", p_str) or re.search(r"result_phy_err_([0-9\.]+)", f)
                if not m_p:
                    continue
                p_err = float(m_p.group(1).rstrip('.'))

                # 搜尋 distance
                dist = None
                for part in rel_parts:
                    m_d = re.search(r"d_(\d+)", part)
                    if m_d:
                        dist = int(m_d.group(1))
                        break

                full_path = os.path.join(root, f)
                ler_osd, ler_bp = fast_extract_lers(full_path)

                if "surface" in code_tag.lower():
                    dist = dist if dist is not None else 3
                    if ler_osd is not None:
                        surface_osd.setdefault(dist, {}).setdefault(algo, {})[p_err] = ler_osd
                    if ler_bp is not None:
                        surface_bp.setdefault(dist, {}).setdefault(algo, {})[p_err] = ler_bp
                else:
                    schedule = "7layer"
                    if code_tag.endswith("_ibm"):
                        schedule = "ibm"
                        code_tag = code_tag[:-len("_ibm")]
                    m_bb = re.search(r'(\[\[.+\]\])', code_tag)
                    clean_code = m_bb.group(1) if m_bb else code_tag
                    if ler_osd is not None:
                        bb_osd.setdefault(schedule, {}).setdefault(clean_code, {}).setdefault(algo, {})[p_err] = ler_osd
                    if ler_bp is not None:
                        bb_bp.setdefault(schedule, {}).setdefault(clean_code, {}).setdefault(algo, {})[p_err] = ler_bp

    return (surface_osd, bb_osd), (surface_bp, bb_bp)

def plot_curve(data: dict, is_surface: bool = True, output_path: str = "ler_curve.png", pure_bp: bool = False, title_suffix: str = ""):
    """
    繪製 LER 圖表 (高解析度、雙圖例、Log-Log 座標)
    """
    if not data:
        print(f"[SKIP] No data found for {'Surface' if is_surface else 'BB'} code ({'Pure BP' if pure_bp else 'BPOSD'}).")
        return

    plt.figure(figsize=(8, 6), dpi=300)
    ax = plt.gca()

    all_p = set()
    used_dists = []
    used_algos = set()

    sorted_keys = sorted(
        data.keys(),
        key=lambda x: int(x) if isinstance(x, int) else (int(re.search(r"\d+", x).group(0)) if re.search(r"\d+", x) else str(x))
    )

    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    for idx, key in enumerate(sorted_keys):
        used_dists.append(key)
        if is_surface:
            color = SURFACE_COLORS.get(key, palette[idx % len(palette)])
        else:
            color = BB_COLORS.get(key, palette[idx % len(palette)])

        algos_dict = data[key]
        for algo, pts in algos_dict.items():
            used_algos.add(algo)
            styles_dict = DECODER_STYLES_PURE_BP if pure_bp else DECODER_STYLES_BPOSD
            style = styles_dict.get(algo, {"marker": "o", "ls": "-", "label": algo})
            
            sorted_p = sorted(pts.keys())
            sorted_ler = [pts[p] for p in sorted_p]
            all_p.update(sorted_p)

            ax.plot(
                sorted_p, sorted_ler,
                color=color,
                marker=style["marker"],
                markersize=6,
                linestyle=style["ls"],
                linewidth=1.5,
                markeredgecolor=color,
                markerfacecolor="none",
                zorder=3
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Physical Error Rate ($p$)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Logical Error Rate ($p_L$)", fontsize=12, fontweight="bold")
    ax.set_title(
        f"{'Surface Code (Rotated Memory X)' if is_surface else 'Bivariate Bicycle (BB) Codes'}{title_suffix} - STIM DEM{' (Pure BP)' if pure_bp else ''}",
        fontsize=14, fontweight="bold", pad=12
    )

    ax.tick_params(direction="in", length=6, width=1.0, which="major", top=True, right=True)
    ax.tick_params(direction="in", length=3, width=0.8, which="minor", top=True, right=True)
    ax.grid(True, which="major", ls="-", color="gray", alpha=0.35, zorder=0)
    ax.grid(True, which="minor", ls=":", color="gray", alpha=0.2, zorder=0)

    # 圖例 1: Distance (顏色)
    dist_handles = []
    for idx, key in enumerate(used_dists):
        color = SURFACE_COLORS.get(key, palette[idx % len(palette)]) if is_surface else BB_COLORS.get(key, palette[idx % len(palette)])
        label = f"d = {key}" if is_surface else f"{key}"
        dist_handles.append(Line2D([0], [0], color=color, marker="s", markersize=6, linestyle="none", label=label))

    # 圖例 2: Decoder (Marker / 線型)
    algo_handles = []
    styles_dict = DECODER_STYLES_PURE_BP if pure_bp else DECODER_STYLES_BPOSD
    for algo in sorted(list(used_algos)):
        style = styles_dict.get(algo, {"marker": "o", "ls": "-", "label": algo})
        algo_handles.append(Line2D([0], [0], color="black", marker=style["marker"], markersize=6, linestyle=style["ls"], markerfacecolor="none", label=style["label"]))

    leg1 = ax.legend(handles=dist_handles, title="Code Size / Distance", loc="lower right", framealpha=0.9, edgecolor="gray")
    ax.add_artist(leg1)
    ax.legend(handles=algo_handles, title="Decoder", loc="upper left", framealpha=0.9, edgecolor="gray")

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"\033[92m[SAVED]\033[0m Plot saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Plot LER for Syndrilla v3 (STIM DEM)")
    parser.add_argument("-r", "--results-dir", type=str, default="results", help="Results directory (default: results)")
    parser.add_argument("-o", "--output-dir", type=str, default="plots", help="Plots output directory (default: plots)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 單次掃描，同時收集 OSD 與純 BP
    (surface_osd, bb_osd), (surface_bp, bb_bp) = collect_data(args.results_dir)

    # BB 依電路版本分開畫: 7layer -> bb_code_*.png (原檔名)，ibm -> bb_code_ibm_*.png
    bb_variants = {"7layer": ("", ""), "ibm": ("_ibm", " [IBM circuit]")}

    # 1. 繪製 BPOSD vs LotteryOSD
    if surface_osd:
        plot_curve(surface_osd, is_surface=True, output_path=os.path.join(args.output_dir, "surface_code_ler.png"), pure_bp=False)
    for schedule, (file_tag, title_suffix) in bb_variants.items():
        if bb_osd.get(schedule):
            plot_curve(bb_osd[schedule], is_surface=False, output_path=os.path.join(args.output_dir, f"bb_code{file_tag}_ler.png"),
                       pure_bp=False, title_suffix=title_suffix)

    # 2. 繪製 純 BP 比較 (不加 OSD)
    if surface_bp:
        plot_curve(surface_bp, is_surface=True, output_path=os.path.join(args.output_dir, "surface_code_pure_bp_ler.png"), pure_bp=True)
    for schedule, (file_tag, title_suffix) in bb_variants.items():
        if bb_bp.get(schedule):
            plot_curve(bb_bp[schedule], is_surface=False, output_path=os.path.join(args.output_dir, f"bb_code{file_tag}_pure_bp_ler.png"),
                       pure_bp=True, title_suffix=title_suffix)

if __name__ == "__main__":
    main()
