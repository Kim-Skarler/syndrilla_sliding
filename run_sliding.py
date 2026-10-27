"""Sweep runner for the DEM sliding window (syndrilla.main_sliding), one task per (code, p).

Each point writes interface.yaml (BB circuit of --bb-schedule) and decoding.yaml, then runs
`python -m syndrilla.main_sliding`, which decodes every window config on the same samples
and writes results_sw/BB_<n>_<schedule>/<algo>/p_<p>/sw_result.yaml (resumed when it already exists).

GOLDEN (or --golden) decodes one batch of GOLDEN_BATCH samples per point and writes the OSD
hardware test patterns to GOLDEN_DIR instead, in a <algo>_p<p> subfolder when there are several
points (one window config, fixed-point algo only).

Usage (repo root):
  python run_sliding.py --codes 144 --rates 0.001,0.002,0.003,0.005 --gpus 0,1 --dry-run
  python run_sliding.py --golden --windows 5,1 --final-commit round --algo lottery_quant_bposd --quant-widths 4.3 --osd-ranking hardware_lzc --rates 0.003
"""
import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tools.bb_circuit_generator import generate_bb_interface_yaml, resolve_bb_prefix  # noqa: E402
from syndrilla.main_sliding import config_label, parse_configs  # noqa: E402
from syndrilla.decoder.osd_0.ranking import RANKING_METHODS  # noqa: E402

DEFAULT_CONFIG = {
    "CODES": ["144"],
    "ERROR_RATES": [0.01, 0.005, 0.003],
    # BB 電路: "7layer_x" (7layer 的 Stim 風格電路與雜訊，只宣告 X detectors，binary 解 Z 錯誤)
    # 或 "ibm" (IBM decoder_setup.py 的電路與雜訊，只宣告 X detectors)
    "BB_SCHEDULE": "7layer_x",

    # ---- 要跑哪些解碼設定 (全部用同一批 sample) ----
    # 完整 DEM 一次解碼 (baseline)
    "RUN_BASELINE": False,
    # sliding window 大小: T 代表 SW(T) (BP、OSD 視窗相同)，(T_BP, T_OSD) 代表 BP 視窗大於 OSD 視窗
    "SW_WINDOWS": [(5, 1)], # 1, 2, 3, 4, (3, 1), (3, 2), (4, 2)
    # sliding window 是否做視窗內合併: [False] 只跑原始，[True] 只跑合併，[False, True] 兩種都跑
    "MERGE": [False],
    # 視窗內 column 順序 (OSD 的 tie-break): "dem" = Stim 原本順序 (不排序)；"prior"、"random" 之後再測
    "COL_ORDER": ["dem"],
    # 結尾處理 (window 都固定 T_BP 輪): 
    # "round" = 滑到最後、每個 window commit 一輪 (等同 padding 全 0 syndrome + 飽和 LLR)；
    # "all" = 最後一個完整 window 一次 commit 剩下全部 (label 加 "+flush")
    "FINAL_COMMIT": ["round"],

    # 解碼器 (可放多個，各自一個 task，結果存到 .../<algo>/p_<p>/):
    # "bposd": bp_norm_min_sum + osd_0 
    # "lottery_bposd": bp_lottery (預設 lottery + PRE) + osd_0,
    # "lottery_nopre_bposd": bp_lottery 只跑 lottery (flip_mode: lottery，不含 PRE) + osd_0
    # "lottery_quant_bposd": bp_lottery_quant (lottery + PRE, fixed point) + osd_0，
    #     每個 QUANT_WIDTHS 各自一個 task，結果存到 .../lottery_quant_bposd_i<int>f<frac>/p_<p>/
    "ALGO": ["lottery_quant_bposd"],
    # fixed point 格式 (int_width, frac_width)，另有 1 個 sign bit；只用在 *_quant algo
    "QUANT_WIDTHS": [(3, 2), (3, 3), (3, 4), (4, 2), (4, 3), (4, 4), (5, 2), (5, 3), (5, 4)],
    # OSD column 排序 (只用在 *_quant algo，用 BP 的 fixed point 格式):
    # "standard" = LLR 由小到大；"hardware_user" = bit-serial 找最小值的粗分級；"hardware_lzc" = |LLR| 位元長度分級
    # 非 standard 的結果存到 .../lottery_quant_bposd_i<int>f<frac>_<ranking>/p_<p>/
    "OSD_RANKING": ["standard"],
    "BATCH_SIZE": 5000,
    # logical errors per config; low p is expensive, so it gets fewer
    "TARGET_ERROR": {0.001: 8, 0.003: 50, 0.005: 200, 0.01: 500, "default": 100},
    "BASE_RESULT_DIR": "results_sw",

    # ---- golden 模式: 給 OSD 硬體 testbench 的測資 ----
    # True = 每個點只解一個 batch (GOLDEN_BATCH 個 sample) 就結束，輸出 OSD 測資到 GOLDEN_DIR，不碰 sw_result.yaml
    # (只能一個 window 設定 + *_quant algo；多個點時各自放在 GOLDEN_DIR/<algo>_p<p>/)
    # python run_sliding.py --golden --force-pytorch --windows 5,1 --final-commit round --algo lottery_quant_bposd --quant-widths 4.3 --osd-ranking hardware_lzc --rates 0.003
    "GOLDEN": False,
    "GOLDEN_BATCH": 100,
    # 最後一個 window (t=12, OSD H 只有 144 欄) 的測資: False = 跳過；
    # True = 嵌進 bulk 的 720 欄，其餘欄位 invalid (需要硬體 sorter 支援 column valid 遮罩)
    "GOLDEN_TAIL": False,
    "GOLDEN_DIR": r"C:\master\QEC\OSD_hardware\testbench\patterns_txts",
}

BP_ALGO = {"bposd": "bp_norm_min_sum", "lottery_bposd": "bp_lottery", "lottery_nopre_bposd": "bp_lottery",
           "lottery_quant_bposd": "bp_lottery_quant"}
# extra keys for the BP stage; bp_norm_min_sum rejects lottery keys, so they are set per algo
BP_EXTRA_CFG = {"lottery_nopre_bposd": {"flip_mode": "lottery"}}


def resolve_target_error(spec, rate):
    if isinstance(spec, dict):
        return int(spec.get(rate, spec.get("default", 100)))
    if isinstance(spec, str) and ":" in spec:
        items = dict(item.split(":") for item in spec.split(","))
        return resolve_target_error({(k if k == "default" else float(k)): int(v) for k, v in items.items()}, rate)
    return int(spec)


def algo_dir(algo, width, ranking="standard"):
    """Result folder of an algo; a fixed-point algo gets its format and a non-standard OSD
    ranking, e.g. lottery_quant_bposd_i4f3 or lottery_quant_bposd_i4f3_hardware_lzc."""
    if width is None:
        return algo
    return f"{algo}_i{width[0]}f{width[1]}" + ("" if ranking == "standard" else f"_{ranking}")


def parse_widths(spec):
    """'3.4,4.3' -> [(3, 4), (4, 3)]."""
    return [tuple(int(x) for x in w.strip().split(".")) for w in spec.split(",") if w.strip()]


def build_decoding_config(algo, max_iter, use_cpu, force_pytorch=False, width=None, ranking="standard"):
    dec = {
        "algorithm": [BP_ALGO[algo], "osd_0"],
        "check_type": "hx",
        "dtype": "float64",
        "device": {"device_type": "cpu" if use_cpu else "cuda", "device_idx": 0},
        "config": {"max_iter": int(max_iter), **BP_EXTRA_CFG.get(algo, {})},
    }
    if width is not None:
        dec["config"].update({"int_width": int(width[0]), "frac_width": int(width[1])})
    if ranking != "standard":
        # the OSD stage grades the BP's fixed-point LLR in the same format
        dec["config"] = [dec["config"], {"ranking_method": ranking, "int_width": int(width[0]),
                                         "frac_width": int(width[1])}]
    if force_pytorch:
        # torch modules on the GPU instead of the JIT-compiled CUDA kernels (no nvcc/MSVC needed)
        dec["force_pytorch"] = True
    return {"decoding": dec}


def is_finished(point_dir, target_error, max_shots, labels):
    """True when every wanted config is already in sw_result.yaml and has reached the target."""
    res_file = os.path.join(point_dir, "sw_result.yaml")
    if not os.path.exists(res_file):
        return False
    try:
        with open(res_file, encoding="utf-8") as f:
            cfgs = (yaml.safe_load(f) or {}).get("configs", {})
    except Exception:
        return False
    if any(label not in cfgs for label in labels):
        return False
    wanted = [cfgs[label] for label in labels]
    if min(c["errors"] for c in wanted) >= target_error:
        return True
    return max_shots > 0 and min(c["shots"] for c in wanted) >= max_shots


def worker_task(task, gpu_queue, args):
    code, rate, algo, width, ranking, target_error = task
    _, bb_key, distance = resolve_bb_prefix(code)
    algo = algo_dir(algo, width, ranking)
    point_dir = os.path.join(args.result_dir, f"BB_{code}_{args.bb_schedule}", algo, f"p_{rate}")
    labels = [config_label(*c) for c in parse_configs(args.windows, args.merge_window, args.col_order,
                                                       args.final_commit)]
    if not args.golden and is_finished(point_dir, target_error, args.max_shots, labels):
        print(f"\033[93m[SKIP]\033[0m BB_{code} {algo} p={rate} already finished in {point_dir}.")
        return

    interface_yaml = os.path.join(point_dir, "interface.yaml")
    decoding_yaml = os.path.join(point_dir, "decoding.yaml")
    # max_iter = n, the BP-family study convention
    max_iter = args.max_iter or int(bb_key)
    if not args.dry_run:
        os.makedirs(point_dir, exist_ok=True)
        generate_bb_interface_yaml(code_type=code, error_rate=rate, rounds=distance,
                                   output_yaml_path=interface_yaml, schedule=args.bb_schedule)
        with open(decoding_yaml, "w", encoding="utf-8") as f:
            yaml.safe_dump(build_decoding_config(task[2], max_iter, args.cpu, args.force_pytorch, width, ranking), f, sort_keys=False)

    cmd = [sys.executable, "-m", "syndrilla.main_sliding", f"-r={point_dir}", f"-i={interface_yaml}",
           f"-d={decoding_yaml}", f"-w={args.windows}", f"-bs={args.batch_size}", f"-te={target_error}",
           f"-ms={args.max_shots}", f"-p={rate}", f"-mw={args.merge_window}", f"-co={args.col_order}",
           f"-fc={args.final_commit}"]
    if args.golden:
        # one batch, written as OSD test patterns; the result file is left alone
        cmd[cmd.index(f"-bs={args.batch_size}")] = f"-bs={args.golden_batch}"
        out = args.golden_dir if args.golden_points == 1 else os.path.join(args.golden_dir, f"{algo}_p{rate}")
        cmd.append(f"--golden={out}")
        if args.golden_tail:
            cmd.append("--golden_tail")
    if args.cpu:
        cmd.append("--cpu")
    if args.dry_run:
        print("\033[96m[DRY-RUN]\033[0m " + " ".join(f'"{c}"' if ";" in c else c for c in cmd))
        return

    gpu = gpu_queue.get()
    try:
        env = os.environ.copy()
        if not args.cpu:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        t0 = time.time()
        print(f"\033[92m[START]\033[0m [GPU {gpu}] BB_{code} {algo} p={rate} te={target_error} -> {point_dir}", flush=True)
        with open(os.path.join(point_dir, "runner_exec.log"), "a", encoding="utf-8") as log:
            res = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        if res.returncode == 0:
            print(f"\033[92m[DONE]\033[0m BB_{code} {algo} p={rate} ({time.time() - t0:.0f}s)", flush=True)
        else:
            print(f"\033[91m[FAILED]\033[0m BB_{code} {algo} p={rate} code={res.returncode}, see {point_dir}/runner_exec.log",
                  flush=True)
    finally:
        gpu_queue.put(gpu)


def parse_args():
    ap = argparse.ArgumentParser(description="DEM sliding-window sweep runner")
    ap.add_argument("--codes", default=None, help="comma separated BB codes, e.g. '144,72'")
    ap.add_argument("--rates", default=None, help="comma separated physical error rates")
    ap.add_argument("--bb-schedule", default=DEFAULT_CONFIG["BB_SCHEDULE"], choices=["7layer_x", "ibm"],
                    help="BB circuit, X detectors only (default: 7layer_x)")
    # leave these unset to use RUN_BASELINE / SW_WINDOWS / MERGE / COL_ORDER above
    ap.add_argument("--windows", default=None, help="override, e.g. '0;1;2;3,1' (0 = baseline)")
    ap.add_argument("--merge-window", default=None, help="override: off, on, or off,on")
    ap.add_argument("--col-order", default=None, help="override, comma separated: dem, prior, random")
    ap.add_argument("--final-commit", default=None, help="override: round, all, or round,all")
    ap.add_argument("--force-pytorch", action="store_true", help="torch modules instead of the CUDA kernels")
    ap.add_argument("--algo", default=None, help="override ALGO, comma separated: bposd, lottery_bposd, "
                    "lottery_nopre_bposd, lottery_quant_bposd")
    ap.add_argument("--osd-ranking", default=None, help="override OSD_RANKING, comma separated: "
                    "standard, hardware_user, hardware_lzc")
    ap.add_argument("--quant-widths", default=None, help="override QUANT_WIDTHS, 'int.frac' comma separated, e.g. '3.4,4.3'")
    ap.add_argument("--max-iter", type=int, default=None, help="default: code length n")
    ap.add_argument("-bs", "--batch-size", type=int, default=DEFAULT_CONFIG["BATCH_SIZE"])
    ap.add_argument("-te", "--target-error", default=None, help="'100' or '0.001:20,0.003:100,default:50'")
    ap.add_argument("--max-shots", type=int, default=0, help="0 = no limit")
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--workers-per-gpu", type=int, default=1)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("-r", "--result-dir", default=DEFAULT_CONFIG["BASE_RESULT_DIR"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--golden", action="store_true", default=DEFAULT_CONFIG["GOLDEN"],
                    help="one batch per point, written as OSD hardware test patterns to --golden-dir")
    ap.add_argument("--golden-batch", type=int, default=DEFAULT_CONFIG["GOLDEN_BATCH"],
                    help="samples of the golden batch")
    ap.add_argument("--golden-tail", action="store_true", default=DEFAULT_CONFIG["GOLDEN_TAIL"],
                    help="golden: also write the tail window, embedded with invalid columns")
    ap.add_argument("--golden-dir", default=DEFAULT_CONFIG["GOLDEN_DIR"],
                    help="golden output; a <algo>_p<p> subfolder per point when there are several")
    return ap.parse_args()


def windows_from_config(cfg):
    """RUN_BASELINE + SW_WINDOWS -> main_sliding's '-w' string, e.g. '0;1;2;3,1'."""
    items = ["0"] if cfg["RUN_BASELINE"] else []
    for w in cfg["SW_WINDOWS"]:
        items.append(f"{w[0]},{w[1]}" if isinstance(w, (tuple, list)) else str(w))
    if not items:
        raise ValueError("nothing to run: RUN_BASELINE is False and SW_WINDOWS is empty")
    return ";".join(items)


def main():
    args = parse_args()
    if args.windows is None:
        args.windows = windows_from_config(DEFAULT_CONFIG)
    if args.merge_window is None:
        args.merge_window = ",".join("on" if m else "off" for m in DEFAULT_CONFIG["MERGE"])
    if args.col_order is None:
        args.col_order = ",".join(DEFAULT_CONFIG["COL_ORDER"])
    if args.final_commit is None:
        args.final_commit = ",".join(DEFAULT_CONFIG["FINAL_COMMIT"])
    configs = parse_configs(args.windows, args.merge_window, args.col_order, args.final_commit)
    codes = [c.strip() for c in args.codes.split(",")] if args.codes else DEFAULT_CONFIG["CODES"]
    rates = [float(r) for r in args.rates.split(",")] if args.rates else DEFAULT_CONFIG["ERROR_RATES"]
    te_spec = args.target_error if args.target_error is not None else DEFAULT_CONFIG["TARGET_ERROR"]
    algo_cfg = [a.strip() for a in args.algo.split(",")] if args.algo else DEFAULT_CONFIG["ALGO"]
    algos = [algo_cfg] if isinstance(algo_cfg, str) else list(algo_cfg)
    for a in algos:
        if a not in BP_ALGO:
            raise ValueError(f"unknown algo <{a}>, choose from {sorted(BP_ALGO)}")
    widths = parse_widths(args.quant_widths) if args.quant_widths else [tuple(w) for w in DEFAULT_CONFIG["QUANT_WIDTHS"]]
    rankings = [m.strip() for m in args.osd_ranking.split(",")] if args.osd_ranking else DEFAULT_CONFIG["OSD_RANKING"]
    for m in rankings:
        if m not in RANKING_METHODS:
            raise ValueError(f"unknown OSD ranking <{m}>, choose from {RANKING_METHODS}")
    tasks = [(c, r, a, w, m, resolve_target_error(te_spec, r)) for c in codes for a in algos
             for w in (widths if a.endswith("_quant_bposd") else [None])
             for m in (rankings if a.endswith("_quant_bposd") else ["standard"]) for r in rates]
    # cheap high-p points first, so every config has its quick points before the slow low-p ones
    tasks.sort(key=lambda t: -t[1])
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()] or ["0"]

    print("=" * 70)
    print("  SYNDRILLA DEM SLIDING WINDOW SWEEP")
    print(f"  Total points: {len(tasks)}")
    print(f"  Codes:        {codes}")
    print(f"  Error Rates:  {rates}")
    print(f"  Windows:      {args.windows}  (merge: {args.merge_window}, col order: {args.col_order}, "
          f"final commit: {args.final_commit})")
    print(f"  Configs:      {len(configs)} -> {', '.join(config_label(*c) for c in configs)}")
    print(f"  Algorithms:   {algos}")
    if any(a.endswith("_quant_bposd") for a in algos):
        print(f"  Quant widths: {', '.join(f'Q{i}.{f}' for i, f in widths)} (+1 sign bit)")
        print(f"  OSD ranking:  {rankings}")
    args.golden_points = len(tasks)
    if args.golden:
        if len(configs) != 1:
            raise ValueError(f"golden mode takes one window config, got {[config_label(*c) for c in configs]}")
        print(f"  GOLDEN MODE:  one batch of {args.golden_batch} samples per point -> {args.golden_dir}"
              + ("" if len(tasks) == 1 else "/<algo>_p<p>"))
    print(f"  BB schedule:  {args.bb_schedule}")
    print(f"  GPUs:         {gpus} (workers/gpu: {args.workers_per_gpu})")
    print("=" * 70)

    gpu_queue = Queue()
    for g in gpus:
        for _ in range(args.workers_per_gpu):
            gpu_queue.put(g)
    workers = 1 if args.dry_run else max(1, len(gpus) * args.workers_per_gpu)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for f in [ex.submit(worker_task, t, gpu_queue, args) for t in tasks]:
            f.result()


if __name__ == "__main__":
    main()
