#!/usr/bin/env python3
"""
run_sweep_v3.py - High-Performance Multi-GPU Automated Sweep Runner for Syndrilla

Features:
1. Multi-GPU dynamic dispatch (Queue-based worker pool across specified GPU IDs).
2. Dynamic Stim Interface generation for BB codes (72, 144, 288, 784) and Surface codes.
3. Dynamic Decoding YAML generation (bp_lottery, bp_norm_min_sum, bposd, etc.).
4. Smart Checkpoint & Skip: seamlessly skips points already finished or resumes in-progress points.
5. Clean logging: per-point main.log and aggregated console status.
"""

import os
import sys
import math
import time
import argparse
import subprocess
import itertools
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
import yaml

# Import local BB circuit generator
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tools.bb_circuit_generator import (
    generate_bb_interface_yaml,
    generate_surface_interface_yaml,
    resolve_bb_prefix
)

# =====================================================================
# 1. 預設參數網格 (Default Parameter Grid)
# 可直接在此修改，亦可透過命令行參數 --codes, --rates, --algos 覆蓋
# =====================================================================
DEFAULT_CONFIG = {
    # 支援: "surface", "72", "144", "288", "360"
    "CODES": ["288", "360"],
    
    # 針對 Surface code 的 distance 列表 (BB code 會自動依尺寸對應預設 distance)
    "SURFACE_DISTANCES": [3, 9, 15, 21], # 3, 9, 15, 21
    
    # 物理錯誤率掃描列表
    "ERROR_RATES": [0.01, 0.005, 0.003, 0.001], # 0.01, 0.005, 0.003, 0.001
    
    # 解碼演算法清單:
    # - "bposd": 標準 Normalized Min-Sum + OSD-0
    # - "lottery_bposd": 新版 Lottery BP (內建 16x5 輪 PRE) + OSD-0，固定排程 flip_phase_iters=40, pre_max_lambda=16, pre_max_iter=5
    # - "lottery_bposd_new_arrangement": 同 lottery_bposd，排程依 bb_lottery_pre_setup.md，
    #   fpi 與 pmi 依 (d, p, max_iter) 推導 (只支援 BB)
    # - "relay_bp": Relay BP-S (純 BP 多 leg 擾動，不含 OSD，依碼別動態注入參數)
    # - "bp" / "lottery_bp" / "lottery_bp_new_arrangement": 對應 bposd / lottery_bposd / lottery_bposd_new_arrangement
    #   的 BP 階段，不接 OSD-0；畫圖時併入對應演算法的 Pure BP 曲線
    "BP_ALGOS": [
        # "bp",
        # "lottery_bp_new_arrangement"
        "bposd",
        # "lottery_bposd",
        "lottery_bposd_new_arrangement"
        # "relay_bp"
    ],

    # 雜訊模型: "CIRCUIT_LEVEL" 或 "PHENOMENOLOGICAL"
    "NOISE_MODEL": "CIRCUIT_LEVEL",

    # BB 電路排程:
    # - "ibm":    與 IBM decoder_setup.py 相同的 depth-8 cycle 與雜訊模型，只宣告 X detectors
    # - "7layer": 參考 Stim generated circuit 的結構與雜訊 (X+Z detectors)，原本的 CNOT 順序
    # - "stim":   同 7layer，但換成 IBM 的 CNOT 順序 (circuit distance 較高)
    # 結果存到 BB_<n>_<schedule>/
    "BB_SCHEDULE": "7layer",

    # 預設執行參數
    "BATCH_SIZE": 2500,
    # TARGET_ERROR 支援單一整數 (全域共用) 或 dict (依物理錯誤率分級設置)
    "TARGET_ERROR": {
        0.01: 200,
        0.005: 100,
        0.003: 20,
        0.001: 5,
        0.0001: 10,
        "default": 100
    },
    # MAX_ITER 支援 "auto" (自動依碼長 n 設置: BB 72->72, 144->144, 360->360; Surface->2d(d-1)+1) 或指定整數
    "MAX_ITER": "auto",
    "BASE_RESULT_DIR": "results"
}

def parse_args():
    parser = argparse.ArgumentParser(description="Distributed Sweep Runner for Syndrilla v3")
    parser.add_argument("--gpus", type=str, default="0", help="Comma-separated GPU IDs to use, e.g. '0,1,2,3' or '0'")
    parser.add_argument("--workers-per-gpu", type=int, default=1, help="Number of concurrent workers per GPU (default: 1)")
    parser.add_argument("--codes", type=str, default=None, help="Comma-separated code types (e.g. '144,72' or 'surface')")
    parser.add_argument("--distances", type=str, default=None, help="Comma-separated distances for surface code (e.g. '3,5,9')")
    parser.add_argument("--rates", type=str, default=None, help="Comma-separated physical error rates (e.g. '0.01,0.005,0.002')")
    parser.add_argument("--algos", type=str, default=None, help="Comma-separated decoding algorithms (e.g. 'bp_lottery,bp_norm_min_sum')")
    parser.add_argument("-bs", "--batch-size", type=int, default=DEFAULT_CONFIG["BATCH_SIZE"], help="Batch size per step")
    parser.add_argument("-te", "--target-error", type=str, default=None, help="Target logical error count (e.g. '100' or '0.01:500,0.005:200,default:50')")
    parser.add_argument("--max-iter", type=str, default=str(DEFAULT_CONFIG["MAX_ITER"]), help="Max BP iterations (default: 'auto' to use code length n, or specify an integer)")
    parser.add_argument("--bb-schedule", type=str, default=DEFAULT_CONFIG["BB_SCHEDULE"], choices=["ibm", "7layer", "stim"], help=f"BB circuit schedule and noise model (default: {DEFAULT_CONFIG['BB_SCHEDULE']})")
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution")
    parser.add_argument("--dry-run", action="store_true", help="Print experiment points without running")
    parser.add_argument("--count-tasks", action="store_true", help="Print the number of experiment points and exit (for Slurm --array sizing)")
    parser.add_argument("--task-index", type=int, default=None, help="Run only the experiment point at this index (Slurm array task)")
    parser.add_argument("-r", "--result-dir", type=str, default=DEFAULT_CONFIG["BASE_RESULT_DIR"], help="Output results directory")
    return parser.parse_args()

LOTTERY_NEW_PRE_MAX_LAMBDA = 16

# 純 BP 版本 -> 帶 OSD-0 的對應演算法 (同一個 BP 設定，只拿掉 osd_0 階段)
PURE_BP_ALGOS = {
    "bp": "bposd",
    "lottery_bp": "lottery_bposd",
    "lottery_bp_new_arrangement": "lottery_bposd_new_arrangement",
}
NEW_ARRANGEMENT_ALGOS = ("lottery_bposd_new_arrangement", "lottery_bp_new_arrangement")

def lottery_new_arrangement_knobs(distance: int, rate: float, max_iter: int, pre_max_lambda: int = LOTTERY_NEW_PRE_MAX_LAMBDA) -> tuple:
    """
    bb_lottery_pre_setup.md 的 BB 排程，回傳 (fpi, pmi):
    - w   = floor(2 + 4880·d²·p²)   首次收斂 CDF 斜率最大的 iteration
    - cap = floor(2520·d^0.6·p)     95% 收斂的 iteration
    - fpi = max(min(n − Λ·w, cap), w)
    - pmi = max(1, ceil((n − fpi) / Λ))
    """
    w = math.floor(2 + 4880 * distance ** 2 * rate ** 2)
    cap = math.floor(2520 * distance ** 0.6 * rate)
    fpi = max(min(max_iter - pre_max_lambda * w, cap), w)
    pmi = max(1, math.ceil((max_iter - fpi) / pre_max_lambda))
    return fpi, pmi

def build_decoding_config(algo: str, max_iter: int, code_type: str = "144", use_cpu: bool = False,
                          distance: int = None, error_rate: float = None) -> dict:
    device_type = "cpu" if use_cpu else "cuda"

    cfg_inner = {
        "max_iter": int(max_iter)
    }

    pure_bp = algo in PURE_BP_ALGOS
    algo = PURE_BP_ALGOS.get(algo, algo)

    if algo == "bposd":
        algorithm_val = ["bp_norm_min_sum", "osd_0"]
    elif algo == "lottery_bposd":
        algorithm_val = ["bp_lottery", "osd_0"]
    elif algo == "lottery_bposd_new_arrangement":
        if str(code_type).lower() == "surface":
            raise ValueError("new arrangement 只支援 BB code")
        algorithm_val = ["bp_lottery", "osd_0"]
        fpi, pmi = lottery_new_arrangement_knobs(distance, error_rate, int(max_iter))
        cfg_inner.update({
            "early_termination": True,
            "flip_phase_iters": [fpi],
            "flip_mode": ["lottery", "pre"],
            "pre_max_lambda": LOTTERY_NEW_PRE_MAX_LAMBDA,
            "pre_max_iter": pmi,
            "random_machine": "sobol"
        })
    elif algo == "relay_bp":
        algorithm_val = "relay_bp"
        is_bb = str(code_type).lower() != "surface"
        cfg_inner.update({
            "legs": 601,
            "iteration_initial": 80,
            "iteration_count": 60,
            "solution": 5,
            "init_mem_strength": 0.125 if is_bb else 0.35,
            "center": 0.21 if is_bb else 0.3655,
            "width": 0.90 if is_bb else 1.239,
            "alpha": 0.0,
            "alpha_scaling": 1.0
        })
    else:
        algorithm_val = algo

    if pure_bp:
        algorithm_val = algorithm_val[0]

    dec_body = {
        "algorithm": algorithm_val,
        "check_type": "hx",
        "dtype": "float64",
        "device": {
            "device_type": device_type,
            "device_idx": 0
        },
        "config": cfg_inner
    }

    # 同時提供 decoding 與 decoder，100% 雙向相容新舊版 Syndrilla
    cfg = {
        "decoding": dec_body,
        "decoder": dec_body
    }
    return cfg

def resolve_target_error(te_cfg, rate: float) -> int:
    """依物理錯誤率解析對應的 target error 目標數"""
    if isinstance(te_cfg, dict):
        if rate in te_cfg:
            return int(te_cfg[rate])
        for k, v in te_cfg.items():
            if isinstance(k, (int, float)) and abs(float(k) - rate) < 1e-7:
                return int(v)
        return int(te_cfg.get("default", 100))
    elif isinstance(te_cfg, str) and (":" in te_cfg or "," in te_cfg):
        items = dict(item.split(":") for item in te_cfg.split(","))
        mapped = {float(k) if k != "default" else "default": int(v) for k, v in items.items()}
        return resolve_target_error(mapped, rate)
    return int(te_cfg)

def resolve_code_n(code: str, distance: int = None) -> int:
    """
    解析代碼之物理資料比特數 n ([[n, k, d]])
    - BB Code (72, 144, 288, 360, 784): n 即為碼名稱之整數
    - Surface Code (unrotated: 2*d*(d-1)+1; 例如 d=3 -> 13, d=9 -> 145)
    """
    s = str(code).strip()
    if s.lower() == "surface":
        dist = int(distance) if distance else 3
        return 2 * dist * (dist - 1) + 1
    elif s.startswith("[["):
        parts = s.strip("[]").split(",")
        return int(parts[0].strip())
    elif s.isdigit():
        return int(s)
    else:
        prefix, bb_key, _ = resolve_bb_prefix(s)
        return int(bb_key) if str(bb_key).isdigit() else 100

def resolve_max_iter(max_iter_setting, code: str, distance: int = None) -> int:
    """
    解析 max_iter：
    - 若設定為 "auto"、"n" 或 None: 自動使用該碼之物理比特數 n
    - 若指定為整數: 使用使用者指定之整數
    """
    if max_iter_setting is None or str(max_iter_setting).lower() in ("auto", "none", "n"):
        return resolve_code_n(code, distance)
    return int(max_iter_setting)

def get_existing_result_file(point_dir: str) -> str:
    """找出該點已存在的 result_phy_err_*.yaml 完整路徑"""
    if not os.path.exists(point_dir):
        return None
    for fname in os.listdir(point_dir):
        if fname.startswith("result_phy_err_") and fname.endswith(".yaml"):
            return os.path.join(point_dir, fname)
    return None

def get_checkpoint_field(res_file: str, field: str):
    """結果檔 decoder_full 區塊的整數欄位 (例如 "batch size"、"target error")，讀不到回傳 None"""
    in_full = False
    try:
        with open(res_file, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith("decoder_full:"):
                    in_full = True
                elif in_full:
                    if not line.startswith(" "):
                        break
                    key, _, val = line.strip().partition(":")
                    if key == field:
                        return int(val.strip())
    except (OSError, ValueError):
        pass
    return None

def set_checkpoint_target_error(res_file: str, target_error: int):
    """改寫 decoder_full 的 "target error"。它只是停止條件，不影響任何統計；
    syndrilla 接續時要求它和 -te 相同，改寫後才能把已完成的點繼續累積到新的目標"""
    with open(res_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    in_full = False
    for i, line in enumerate(lines):
        if line.startswith("decoder_full:"):
            in_full = True
        elif in_full:
            if not line.startswith(" "):
                break
            if line.strip().startswith("target error:"):
                indent = line[:len(line) - len(line.lstrip())]
                lines[i] = f"{indent}target error: {target_error}\n"
                break
    tmp = res_file + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.writelines(lines)
    os.replace(tmp, res_file)

def is_point_finished(point_dir: str, target_error: int) -> bool:
    """
    Check if a result_phy_err_*.yaml already exists and has reached target error.
    """
    res_file = get_existing_result_file(point_dir)
    if res_file:
        try:
            with open(res_file, "r", encoding="utf-8") as f:
                res = yaml.safe_load(f)
                df = res.get("decoder_full", {})
                reached = df.get("target error reached", 0)
                if reached >= target_error:
                    return True
        except Exception:
            pass
    return False

def worker_task(task_info: tuple, gpu_queue: Queue, dry_run: bool = False):
    code, distance, error_rate, algo, batch_size, target_error, max_iter, base_dir, use_cpu, bb_schedule = task_info

    code_tag = f"BB_{code}" if str(code).isdigit() else str(code).replace("[", "").replace("]", "").replace(",", "_")
    if code.lower() != "surface":
        code_tag += f"_{bb_schedule}"
    point_dir = os.path.join(base_dir, code_tag, f"d_{distance}", algo, f"p_{error_rate}")
    
    # 1. 斷點略過檢查
    if is_point_finished(point_dir, target_error):
        print(f"\033[93m[SKIP]\033[0m {code_tag} d={distance} {algo} p={error_rate} already finished in {point_dir}.")
        return

    os.makedirs(point_dir, exist_ok=True)
    interface_yaml_path = os.path.join(point_dir, "interface.yaml")
    decoding_yaml_path = os.path.join(point_dir, "decoding.yaml")

    # 2. 動態生成 interface.yaml
    if code.lower() == "surface":
        generate_surface_interface_yaml(
            distance=distance,
            error_rate=error_rate,
            rounds=distance,
            output_yaml_path=interface_yaml_path
        )
    else:
        generate_bb_interface_yaml(
            code_type=code,
            error_rate=error_rate,
            rounds=distance,
            output_yaml_path=interface_yaml_path,
            noise_model=DEFAULT_CONFIG["NOISE_MODEL"],
            schedule=bb_schedule
        )

    # 3. 動態生成 decoding.yaml
    dec_dict = build_decoding_config(algo, max_iter, code_type=code, use_cpu=use_cpu,
                                     distance=distance, error_rate=error_rate)
    with open(decoding_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(dec_dict, f, default_flow_style=False, sort_keys=False)

    if algo in NEW_ARRANGEMENT_ALGOS:
        knobs = dec_dict["decoding"]["config"]
        print(f"\033[96m[KNOBS]\033[0m {code_tag} d={distance} p={error_rate}: max_iter={max_iter}, "
              f"fpi={knobs['flip_phase_iters'][0]}, pmi={knobs['pre_max_iter']}, lambda={knobs['pre_max_lambda']}")

    if dry_run:
        cmd = [
            sys.executable, "-m", "syndrilla.main",
            f"-r={point_dir}", f"-i={interface_yaml_path}", f"-d={decoding_yaml_path}",
            f"-bs={batch_size}", f"-te={target_error}"
        ]
        print(f"\033[96m[DRY-RUN]\033[0m {' '.join(cmd)}")
        return

    # 4. 取得 GPU 資源執行
    gpu_id = gpu_queue.get()
    try:
        env = os.environ.copy()
        if not use_cpu:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            gpu_str = f"GPU {gpu_id}"
        else:
            gpu_str = "CPU"

        start_time = time.time()
        current_bs = batch_size
        oom_shrunk = False

        while True:
            # syndrilla 接續時要求 batch size 與 checkpoint 相同:
            # - 一般重啟: 改用 checkpoint 的 batch size 接續
            # - OOM 減半後: 無法用較小的 batch 接續，把舊結果備份成 .bak 後從頭跑這個點
            existing_ckpt = get_existing_result_file(point_dir)
            ckpt_bs = get_checkpoint_field(existing_ckpt, "batch size") if existing_ckpt else None
            if ckpt_bs and ckpt_bs != current_bs:
                if not oom_shrunk:
                    print(f"\033[96m[RESUME]\033[0m {code_tag} d={distance} {algo} p={error_rate}: 沿用 checkpoint 的 bs={ckpt_bs} (設定為 {current_bs})")
                    current_bs = ckpt_bs
                else:
                    backup = f"{existing_ckpt}.bs{ckpt_bs}.bak"
                    os.replace(existing_ckpt, backup)
                    print(f"\033[93m[RESTART]\033[0m {code_tag} d={distance} {algo} p={error_rate}: bs={current_bs} 無法接續 bs={ckpt_bs} 的 checkpoint，舊結果備份為 {os.path.basename(backup)}，從頭跑")
                    existing_ckpt = None

            cmd = [
                sys.executable, "-m", "syndrilla.main",
                f"-r={point_dir}",
                f"-i={interface_yaml_path}",
                f"-d={decoding_yaml_path}",
                f"-bs={current_bs}",
                f"-te={target_error}"
            ]

            # 斷點接續: 如果已有未達標的結果檔案，傳入 -ckpt 進行接續累積 (修復原先誤用 -c 的 Bug)
            if existing_ckpt:
                ckpt_te = get_checkpoint_field(existing_ckpt, "target error")
                if ckpt_te is not None and ckpt_te != target_error:
                    print(f"\033[96m[RESUME]\033[0m {code_tag} d={distance} {algo} p={error_rate}: target error {ckpt_te} -> {target_error}，接續累積")
                    set_checkpoint_target_error(existing_ckpt, target_error)
                cmd.append(f"-ckpt={existing_ckpt}")

            print(f"\033[92m[START]\033[0m [{gpu_str}] {code_tag} d={distance} {algo} p={error_rate} (bs={current_bs}) -> {point_dir}")

            # 執行 Syndrilla 模擬
            log_file = os.path.join(point_dir, "runner_exec.log")
            with open(log_file, "w") as lf:
                res = subprocess.run(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)

            if res.returncode == 0:
                elapsed = time.time() - start_time
                print(f"\033[92m[DONE]\033[0m [{gpu_str}] {code_tag} d={distance} {algo} p={error_rate} (Success at bs={current_bs}, {elapsed:.1f}s)")
                break
            else:
                # 檢查是否為 Out of Memory 錯誤
                log_content = ""
                if os.path.exists(log_file):
                    try:
                        with open(log_file, "r", encoding="utf-8", errors="ignore") as lf:
                            log_content = lf.read()
                    except Exception:
                        pass

                oom_keywords = ["cuda out of memory", "out of memory", "alloc", "cublas_status_not_initialized"]
                if any(kw in log_content.lower() for kw in oom_keywords):
                    if current_bs <= 1:
                        print(f"\033[91m[FAILED]\033[0m [{gpu_str}] {code_tag} d={distance} {algo} 在 bs=1 仍然 OOM，放棄此點！")
                        break
                    next_bs = max(1, current_bs // 2)
                    print(f"\033[93m[OOM 💥]\033[0m [{gpu_str}] {code_tag} d={distance} {algo} p={error_rate} bs={current_bs} 記憶體不足，自動減半至 bs={next_bs} 重跑...")
                    current_bs = next_bs
                    oom_shrunk = True
                else:
                    print(f"\033[91m[FAILED]\033[0m [{gpu_str}] {code_tag} d={distance} {algo} p={error_rate} 非記憶體錯誤 (code={res.returncode}, see {log_file})")
                    break

    finally:
        gpu_queue.put(gpu_id)

def main():
    args = parse_args()

    codes = [c.strip() for c in args.codes.split(",")] if args.codes else DEFAULT_CONFIG["CODES"]
    rates = [float(r.strip()) for r in args.rates.split(",")] if args.rates else DEFAULT_CONFIG["ERROR_RATES"]
    algos = [a.strip() for a in args.algos.split(",")] if args.algos else DEFAULT_CONFIG["BP_ALGOS"]
    surface_dists = [int(d.strip()) for d in args.distances.split(",")] if args.distances else DEFAULT_CONFIG["SURFACE_DISTANCES"]

    if any(a in NEW_ARRANGEMENT_ALGOS for a in algos) and any(c.lower() == "surface" for c in codes):
        sys.exit("new arrangement 只支援 BB code (bb_lottery_pre_setup.md 的 surface 60/40 override 未定義)，請從 --codes 拿掉 surface")

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    if not gpus and not args.cpu:
        gpus = ["0"]

    # 展開所有待測試點
    tasks = []
    te_setting = args.target_error if args.target_error is not None else DEFAULT_CONFIG["TARGET_ERROR"]

    for code in codes:
        if code.lower() == "surface":
            dists = surface_dists
        else:
            _, _, default_dist = resolve_bb_prefix(code)
            dists = [default_dist]

        for dist, rate, algo in itertools.product(dists, rates, algos):
            pt_target_error = resolve_target_error(te_setting, rate)
            pt_max_iter = resolve_max_iter(args.max_iter, code, dist)
            tasks.append((
                code, dist, rate, algo,
                args.batch_size, pt_target_error, pt_max_iter,
                args.result_dir, args.cpu, args.bb_schedule
            ))

    if args.count_tasks:
        print(len(tasks))
        return
    if args.task_index is not None:
        if not 0 <= args.task_index < len(tasks):
            sys.exit(f"--task-index {args.task_index} 超出範圍 (共 {len(tasks)} 個點)")
        tasks = [tasks[args.task_index]]

    print("=" * 70)
    print("  SYNDRILLA SWEEP RUNNER v3")
    print(f"  Total points: {len(tasks)}")
    print(f"  Codes:        {codes}")
    print(f"  Error Rates:  {rates}")
    print(f"  Algorithms:   {algos}")
    print(f"  BB schedule:  {args.bb_schedule}")
    print(f"  GPUs:         {gpus} (workers/gpu: {args.workers_per_gpu})")
    print(f"  Batch size:   {args.batch_size}, Target error config: {te_setting}, Max iter config: {args.max_iter}")
    print("=" * 70)

    if args.dry_run:
        print("\n--- Dry Run Mode: Previewing all tasks ---")
        dummy_q = Queue()
        for g in gpus: dummy_q.put(g)
        for t in tasks:
            worker_task(t, dummy_q, dry_run=True)
        return

    # 初始化 GPU Queue (每個 GPU 可以放 workers_per_gpu 個 token)
    gpu_queue = Queue()
    for g in gpus:
        for _ in range(args.workers_per_gpu):
            gpu_queue.put(g)

    total_workers = len(gpus) * args.workers_per_gpu if not args.cpu else args.workers_per_gpu
    total_workers = max(1, total_workers)

    start_all = time.time()
    with ThreadPoolExecutor(max_workers=total_workers) as executor:
        futures = [executor.submit(worker_task, t, gpu_queue, False) for t in tasks]
        for f in futures:
            f.result()

    total_elapsed = time.time() - start_all
    print("=" * 70)
    print(f"\033[92mALL SWEEP TASKS COMPLETED!\033[0m Total time: {total_elapsed:.1f}s")
    print("=" * 70)

if __name__ == "__main__":
    main()
