"""Sweep runner for main_kshot_dem.py: one task per (code, p), spread over GPUs.

Each task runs every window config on the same samples and writes
results_sw/BB_<n>_ibm/p_<p>/sw_result.yaml (resumed when it already exists).

Usage (repo root):
  python dem_sw/run_sliding.py --codes 144 --rates 0.001,0.002,0.003,0.005 --gpus 0,1 --dry-run
"""
import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)

DEFAULT_WINDOWS = "0;1;2;3;4;3,1;3,2;4,2"
DEFAULT_RATES = [0.001, 0.002, 0.003, 0.005]
# logical errors per config; low p is expensive, so it gets fewer
DEFAULT_TARGET_ERROR = {0.001: 20, 0.002: 50, 0.003: 100, 0.005: 200, "default": 100}


def resolve_target_error(spec, rate):
    if isinstance(spec, dict):
        return int(spec.get(rate, spec.get("default", 100)))
    if isinstance(spec, str) and ":" in spec:
        items = dict(item.split(":") for item in spec.split(","))
        return resolve_target_error({(k if k == "default" else float(k)): int(v) for k, v in items.items()}, rate)
    return int(spec)


def point_dir(base, code, rate):
    return os.path.join(base, f"BB_{code}_ibm", f"p_{rate}")


def is_finished(path, target_error, max_shots, num_configs):
    res_file = os.path.join(path, "sw_result.yaml")
    if not os.path.exists(res_file):
        return False
    try:
        with open(res_file, encoding="utf-8") as f:
            cfgs = (yaml.safe_load(f) or {}).get("configs", {})
    except Exception:
        return False
    if len(cfgs) < num_configs:
        return False
    errors_done = min(c["errors"] for c in cfgs.values()) >= target_error
    shots_done = max_shots > 0 and min(c["shots"] for c in cfgs.values()) >= max_shots
    return errors_done or shots_done


def worker(task, gpu_queue, args):
    code, rate, target_error = task
    out = point_dir(args.result_dir, code, rate)
    num_configs = len([w for w in args.windows.split(";") if w.strip()])
    if is_finished(out, target_error, args.max_shots, num_configs):
        print(f"[SKIP] BB_{code} p={rate} finished in {out}")
        return
    cmd = [sys.executable, os.path.join(HERE, "main_kshot_dem.py"), "--code", str(code), "-p", str(rate),
           "--windows", args.windows, "--algo", args.algo, "-bs", str(args.batch_size),
           "-te", str(target_error), "--max-shots", str(args.max_shots), "-r", out]
    if args.max_iter:
        cmd += ["--max-iter", str(args.max_iter)]
    if args.cpu:
        cmd.append("--cpu")
    if args.dry_run:
        print("[DRY-RUN] " + " ".join(f'"{c}"' if ";" in c else c for c in cmd))
        return

    gpu = gpu_queue.get()
    try:
        os.makedirs(out, exist_ok=True)
        env = dict(os.environ)
        if not args.cpu:
            env["CUDA_VISIBLE_DEVICES"] = gpu
        t0 = time.time()
        print(f"[START] gpu={gpu} BB_{code} p={rate} te={target_error} -> {out}", flush=True)
        with open(os.path.join(out, "run.log"), "a", encoding="utf-8") as log:
            res = subprocess.run(cmd, cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        status = "DONE" if res.returncode == 0 else f"FAILED code={res.returncode}"
        print(f"[{status}] BB_{code} p={rate} ({time.time() - t0:.0f}s), log: {out}/run.log", flush=True)
    finally:
        gpu_queue.put(gpu)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--codes", default="144")
    ap.add_argument("--rates", default=None, help=f"comma separated, default {DEFAULT_RATES}")
    ap.add_argument("--windows", default=DEFAULT_WINDOWS)
    ap.add_argument("--algo", default="bposd", choices=["bposd", "lottery_bposd"])
    ap.add_argument("--max-iter", type=int, default=None, help="default: code length n")
    ap.add_argument("-bs", "--batch-size", type=int, default=5000)
    ap.add_argument("-te", "--target-error", default=None, help="'100' or '0.001:20,0.003:100,default:50'")
    ap.add_argument("--max-shots", type=int, default=0)
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--workers-per-gpu", type=int, default=1)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("-r", "--result-dir", default=os.path.join(REPO_ROOT, "results_sw"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    codes = [c.strip() for c in args.codes.split(",") if c.strip()]
    rates = [float(r) for r in args.rates.split(",")] if args.rates else DEFAULT_RATES
    te_spec = args.target_error if args.target_error is not None else DEFAULT_TARGET_ERROR
    tasks = [(c, r, resolve_target_error(te_spec, r)) for c in codes for r in rates]
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()] or ["0"]

    print("=" * 70)
    print(f"  DEM SLIDING WINDOW SWEEP: {len(tasks)} points")
    print(f"  codes={codes} rates={rates} algo={args.algo}")
    print(f"  windows={args.windows}")
    print(f"  gpus={gpus} x {args.workers_per_gpu}, batch={args.batch_size}, max_shots={args.max_shots}")
    print("=" * 70)

    gpu_queue = Queue()
    for g in gpus:
        for _ in range(args.workers_per_gpu):
            gpu_queue.put(g)
    workers = 1 if args.dry_run else max(1, len(gpus) * args.workers_per_gpu)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for f in [ex.submit(worker, t, gpu_queue, args) for t in tasks]:
            f.result()


if __name__ == "__main__":
    main()
