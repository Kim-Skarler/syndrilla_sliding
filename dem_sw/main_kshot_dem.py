"""Sliding-window BP+OSD decoding directly on the Stim DEM (BB memory-X, IBM circuit).

Windows follow stim_dem_migration_plan.md §2.2:
- a class belongs to round r_min, the first detector round it touches
- window t covers detector rounds [t, min(t+T, D)) and the classes with r_min in [t, t+T);
  classes reaching past the window keep only their rows inside it, with the DEM prior
- BP runs on the T_BP window; samples BP does not converge go to OSD-0 on the T_OSD window,
  fed with the BP posterior of those columns (as in syndrilla_STIM_new main_kshot.py)
- only classes with r_min == t are committed; they update the residual syndrome of later
  windows (H) and the predicted observables (L_dem), so no data error vector or Lx is needed

Every config listed in --windows decodes the same Stim samples. "0" is the baseline: one
window holding the whole DEM. "T" is SW(T) with T_BP = T_OSD = T; "a,b" is SW(T_BP=a, T_OSD=b).

Usage (repo root):
  python dem_sw/main_kshot_dem.py -p 0.003 --windows "0;1;2;3;4;3,1;3,2;4,2" -te 100
  python dem_sw/main_kshot_dem.py -p 0.003 --check      # window bookkeeping test, numpy only
"""
import argparse
import hashlib
import os
import time

import numpy as np
import scipy.sparse as sp
import yaml

from dem_tools import REPO_ROOT, build_dem

ALGOS = {"bposd": "bp_norm_min_sum", "lottery_bposd": "bp_lottery"}


def parse_windows(spec):
    configs = []
    for item in spec.split(";"):
        item = item.strip()
        if not item:
            continue
        parts = [int(x) for x in item.split(",")]
        t_bp, t_osd = (parts[0], parts[0]) if len(parts) == 1 else parts
        if t_osd > t_bp:
            raise ValueError(f"window <{item}>: T_OSD must not exceed T_BP")
        configs.append((t_bp, t_osd))
    return configs


def config_label(t_bp, t_osd):
    if t_bp == 0:
        return "baseline"
    return f"SW({t_bp})" if t_bp == t_osd else f"SW({t_bp},{t_osd})"


class Window:
    """Row/column sets of one window position, as global DEM indices and local positions."""

    def __init__(self, d, t, t_bp, t_osd):
        D = d.rounds + 1
        if t_bp == 0:  # baseline: the whole DEM, committed at once
            self.rows_bp = self.rows_osd = np.arange(d.H.shape[0])
            self.cols_bp = self.cols_osd = self.commit = np.arange(d.H.shape[1])
        else:
            self.rows_bp = np.nonzero((d.det_round >= t) & (d.det_round < min(t + t_bp, D)))[0]
            self.rows_osd = np.nonzero((d.det_round >= t) & (d.det_round < min(t + t_osd, D)))[0]
            self.cols_bp = np.nonzero((d.col_rmin >= t) & (d.col_rmin < t + t_bp))[0]
            self.cols_osd = np.nonzero((d.col_rmin >= t) & (d.col_rmin < t + t_osd))[0]
            self.commit = np.nonzero(d.col_rmin == t)[0]
        self.pos_osd_in_bp = np.searchsorted(self.cols_bp, self.cols_osd)
        self.pos_commit_in_osd = np.searchsorted(self.cols_osd, self.commit)
        self.H_bp = d.H[self.rows_bp][:, self.cols_bp].tocsr()
        self.H_osd = d.H[self.rows_osd][:, self.cols_osd].tocsr()
        self.key_bp = _structure_key(self.H_bp, d.prior[self.cols_bp])
        self.key_osd = _structure_key(self.H_osd, d.prior[self.cols_osd])


def _structure_key(H, prior):
    h = hashlib.sha1()
    for arr in (np.array(H.shape), H.indptr, H.indices, np.round(prior, 15)):
        h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def build_windows(d, t_bp, t_osd):
    if t_bp == 0:
        return [Window(d, 0, 0, 0)]
    return [Window(d, t, t_bp, t_osd) for t in range(d.rounds + 1)]


# ----------------------------------------------------------------------------- check mode

def run_check(d, configs, shots, seed):
    """With the true errors as the decision, every window's residual syndrome must be
    explained by the window's own columns, and the run must end with zero residual and
    the true observables."""
    sampler = d.dem.compile_sampler(seed=seed)
    dets, obs, errs = sampler.sample(shots, return_errors=True)
    dets, obs, errs = dets.astype(np.uint8), obs.astype(np.uint8), errs.astype(np.uint8)
    H, L = d.H.tocsc(), d.L.tocsc()
    for t_bp, t_osd in configs:
        windows = build_windows(d, t_bp, t_osd)
        committed = np.zeros(d.H.shape[1], dtype=int)
        resid, obs_pred = dets.copy(), np.zeros_like(obs)
        for w in windows:
            for rows, cols, Hw in ((w.rows_bp, w.cols_bp, w.H_bp), (w.rows_osd, w.cols_osd, w.H_osd)):
                explained = (errs[:, cols] @ Hw.T.toarray().astype(np.int64)) % 2
                assert (resid[:, rows] == explained).all(), f"{config_label(t_bp, t_osd)}: window rows not explained"
            e = errs[:, w.commit]
            resid ^= ((e @ H[:, w.commit].T.toarray().astype(np.int64)) % 2).astype(np.uint8)
            obs_pred ^= ((e @ L[:, w.commit].T.toarray().astype(np.int64)) % 2).astype(np.uint8)
            committed[w.commit] += 1
        assert (committed == 1).all(), "each class must be committed exactly once"
        assert not resid.any() and (obs_pred == obs).all()
        bulk = windows[len(windows) // 2]
        print(f"[check] {config_label(t_bp, t_osd):12s} windows={len(windows):2d} "
              f"bulk BP H={bulk.H_bp.shape} OSD H={bulk.H_osd.shape} commit={len(bulk.commit)}: OK")


# ----------------------------------------------------------------------------- decoding

class WindowDecoders:
    """BP and OSD-0 decoders per distinct window structure, built with syndrilla."""

    def __init__(self, d, algo, max_iter, device, dtype):
        import torch
        from syndrilla.decoder.decoder import create_decoder
        from syndrilla.matrix.matrix import MatrixBundle, dense_to_index_format

        self.torch = torch
        self._create_decoder = create_decoder
        self._bundle_cls = MatrixBundle
        self._to_index = dense_to_index_format
        self.bp_algo = ALGOS[algo]
        self.max_iter = max_iter
        self.device = device
        self.dtype = dtype
        self.cache = {}

    def _bundle(self, H):
        to_index, device = self._to_index, self.device

        class _WindowMatrix:
            path = "dem_window"

            def __init__(self):
                self.shape, self.V_c_row, self.V_c_col, self.matrix_tensor = to_index(H, device)

            def get_index(self):
                return self.shape, self.V_c_row, self.V_c_col, self.matrix_tensor

        m = _WindowMatrix()
        return self._bundle_cls(m, m, None, None)

    def _cfg(self, algorithm, with_iter):
        cfg = {"algorithm": algorithm, "check_type": "hx", "dtype": str(self.dtype).replace("torch.", ""),
               "device": {"device_type": self.device.type, "device_idx": self.device.index or 0}}
        if with_iter:
            cfg["config"] = {"max_iter": int(self.max_iter)}
        return cfg

    def get(self, key, H, kind):
        if (key, kind) not in self.cache:
            bundle = self._bundle(H)
            cfg = self._cfg(self.bp_algo, True) if kind == "bp" else self._cfg("osd_0", False)
            dec = self._create_decoder(cfg=cfg, bundle=bundle)[0]
            dec.to(self.device).eval()
            self.cache[(key, kind)] = (dec, bundle.select("hx")[3])
        return self.cache[(key, kind)]


def decode_config(wd, d, windows, dets, llr_all, H_t, L_t):
    """Decode one batch with one window config. Returns per-sample failure and stats."""
    torch = wd.torch
    B = dets.shape[0]
    resid = dets.clone()
    obs_pred = torch.zeros((B, d.L.shape[0]), dtype=torch.float32, device=wd.device)
    bp_conv = torch.zeros(B, device=wd.device)
    bp_iter = torch.zeros(B, device=wd.device)
    for w in windows:
        bp, H_bp = wd.get(w.key_bp, w.H_bp, "bp")
        llr0 = llr_all[w.cols_bp].unsqueeze(0).expand(B, -1).contiguous()
        out = bp({"synd": resid[:, w.rows_bp].long(), "llr0": llr0, "H_matrix": H_bp})
        conv = out["converge"].reshape(B)
        bp_conv += conv.float()
        bp_iter += out["iter"].reshape(B).float()
        e_bp = out["e_v"].reshape(B, -1)
        pos = torch.as_tensor(w.pos_osd_in_bp, device=wd.device)
        if (conv == 0).any():
            osd, _ = wd.get(w.key_osd, w.H_osd, "osd")
            o = osd({"synd": resid[:, w.rows_osd].long(), "llr": out["llr"].reshape(B, -1)[:, pos],
                     "e_v": e_bp[:, pos], "converge": conv, "iter": out["iter"].reshape(B)})
            e_win = o["e_v"].reshape(B, -1)
        else:
            e_win = e_bp[:, pos]
        e = e_win[:, torch.as_tensor(w.pos_commit_in_osd, device=wd.device)].float()
        commit = torch.as_tensor(w.commit, device=wd.device)
        resid = ((resid.float() + e @ H_t[commit]) % 2).to(resid.dtype)
        obs_pred = (obs_pred + e @ L_t[commit]) % 2
    return obs_pred, resid, bp_conv / len(windows), bp_iter / len(windows)


def run_decode(args, d, configs):
    import torch

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda:0")
    dtype = torch.float64
    wd = WindowDecoders(d, args.algo, args.max_iter, device, dtype)
    windows = {c: build_windows(d, *c) for c in configs}
    H_t = torch.as_tensor(d.H.T.toarray(), dtype=torch.float32, device=device)  # cols x dets
    L_t = torch.as_tensor(d.L.T.toarray(), dtype=torch.float32, device=device)
    llr_all = torch.as_tensor(np.log((1 - d.prior) / d.prior), dtype=dtype, device=device)

    out_dir = args.run_dir or os.path.join(REPO_ROOT, "results_sw", f"BB_{d.code}_ibm", f"p_{args.p}")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "sw_result.yaml")
    meta = {"code": d.code, "p": args.p, "rounds": d.rounds, "algo": args.algo, "max_iter": args.max_iter,
            "seed": args.seed, "dem_shape": list(d.H.shape)}
    stats = {c: {"shots": 0, "errors": 0, "bp_conv": 0.0, "bp_iter": 0.0, "resid_nonzero": 0, "time_s": 0.0}
             for c in configs}
    if os.path.exists(out_path) and not args.fresh:
        with open(out_path, encoding="utf-8") as f:
            prev = yaml.safe_load(f)
        if prev and prev.get("meta") == meta:
            for c in configs:
                r = prev["configs"].get(config_label(*c))
                if r:
                    stats[c] = {k: r[k] for k in stats[c]}
            print(f"[resume] {out_path}")

    def shots_done():
        return min(s["shots"] for s in stats.values())

    batch = 0
    while (min(s["errors"] for s in stats.values()) < args.target_error
           and (args.max_shots <= 0 or shots_done() < args.max_shots)):
        sampler = d.dem.compile_sampler(seed=args.seed + shots_done())
        dets_np, obs_np, _ = sampler.sample(args.batch_size)
        dets = torch.as_tensor(dets_np.astype(np.uint8), device=device)
        obs = torch.as_tensor(obs_np.astype(np.float32), device=device)
        for c in configs:
            t0 = time.time()
            obs_pred, resid, conv, iters = decode_config(wd, d, windows[c], dets, llr_all, H_t, L_t)
            if device.type == "cuda":
                torch.cuda.synchronize()
            s = stats[c]
            s["shots"] += args.batch_size
            s["errors"] += int((obs_pred != obs).any(dim=1).sum())
            s["bp_conv"] += float(conv.sum())
            s["bp_iter"] += float(iters.sum())
            s["resid_nonzero"] += int(resid.any(dim=1).sum())
            s["time_s"] += time.time() - t0
        batch += 1
        save(out_path, meta, stats, windows, d)
        print(f"[batch {batch}] shots={shots_done()} " +
              " ".join(f"{config_label(*c)}={stats[c]['errors']}" for c in configs), flush=True)
    save(out_path, meta, stats, windows, d)
    print(f"[saved] {out_path}")


def save(path, meta, stats, windows, d):
    res = {}
    for c, s in stats.items():
        n = max(s["shots"], 1)
        ler = s["errors"] / n
        bulk = windows[c][len(windows[c]) // 2]
        res[config_label(*c)] = {
            "t_bp": c[0], "t_osd": c[1], **s,
            "ler": ler,
            "ler_per_round": 1 - (1 - ler) ** (1 / d.rounds) if ler < 1 else 1.0,
            "bp_conv_rate": s["bp_conv"] / n,
            "bp_avg_iter": s["bp_iter"] / n,
            "num_windows": len(windows[c]),
            "bulk_bp_H": list(bulk.H_bp.shape),
            "bulk_osd_H": list(bulk.H_osd.shape),
        }
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        yaml.safe_dump({"meta": meta, "configs": res}, f, sort_keys=False)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--code", default="144")
    ap.add_argument("-p", type=float, required=True)
    ap.add_argument("--rounds", type=int, default=None, help="default: code distance")
    ap.add_argument("--windows", default="0;1;2;3;4;3,1;3,2;4,2")
    ap.add_argument("--algo", default="bposd", choices=sorted(ALGOS))
    ap.add_argument("--max-iter", type=int, default=None, help="default: code length n")
    ap.add_argument("-bs", "--batch-size", type=int, default=5000)
    ap.add_argument("-te", "--target-error", type=int, default=100,
                    help="stop once every config has at least this many logical errors")
    ap.add_argument("--max-shots", type=int, default=0, help="0 = no limit")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-r", "--run-dir", default=None, help="default: results_sw/BB_<n>_ibm/p_<p>")
    ap.add_argument("--fresh", action="store_true", help="ignore an existing result file")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--check", action="store_true", help="window bookkeeping test with the true errors")
    ap.add_argument("--check-shots", type=int, default=200)
    args = ap.parse_args()

    d = build_dem(args.code, args.p, args.rounds)
    if args.max_iter is None:
        args.max_iter = d.H.shape[0] // (d.rounds + 1) * 2  # n = 2 * checks for BB codes
    configs = parse_windows(args.windows)
    print(f"code={d.code} p={args.p} rounds={d.rounds} DEM={d.H.shape} configs="
          f"{[config_label(*c) for c in configs]}")
    if args.check:
        run_check(d, configs, args.check_shots, args.seed)
    else:
        run_decode(args, d, configs)


if __name__ == "__main__":
    main()
