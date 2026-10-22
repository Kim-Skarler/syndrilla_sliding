"""Sliding-window BP+OSD decoding directly on a Stim DEM.

Same inputs as `syndrilla.main`: an interface YAML holding the Stim circuit (`-i`) and a
decoding YAML (`-d`) whose `algorithm` is `[<bp>, osd_0]`. The circuit's detectors must
carry their round as the last coordinate (true for tools/bb_circuit_generator.py and for
stim's generated circuits).

Windows (stim_dem_migration_plan.md §2.2):
- a class (DEM column) belongs to round r_min, the first detector round it touches
- window t covers detector rounds [t, min(t+T, D)) and the classes with r_min in [t, t+T);
  classes reaching past the window keep only their rows inside it, with the DEM prior
- BP runs on the T_BP window; samples BP does not converge go to OSD-0 on the T_OSD window,
  fed with the BP posterior of those columns
- only classes with r_min == t are committed; they update the residual syndrome of later
  windows (H) and the predicted observables (L), so no data error vector is needed

Every config in `-w` decodes the same Stim samples. "0" is the baseline (one window holding
the whole DEM), "T" is SW(T) with T_BP = T_OSD = T, and "a,b" is SW(T_BP=a, T_OSD=b).

  python -m syndrilla.main_sliding -r=<dir> -i=interface.yaml -d=decoding.yaml -w="0;1;2;3" -te=100
  python -m syndrilla.main_sliding -i=interface.yaml --check     # window bookkeeping test, no torch
"""
import argparse
import hashlib
import os
import time

import numpy as np
import scipy.sparse as sp
import yaml


# ----------------------------------------------------------------------------- DEM

class Dem:
    """H (detectors x classes), L (observables x classes), priors and round bookkeeping."""

    def __init__(self, circuit):
        self.circuit = circuit
        self.dem = circuit.detector_error_model(decompose_errors=False)
        coords = circuit.get_detector_coordinates()
        num_det = circuit.num_detectors
        if any(not coords.get(i) for i in range(num_det)):
            raise ValueError("every detector needs coordinates whose last entry is its round")
        self.det_round = np.array([int(coords[i][-1]) for i in range(num_det)])
        self.rounds = int(self.det_round.max())  # number of detector rounds D = rounds + 1

        h_rows, h_cols, l_rows, l_cols, prior = [], [], [], [], []
        for inst in self.dem.flattened():
            if inst.type != "error":
                continue
            col = len(prior)
            prior.append(inst.args_copy()[0])
            for tgt in inst.targets_copy():
                if tgt.is_relative_detector_id():
                    h_rows.append(tgt.val); h_cols.append(col)
                elif tgt.is_logical_observable_id():
                    l_rows.append(tgt.val); l_cols.append(col)
        n_col = len(prior)
        self.H = sp.csr_matrix((np.ones(len(h_rows), dtype=np.uint8), (h_rows, h_cols)), shape=(num_det, n_col))
        self.L = sp.csr_matrix((np.ones(len(l_rows), dtype=np.uint8), (l_rows, l_cols)),
                               shape=(circuit.num_observables, n_col))
        self.prior = np.array(prior)

        Hc = self.H.tocsc()
        self.col_rmin = np.full(n_col, -1)
        for c in range(n_col):
            dets = Hc.indices[Hc.indptr[c]:Hc.indptr[c + 1]]
            if dets.size:
                self.col_rmin[c] = self.det_round[dets].min()
        undetectable = int(((self.col_rmin < 0) & (np.asarray(self.L.sum(axis=0)).ravel() > 0)).sum())
        if undetectable:
            raise ValueError(f"DEM has <{undetectable}> classes that flip an observable but no detector")


def load_circuit(interface_yaml):
    import stim

    with open(interface_yaml, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    circuit = (cfg.get("interface") or {}).get("circuit")
    if not isinstance(circuit, str):
        raise ValueError(f"<{interface_yaml}> needs an inline stim circuit under interface.circuit")
    return stim.Circuit(circuit)


# ----------------------------------------------------------------------------- windows

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
        # windows with the same structure and priors share one decoder
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


def run_check(d, configs, shots, seed):
    """With the true errors as the decision, every window's residual syndrome must be
    explained by the window's own columns, each class must be committed once, and the run
    must end with zero residual and the true observables."""
    dets, obs, errs = d.dem.compile_sampler(seed=seed).sample(shots, return_errors=True)
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

def read_decoding(decoding_yaml, force_cpu):
    """BP and OSD stage configs from a decoding YAML with `algorithm: [<bp>, osd_0]`."""
    import torch

    with open(decoding_yaml, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["decoding"]
    algos = cfg["algorithm"] if isinstance(cfg["algorithm"], list) else [cfg["algorithm"]]
    if len(algos) != 2 or algos[1] != "osd_0":
        raise ValueError(f"main_sliding needs algorithm: [<bp>, osd_0], got <{algos}>")
    stage_cfg = cfg.get("config") or {}
    stage_cfg = stage_cfg if isinstance(stage_cfg, list) else [stage_cfg, {}]

    want_cuda = (cfg.get("device") or {}).get("device_type", "cuda") == "cuda"
    device = torch.device("cuda:0" if want_cuda and torch.cuda.is_available() and not force_cpu else "cpu")
    dtype = getattr(torch, str(cfg.get("dtype", "float64")))
    shared = {k: v for k, v in cfg.items() if k not in ("algorithm", "config", "device")}
    shared["device"] = {"device_type": device.type, "device_idx": device.index or 0}

    stages = []
    for algo, sc in zip(algos, stage_cfg):
        stage = {**shared, "algorithm": algo}
        if sc:
            stage["config"] = sc
        stages.append(stage)
    return stages, device, dtype, algos


class WindowDecoders:
    """BP and OSD-0 decoders per distinct window structure, built with syndrilla."""

    def __init__(self, stages, device):
        from syndrilla.decoder.decoder import create_decoder
        from syndrilla.matrix.matrix import MatrixBundle, dense_to_index_format

        self.stages = {"bp": stages[0], "osd": stages[1]}
        self.device = device
        self._create_decoder = create_decoder
        self._bundle_cls = MatrixBundle
        self._to_index = dense_to_index_format
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

    def get(self, key, H, kind):
        if (key, kind) not in self.cache:
            bundle = self._bundle(H)
            dec = self._create_decoder(cfg=dict(self.stages[kind]), bundle=bundle)[0]
            dec.to(self.device).eval()
            self.cache[(key, kind)] = (dec, bundle.select("hx")[3])
        return self.cache[(key, kind)]


def decode_config(wd, d, windows, dets, llr_all, H_t, L_t):
    """Decode one batch with one window config."""
    import torch

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

    stages, device, dtype, algos = read_decoding(args.decoding_yaml, args.cpu)
    wd = WindowDecoders(stages, device)
    windows = {c: build_windows(d, *c) for c in configs}
    H_t = torch.as_tensor(d.H.T.toarray(), dtype=torch.float32, device=device)  # classes x detectors
    L_t = torch.as_tensor(d.L.T.toarray(), dtype=torch.float32, device=device)
    llr_all = torch.as_tensor(np.log((1 - d.prior) / d.prior), dtype=dtype, device=device)

    os.makedirs(args.run_dir, exist_ok=True)
    out_path = os.path.join(args.run_dir, "sw_result.yaml")
    meta = {"interface": os.path.abspath(args.interface_yaml), "algorithm": algos,
            "bp_config": stages[0].get("config", {}), "p": args.p, "rounds": d.rounds,
            "seed": args.seed, "dem_shape": list(d.H.shape)}
    keys = ("shots", "errors", "bp_conv", "bp_iter", "resid_nonzero", "time_s")
    stats = {c: dict.fromkeys(keys, 0) for c in configs}
    if os.path.exists(out_path) and not args.fresh:
        with open(out_path, encoding="utf-8") as f:
            prev = yaml.safe_load(f) or {}
        if prev.get("meta") == meta:
            for c in configs:
                r = prev["configs"].get(config_label(*c))
                if r:
                    stats[c] = {k: r[k] for k in keys}
            print(f"[resume] {out_path}")

    def shots_done():
        return min(s["shots"] for s in stats.values())

    batch = 0
    while (min(s["errors"] for s in stats.values()) < args.target_error
           and (args.max_shots <= 0 or shots_done() < args.max_shots)):
        # seeding by the shot count keeps a resumed run on fresh samples
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


def parse_commandline_args():
    ap = argparse.ArgumentParser(description="Sliding-window BP+OSD on a Stim DEM")
    ap.add_argument("-r", "--run_dir", default="tests/test_outputs", help="output directory")
    ap.add_argument("-i", "--interface_yaml", required=True, help="interface YAML with the stim circuit")
    ap.add_argument("-d", "--decoding_yaml", default=None, help="decoding YAML, algorithm [<bp>, osd_0]")
    ap.add_argument("-w", "--windows", default="0;1;2;3;4;3,1;3,2;4,2",
                    help="'0' baseline, 'T' SW(T), 'a,b' SW(T_BP=a, T_OSD=b), separated by ';'")
    ap.add_argument("-bs", "--batch_size", type=int, default=5000)
    ap.add_argument("-te", "--target_error", type=int, default=100,
                    help="stop once every config has at least this many logical errors")
    ap.add_argument("-ms", "--max_shots", type=int, default=0, help="0 = no limit")
    ap.add_argument("-p", "--p", type=float, default=None, help="physical error rate, recorded in the result")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fresh", action="store_true", help="ignore an existing result file")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--check", action="store_true", help="window bookkeeping test with the true errors")
    ap.add_argument("--check_shots", type=int, default=200)
    return ap.parse_args()


def main():
    args = parse_commandline_args()
    d = Dem(load_circuit(args.interface_yaml))
    configs = parse_windows(args.windows)
    print(f"DEM={d.H.shape} rounds={d.rounds} configs={[config_label(*c) for c in configs]}")
    if args.check:
        run_check(d, configs, args.check_shots, args.seed)
    else:
        if not args.decoding_yaml:
            raise ValueError("decoding needs -d <decoding yaml>")
        run_decode(args, d, configs)


if __name__ == "__main__":
    main()
