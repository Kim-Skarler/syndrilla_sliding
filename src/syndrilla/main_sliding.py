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

Window merge and column order (§2.5), both switchable so their effect can be measured:
- merge on: classes whose rows inside a window part are identical (a truncation side effect)
  become one column with prior P(odd number fire) = (1 - prod(1 - 2p)) / 2. When the OSD part
  merges columns BP kept apart, their BP posteriors combine by the tanh rule. A committed
  merged column is restored to its member with the largest prior.
- column order sets the local index of each window column, which OSD-0's stable sort uses to
  break LLR ties: dem (Stim order), prior (largest prior first), random (fixed shuffle).

Every combination of `-w`, `-mw` and `-co` decodes the same Stim samples. In `-w`, "0" is the
baseline (one window holding the whole DEM), "T" is SW(T) with T_BP = T_OSD = T, and "a,b" is
SW(T_BP=a, T_OSD=b).

  python -m syndrilla.main_sliding -r=<dir> -i=interface.yaml -d=decoding.yaml -w="0;1;2;3" -mw=off,on -te=100
  python -m syndrilla.main_sliding -i=interface.yaml --check     # window bookkeeping test, no torch
"""
import argparse
import hashlib
import itertools
import os
import time
from collections import defaultdict

import numpy as np
import scipy.sparse as sp
import yaml

COL_ORDERS = ("dem", "prior", "random")


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
        self.Hc = self.H.tocsc()
        self.prior = np.array(prior)

        self.col_rmin = np.full(n_col, -1)
        for c in range(n_col):
            dets = self.col_dets(c)
            if dets.size:
                self.col_rmin[c] = self.det_round[dets].min()
        undetectable = int(((self.col_rmin < 0) & (np.asarray(self.L.sum(axis=0)).ravel() > 0)).sum())
        if undetectable:
            raise ValueError(f"DEM has <{undetectable}> classes that flip an observable but no detector")

    def col_dets(self, c):
        return self.Hc.indices[self.Hc.indptr[c]:self.Hc.indptr[c + 1]]


def load_circuit(interface_yaml):
    import stim

    with open(interface_yaml, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    circuit = (cfg.get("interface") or {}).get("circuit")
    if not isinstance(circuit, str):
        raise ValueError(f"<{interface_yaml}> needs an inline stim circuit under interface.circuit")
    return stim.Circuit(circuit)


# ----------------------------------------------------------------------------- configs

def parse_windows(spec):
    windows = []
    for item in spec.split(";"):
        item = item.strip()
        if not item:
            continue
        parts = [int(x) for x in item.split(",")]
        t_bp, t_osd = (parts[0], parts[0]) if len(parts) == 1 else parts
        if t_osd > t_bp:
            raise ValueError(f"window <{item}>: T_OSD must not exceed T_BP")
        windows.append((t_bp, t_osd))
    return windows


def parse_configs(windows_spec, merge_spec, order_spec):
    """Every (T_BP, T_OSD, merge, column order) combination."""
    merges = []
    for m in merge_spec.split(","):
        m = m.strip()
        if m not in ("off", "on"):
            raise ValueError(f"merge_window entries are off/on, got <{m}>")
        merges.append(m == "on")
    orders = [o.strip() for o in order_spec.split(",")]
    for o in orders:
        if o not in COL_ORDERS:
            raise ValueError(f"col_order entries are {COL_ORDERS}, got <{o}>")
    return [(t_bp, t_osd, m, o) for (t_bp, t_osd), m, o in itertools.product(parse_windows(windows_spec), merges, orders)]


def config_label(t_bp, t_osd, merge=False, order="dem"):
    if t_bp == 0:
        label = "baseline"
    else:
        label = f"SW({t_bp})" if t_bp == t_osd else f"SW({t_bp},{t_osd})"
    if merge:
        label += "+merge"
    if order != "dem":
        label += f"+{order}"
    return label


# ----------------------------------------------------------------------------- windows

class WindowPart:
    """The BP or the OSD part of one window: its rows and its (possibly merged) columns in
    local order. groups[j] lists the DEM classes behind local column j."""

    def __init__(self, d, rows, cols, merge, order, rng):
        self.rows = rows
        in_rows = np.zeros(d.H.shape[0], dtype=bool)
        in_rows[rows] = True
        by_rows = defaultdict(list)
        for c in cols:
            dets = d.col_dets(c)
            key = tuple(dets[in_rows[dets]]) if merge else (int(c),)
            by_rows[key].append(int(c))
        groups = [np.array(sorted(g)) for g in by_rows.values()]

        # P(odd number of members fire)
        prior = np.array([(1 - np.prod(1 - 2 * d.prior[g])) / 2 for g in groups])
        first = np.array([g[0] for g in groups])
        if order == "dem":
            perm = np.argsort(first, kind="stable")
        elif order == "prior":
            perm = np.lexsort((first, -prior))
        else:
            perm = rng.permutation(len(groups))
        self.groups = [groups[i] for i in perm]
        self.prior = prior[perm]
        # a committed merged column is restored to its most likely member
        self.rep = np.array([g[np.argmax(d.prior[g])] for g in self.groups])

        row_pos = {int(r): i for i, r in enumerate(rows)}
        h_rows, h_cols = [], []
        for j, g in enumerate(self.groups):
            dets = d.col_dets(g[0])
            for r in dets[in_rows[dets]]:
                h_rows.append(row_pos[int(r)]); h_cols.append(j)
        self.H = sp.csr_matrix((np.ones(len(h_rows), dtype=np.uint8), (h_rows, h_cols)),
                               shape=(len(rows), len(self.groups)))
        h = hashlib.sha1()  # parts with the same structure and priors share one decoder
        for arr in (np.array(self.H.shape), self.H.indptr, self.H.indices, np.round(self.prior, 15)):
            h.update(np.ascontiguousarray(arr).tobytes())
        self.key = h.hexdigest()


class Window:
    """One window position: its BP part, its OSD part, how OSD columns read the BP output,
    and which OSD columns are committed."""

    def __init__(self, d, t, t_bp, t_osd, merge, order, seed):
        D = d.rounds + 1
        rng = np.random.default_rng(seed + 1000 * t)
        if t_bp == 0:  # baseline: the whole DEM, committed at once
            rows = np.arange(d.H.shape[0])
            cols = np.arange(d.H.shape[1])
            self.bp = WindowPart(d, rows, cols, merge, order, rng)
            self.osd = self.bp
            commit_round = None
        else:
            rows_bp = np.nonzero((d.det_round >= t) & (d.det_round < min(t + t_bp, D)))[0]
            rows_osd = np.nonzero((d.det_round >= t) & (d.det_round < min(t + t_osd, D)))[0]
            cols_bp = np.nonzero((d.col_rmin >= t) & (d.col_rmin < t + t_bp))[0]
            cols_osd = np.nonzero((d.col_rmin >= t) & (d.col_rmin < t + t_osd))[0]
            self.bp = WindowPart(d, rows_bp, cols_bp, merge, order, rng)
            self.osd = self.bp if t_osd == t_bp else WindowPart(d, rows_osd, cols_osd, merge, order, rng)
            commit_round = t

        # OSD column j reads BP columns bp_idx[j, :] (padded with -1): a plain copy when the
        # two parts hold the same group, a tanh-rule combination of single BP columns otherwise
        bp_of = {}
        for j, g in enumerate(self.bp.groups):
            for c in g:
                bp_of[int(c)] = j
        width = max(len(g) for g in self.osd.groups)
        self.bp_idx = np.full((len(self.osd.groups), width), -1)
        for j, g in enumerate(self.osd.groups):
            src = sorted({bp_of[int(c)] for c in g})
            if len(src) == 1:
                if len(self.bp.groups[src[0]]) != len(g):
                    raise AssertionError("OSD group is a strict part of a BP group")
            elif any(len(self.bp.groups[s]) != 1 for s in src):
                raise AssertionError("OSD group mixes merged BP groups")
            self.bp_idx[j, :len(src)] = src

        # merged columns share their rows, so their members share r_min: a group is either
        # fully committed or not at all
        if commit_round is None:
            self.commit_pos = np.arange(len(self.osd.groups))
        else:
            self.commit_pos = np.array([j for j, g in enumerate(self.osd.groups)
                                        if d.col_rmin[g[0]] == commit_round], dtype=int)
        self.commit_rep = self.osd.rep[self.commit_pos]
        self.commit_members = np.concatenate([self.osd.groups[j] for j in self.commit_pos])


def build_windows(d, t_bp, t_osd, merge, order, seed):
    if t_bp == 0:
        return [Window(d, 0, 0, 0, merge, order, seed)]
    return [Window(d, t, t_bp, t_osd, merge, order, seed) for t in range(d.rounds + 1)]


def run_check(d, configs, shots, seed):
    """With the true errors as the decision, every window part's residual syndrome must be
    explained by its own (merged) columns, each class must be committed once, and the run
    must end with zero residual and the true observables."""
    dets, obs, errs = d.dem.compile_sampler(seed=seed).sample(shots, return_errors=True)
    dets, obs, errs = dets.astype(np.uint8), obs.astype(np.uint8), errs.astype(np.uint8)
    H, L = d.H.tocsc(), d.L.tocsc()
    for cfg in configs:
        label = config_label(*cfg)
        windows = build_windows(d, *cfg, seed)
        committed = np.zeros(d.H.shape[1], dtype=int)
        resid, obs_pred = dets.copy(), np.zeros_like(obs)
        for w in windows:
            for part in (w.bp, w.osd):
                group_err = np.stack([errs[:, g].sum(axis=1) % 2 for g in part.groups], axis=1)
                explained = (group_err @ part.H.T.toarray().astype(np.int64)) % 2
                assert (resid[:, part.rows] == explained).all(), f"{label}: window rows not explained"
            e = errs[:, w.commit_members]
            resid ^= ((e @ H[:, w.commit_members].T.toarray().astype(np.int64)) % 2).astype(np.uint8)
            obs_pred ^= ((e @ L[:, w.commit_members].T.toarray().astype(np.int64)) % 2).astype(np.uint8)
            committed[w.commit_members] += 1
        assert (committed == 1).all(), f"{label}: each class must be committed exactly once"
        assert not resid.any() and (obs_pred == obs).all()
        bulk = windows[len(windows) // 2]
        print(f"[check] {label:22s} windows={len(windows):2d} bulk BP H={bulk.bp.H.shape} "
              f"OSD H={bulk.osd.H.shape} commit={len(bulk.commit_pos)}: OK")


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
    """BP and OSD-0 decoders per distinct window part, built with syndrilla."""

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

    def get(self, part, kind):
        if (part.key, kind) not in self.cache:
            bundle = self._bundle(part.H)
            dec = self._create_decoder(cfg=dict(self.stages[kind]), bundle=bundle)[0]
            dec.to(self.device).eval()
            self.cache[(part.key, kind)] = (dec, bundle.select("hx")[3])
        return self.cache[(part.key, kind)]


class WindowTensors:
    """Per-window index tensors on the decoding device."""

    def __init__(self, w, dtype, device):
        import torch

        self.llr0 = torch.as_tensor(np.log((1 - w.bp.prior) / w.bp.prior), dtype=dtype, device=device)
        self.rows_bp = torch.as_tensor(w.bp.rows, device=device)
        self.rows_osd = torch.as_tensor(w.osd.rows, device=device)
        idx = torch.as_tensor(w.bp_idx, device=device)
        self.bp_idx = idx.clamp(min=0)
        self.pad = idx < 0
        self.single = bool((idx[:, 1:] < 0).all()) if idx.shape[1] > 1 else True
        self.commit_pos = torch.as_tensor(w.commit_pos, device=device)
        self.commit_rep = torch.as_tensor(w.commit_rep, device=device)


def osd_inputs(T, llr_bp, e_bp):
    """OSD-column posterior and hard decision from the BP output: copied, or for merged
    columns P(odd) by the tanh rule, 1 - 2p_G = prod(1 - 2q_i)."""
    import torch

    if T.single:
        return llr_bp[:, T.bp_idx[:, 0]], e_bp[:, T.bp_idx[:, 0]]
    tanh = torch.tanh(llr_bp[:, T.bp_idx] / 2).masked_fill(T.pad, 1.0)
    prod = tanh.prod(dim=2).clamp(-1 + 1e-15, 1 - 1e-15)
    llr = torch.where(T.pad[:, 1:].all(dim=1), llr_bp[:, T.bp_idx[:, 0]], 2 * torch.atanh(prod))
    e = (e_bp[:, T.bp_idx].masked_fill(T.pad, 0).sum(dim=2) % 2).to(e_bp.dtype)
    return llr, e


def decode_config(wd, d, windows, tensors, dets, H_t, L_t):
    """Decode one batch with one config."""
    import torch

    B = dets.shape[0]
    resid = dets.clone()
    obs_pred = torch.zeros((B, d.L.shape[0]), dtype=torch.float32, device=wd.device)
    bp_conv = torch.zeros(B, device=wd.device)
    bp_iter = torch.zeros(B, device=wd.device)
    for w, T in zip(windows, tensors):
        bp, H_bp = wd.get(w.bp, "bp")
        llr0 = T.llr0.unsqueeze(0).expand(B, -1).contiguous()
        out = bp({"synd": resid[:, T.rows_bp].long(), "llr0": llr0, "H_matrix": H_bp})
        conv = out["converge"].reshape(B)
        bp_conv += conv.float()
        bp_iter += out["iter"].reshape(B).float()
        llr_osd, e_osd = osd_inputs(T, out["llr"].reshape(B, -1), out["e_v"].reshape(B, -1))
        if (conv == 0).any():
            osd, _ = wd.get(w.osd, "osd")
            o = osd({"synd": resid[:, T.rows_osd].long(), "llr": llr_osd, "e_v": e_osd,
                     "converge": conv, "iter": out["iter"].reshape(B)})
            e_osd = o["e_v"].reshape(B, -1)
        e = e_osd[:, T.commit_pos].float()
        resid = ((resid.float() + e @ H_t[T.commit_rep]) % 2).to(resid.dtype)
        obs_pred = (obs_pred + e @ L_t[T.commit_rep]) % 2
    return obs_pred, resid, bp_conv / len(windows), bp_iter / len(windows)


def run_decode(args, d, configs):
    import torch

    stages, device, dtype, algos = read_decoding(args.decoding_yaml, args.cpu)
    wd = WindowDecoders(stages, device)
    windows = {c: build_windows(d, *c, args.seed) for c in configs}
    tensors = {c: [WindowTensors(w, dtype, device) for w in windows[c]] for c in configs}
    H_t = torch.as_tensor(d.H.T.toarray(), dtype=torch.float32, device=device)  # classes x detectors
    L_t = torch.as_tensor(d.L.T.toarray(), dtype=torch.float32, device=device)

    os.makedirs(args.run_dir, exist_ok=True)
    out_path = os.path.join(args.run_dir, "sw_result.yaml")
    meta = {"interface": os.path.abspath(args.interface_yaml), "algorithm": algos,
            "bp_config": stages[0].get("config", {}), "p": args.p, "rounds": d.rounds,
            "windows": args.windows, "merge_window": args.merge_window, "col_order": args.col_order,
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
        else:
            print(f"[fresh] {out_path} was written with other settings; starting over")

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
            obs_pred, resid, conv, iters = decode_config(wd, d, windows[c], tensors[c], dets, H_t, L_t)
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
            "t_bp": c[0], "t_osd": c[1], "merge": c[2], "col_order": c[3], **s,
            "ler": ler,
            "ler_per_round": 1 - (1 - ler) ** (1 / d.rounds) if ler < 1 else 1.0,
            "bp_conv_rate": s["bp_conv"] / n,
            "bp_avg_iter": s["bp_iter"] / n,
            "num_windows": len(windows[c]),
            "bulk_bp_H": list(bulk.bp.H.shape),
            "bulk_osd_H": list(bulk.osd.H.shape),
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
    ap.add_argument("-mw", "--merge_window", default="off",
                    help="merge columns made identical by window truncation: off, on, or off,on")
    ap.add_argument("-co", "--col_order", default="dem",
                    help=f"window column order for OSD tie-breaks, comma separated from {COL_ORDERS}")
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
    configs = parse_configs(args.windows, args.merge_window, args.col_order)
    print(f"DEM={d.H.shape} rounds={d.rounds} configs={[config_label(*c) for c in configs]}")
    if args.check:
        run_check(d, configs, args.check_shots, args.seed)
    else:
        if not args.decoding_yaml:
            raise ValueError("decoding needs -d <decoding yaml>")
        run_decode(args, d, configs)


if __name__ == "__main__":
    main()
