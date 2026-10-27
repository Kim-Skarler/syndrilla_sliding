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

End of the run (`-fc`), both keep the hardware window at T_BP rounds:
- round: slide to t = D - 1 and commit one round per window. The tail windows shrink, which
  is the same decode as padding them to T_BP rounds with zero syndrome and saturated LLRs
  (no class touches the padded rows, so BP and OSD-0 see two disconnected parts).
- all ("+flush"): stop at the last full window t = D - T_BP and commit every remaining class
  there, with OSD-0 on the whole BP window (T_OSD is raised to T_BP for that window only).

Window merge and column order (§2.5), both switchable so their effect can be measured:
- merge on: classes whose rows inside a window part are identical (a truncation side effect)
  become one column with prior P(odd number fire) = (1 - prod(1 - 2p)) / 2. When the OSD part
  merges columns BP kept apart, their BP posteriors combine by the tanh rule. A committed
  merged column is restored to its member with the largest prior.
- column order sets the local index of each window column, which OSD-0's stable sort uses to
  break LLR ties: dem (Stim order), prior (largest prior first), random (fixed shuffle).

Every combination of `-w`, `-mw`, `-co` and `-fc` decodes the same Stim samples. In `-w`, "0" is the
baseline (one window holding the whole DEM), "T" is SW(T) with T_BP = T_OSD = T, and "a,b" is
SW(T_BP=a, T_OSD=b).

  python -m syndrilla.main_sliding -r=<dir> -i=interface.yaml -d=decoding.yaml -w="0;1;2;3" -mw=off,on -te=100
  python -m syndrilla.main_sliding -i=interface.yaml --check     # window bookkeeping test, no torch

Golden mode (`--golden <dir>`) decodes one batch of `-bs` samples with one window config and
writes the OSD-0 calls of the windows whose OSD H equals the bulk window's as test patterns
for the OSD hardware, without touching any result file (see GoldenDump).
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
FINAL_COMMITS = ("round", "all")


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


def parse_configs(windows_spec, merge_spec, order_spec, final_spec="round"):
    """Every (T_BP, T_OSD, merge, column order, flush) combination. The baseline runs unmerged
    and unflushed only: Stim already merged identical DEM columns, so a whole-DEM merge
    changes nothing, and its one window already commits everything. SW(1) never flushes:
    its last full window is the last round, so the flush is the same decode."""
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
    finals = [f.strip() for f in final_spec.split(",")]
    for f in finals:
        if f not in FINAL_COMMITS:
            raise ValueError(f"final_commit entries are {FINAL_COMMITS}, got <{f}>")
    configs = []
    for (t_bp, t_osd), m, o, f in itertools.product(parse_windows(windows_spec), merges, orders, finals):
        cfg = (t_bp, t_osd, m and t_bp != 0, o, f == "all" and t_bp > 1)
        if cfg not in configs:
            configs.append(cfg)
    return configs


def config_label(t_bp, t_osd, merge=False, order="dem", flush=False):
    if t_bp == 0:
        label = "baseline"
    else:
        label = f"SW({t_bp})" if t_bp == t_osd else f"SW({t_bp},{t_osd})"
    if merge:
        label += "+merge"
    if order != "dem":
        label += f"+{order}"
    if flush:
        label += "+flush"
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

    def __init__(self, d, t, t_bp, t_osd, merge, order, seed, final=False):
        D = d.rounds + 1
        rng = np.random.default_rng(seed + 1000 * t)
        if final:  # flush: OSD on the whole BP window, every class in it committed
            t_osd = t_bp
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
            commit_round = None if final else t

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


def build_windows(d, t_bp, t_osd, merge, order, flush, seed):
    if t_bp == 0:
        return [Window(d, 0, 0, 0, merge, order, seed)]
    if not flush:
        return [Window(d, t, t_bp, t_osd, merge, order, seed) for t in range(d.rounds + 1)]
    last = max(0, d.rounds + 1 - t_bp)  # last window that still holds T_BP rounds
    return [Window(d, t, t_bp, t_osd, merge, order, seed, final=t == last) for t in range(last + 1)]


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


def decode_config(wd, d, windows, tensors, dets, H_t, L_t, dump=None):
    """Decode one batch with one config. dump(t, synd, llr, e, conv) receives every OSD-0
    call: window index, OSD syndrome, OSD-column LLR, OSD estimate and BP convergence."""
    import torch

    B = dets.shape[0]
    resid = dets.clone()
    obs_pred = torch.zeros((B, d.L.shape[0]), dtype=torch.float32, device=wd.device)
    bp_conv = torch.zeros(B, device=wd.device)
    bp_iter = torch.zeros(B, device=wd.device)
    for t, (w, T) in enumerate(zip(windows, tensors)):
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
            if dump is not None:
                dump(t, resid[:, T.rows_osd], llr_osd, e_osd, conv)
        e = e_osd[:, T.commit_pos].float()
        resid = ((resid.float() + e @ H_t[T.commit_rep]) % 2).to(resid.dtype)
        obs_pred = (obs_pred + e @ L_t[T.commit_rep]) % 2
    return obs_pred, resid, bp_conv / len(windows), bp_iter / len(windows)


class GoldenDump:
    """OSD-0 test patterns for the hardware, one line per OSD call (a BP-unconverged sample),
    all on the bulk window's OSD H. A window with another OSD H (the tail of a round run)
    is skipped, or with tail=True embedded in it: each of its columns sits on the bulk column with the same rows, the
    other bulk columns are invalid (largest fixed-point LLR, sorted after every valid
    column), and the golden result is OSD-0 in that order, compared with the window's own
    decode in golden_meta.yaml. Files, those of the earlier OSD testbench first:
        sorting_llr_input_hex.txt  N LLRs, two's complement Q(int_width).(frac_width) of
                                   int_width + frac_width + 1 bits, column 0 first
        sorting_syndrome.txt       the M syndrome bits as one hex number, bit i = row i
        sorting_golden_idx.txt     the N column indices in OSD order (the ranking_method's
                                   stable ascending key, invalid columns last), hex
        sorting_osd_result.txt     the N-bit OSD estimate as one hex number, bit j = column j
        sorting_col_valid.txt      the N column valid bits as one hex number, bit j = column j
                                   (all ones except in an embedded window)
        pattern_info.txt           pattern, sample, window t, syndrome weight, embedded per line
        osd_H.txt                  the bulk OSD H, one row per line as a hex number, bit j = column j
        osd_embed_t<t>.txt         for an embedded window t, the bulk column of each of its columns
        golden_meta.yaml           formats, sizes, embedding counts and the run settings"""

    def __init__(self, out_dir, windows, ranking, int_width, frac_width, meta, tail=False):
        bulk = windows[len(windows) // 2].osd.H.tocsc()
        self.H = bulk.toarray().astype(np.uint8)
        by_rows = defaultdict(list)
        for j in range(bulk.shape[1]):
            by_rows[tuple(bulk.indices[bulk.indptr[j]:bulk.indptr[j + 1]])].append(j)
        # embed[t]: bulk column of each column of window t, None when its OSD H is the bulk one
        self.embed = []
        for t, w in enumerate(windows):
            h = w.osd.H.tocsc()
            if h.shape == bulk.shape and (h != bulk).nnz == 0:
                self.embed.append(None)
                continue
            if h.shape[0] != bulk.shape[0]:
                raise ValueError(f"window {t}: OSD rows {h.shape[0]} differ from the bulk window's {bulk.shape[0]}")
            free = {k: list(v) for k, v in by_rows.items()}
            cols = []
            for j in range(h.shape[1]):
                slots = free.get(tuple(h.indices[h.indptr[j]:h.indptr[j + 1]]))
                if not slots:
                    raise ValueError(f"window {t}: column {j} has no bulk column with the same rows")
                cols.append(slots.pop(0))
            self.embed.append(np.array(cols))
        self.out_dir, self.ranking, self.iw, self.fw, self.meta = out_dir, ranking, int_width, frac_width, meta
        self.rows = []  # (sample, t, synd, llr, order, e, valid)
        self.embedded = self.embedded_differ = self.skipped = 0
        self.tail = tail

    def __call__(self, t, synd, llr, e, conv):
        import torch
        from syndrilla.decoder.osd_0.ranking import ranking_key

        idx = (conv == 0).nonzero(as_tuple=True)[0]
        if not idx.numel():
            return
        llr_s, e_s, synd_s = llr[idx], e[idx], synd[idx]
        cols = self.embed[t]
        if cols is not None and not self.tail:
            self.skipped += idx.numel()
            return
        N = self.H.shape[1]
        valid = np.ones(N, dtype=np.uint8)
        if cols is None:
            order = torch.sort(ranking_key(llr_s, self.ranking, self.iw, self.fw), dim=1, stable=True).indices
            order = order.cpu().numpy()
            e_np = e_s.cpu().numpy().astype(np.uint8)
        else:
            # the window's columns on their bulk columns; every other column is invalid: the
            # largest LLR and sorted after every valid column (a fixed-point LLR alone cannot
            # do that, since valid columns reach the largest value and tie with it)
            valid[:] = 0
            valid[cols] = 1
            top = ((1 << (self.iw + self.fw)) - 1) / (1 << self.fw)
            full = llr_s.new_full((idx.numel(), N), top)
            full[:, torch.as_tensor(cols, device=full.device)] = llr_s
            # a copy: the standard key is the LLR tensor itself
            key = ranking_key(full, self.ranking, self.iw, self.fw).clone()
            key[:, torch.as_tensor(valid == 0, device=key.device)] = key.max() + 1
            order = torch.sort(key, dim=1, stable=True).indices.cpu().numpy()
            synd_np = synd_s.cpu().numpy().astype(np.uint8)
            e_np = np.stack([self._osd0(order[k], synd_np[k]) for k in range(idx.numel())])
            back = np.zeros_like(e_np)
            back[:, cols] = e_s.cpu().numpy().astype(np.uint8)
            self.embedded += idx.numel()
            self.embedded_differ += int((e_np != back).any(axis=1).sum())
            llr_s = full
        for k, b in enumerate(idx.tolist()):
            self.rows.append((b, t, synd_s[k].cpu().numpy().astype(np.uint8), llr_s[k].cpu().numpy(),
                              order[k], e_np[k], valid))

    def _osd0(self, order, s):
        """OSD-0 on the bulk H in a given column order: the first independent columns, solved
        on them (the same estimate as osd_0 for a syndrome in their span)."""
        M, N = self.H.shape
        A = np.concatenate([self.H[:, order], s[:, None]], axis=1)
        piv, r = [], 0
        for j in range(N):
            rows = np.nonzero(A[r:, j])[0]
            if not rows.size:
                continue
            p = r + rows[0]
            A[[r, p]] = A[[p, r]]
            for q in np.nonzero(A[:, j])[0]:
                if q != r:
                    A[q] ^= A[r]
            piv.append(j)
            r += 1
            if r == M:
                break
        e = np.zeros(N, dtype=np.uint8)
        for k, j in enumerate(piv):
            e[order[j]] = A[k, -1]
        return e

    @staticmethod
    def _hex(bits):
        """bit i of the number = bits[i], zero padded to ceil(len / 4) hex digits."""
        value = int("".join("1" if b else "0" for b in bits[::-1]) or "0", 2)
        return f"{value:0{(len(bits) + 3) // 4}x}"

    def write(self):
        os.makedirs(self.out_dir, exist_ok=True)
        M, N = self.H.shape
        bits = self.iw + self.fw
        limit = 1 << bits
        mask = (1 << (bits + 1)) - 1
        llr_digits, idx_digits = (bits + 1 + 3) // 4, max(1, ((N - 1).bit_length() + 3) // 4)
        names = ("sorting_llr_input_hex.txt", "sorting_syndrome.txt", "sorting_golden_idx.txt",
                 "sorting_osd_result.txt", "sorting_col_valid.txt", "pattern_info.txt")
        files = {name: open(os.path.join(self.out_dir, name), "w", encoding="utf-8", newline="\n")
                 for name in names}
        bad = 0
        try:
            files["pattern_info.txt"].write("# pattern sample window_t syndrome_weight embedded\n")
            for n, (b, t, synd, llr, order, e, valid) in enumerate(self.rows):
                v = np.clip(np.round(llr * (1 << self.fw)), -limit, limit - 1).astype(np.int64)
                if not np.array_equal(v / (1 << self.fw), llr):
                    raise ValueError("OSD LLR is not on the fixed-point grid: golden mode needs a "
                                     "quantized BP with the same int_width / frac_width")
                files["sorting_llr_input_hex.txt"].write(" ".join(f"{x & mask:0{llr_digits}x}" for x in v) + "\n")
                files["sorting_syndrome.txt"].write(self._hex(synd) + "\n")
                files["sorting_golden_idx.txt"].write(" ".join(f"{x:0{idx_digits}x}" for x in order) + "\n")
                files["sorting_osd_result.txt"].write(self._hex(e) + "\n")
                files["sorting_col_valid.txt"].write(self._hex(valid) + "\n")
                files["pattern_info.txt"].write(f"{n} {b} {t} {int(synd.sum())} {int(not valid.all())}\n")
                bad += int(((self.H.astype(np.int64) @ e) % 2 != synd).any())
        finally:
            for f in files.values():
                f.close()
        with open(os.path.join(self.out_dir, "osd_H.txt"), "w", encoding="utf-8", newline="\n") as f:
            for row in self.H:
                f.write(self._hex(row) + "\n")
        for t, cols in enumerate(self.embed):
            if cols is not None:
                with open(os.path.join(self.out_dir, f"osd_embed_t{t}.txt"), "w", encoding="utf-8",
                          newline="\n") as f:
                    f.write(" ".join(f"{c:0{idx_digits}x}" for c in cols) + "\n")
        info = {**self.meta, "patterns": len(self.rows), "patterns_not_matching_syndrome": bad,
                "M": int(M), "N": int(N), "ranking_method": self.ranking, "int_width": self.iw,
                "frac_width": self.fw, "llr_hex_digits": llr_digits, "idx_hex_digits": idx_digits,
                "tail_windows": [t for t, c in enumerate(self.embed) if c is not None],
                "tail_embedded": self.tail, "tail_osd_calls_skipped": self.skipped,
                "embedded_patterns": self.embedded,
                "embedded_patterns_differing_from_window_decode": self.embedded_differ}
        with open(os.path.join(self.out_dir, "golden_meta.yaml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(info, f, sort_keys=False)
        tail = (f"{self.embedded} tail calls embedded, {self.embedded_differ} of them differing from the "
                f"window decode" if self.tail else f"{self.skipped} tail calls skipped")
        print(f"[golden] {len(self.rows)} patterns ({tail}; {bad} not matching their syndrome) -> {self.out_dir}")


def run_golden(args, d, configs, wd, windows, tensors, H_t, L_t, device, meta):
    """One batch of -bs samples, one config, its OSD-0 calls written by GoldenDump."""
    import torch

    if len(configs) != 1:
        raise ValueError(f"golden mode takes one window config, got {[config_label(*c) for c in configs]}")
    c = configs[0]
    osd_cfg = wd.stages["osd"].get("config") or {}
    bp_cfg = wd.stages["bp"].get("config") or {}
    iw = osd_cfg.get("int_width", bp_cfg.get("int_width"))
    fw = osd_cfg.get("frac_width", bp_cfg.get("frac_width"))
    if iw is None or fw is None:
        raise ValueError("golden mode needs a fixed-point format: int_width / frac_width in the BP or OSD config")
    dump = GoldenDump(args.golden, windows[c], osd_cfg.get("ranking_method", "standard"), int(iw), int(fw),
                      {**meta, "config": config_label(*c), "batch_size": args.batch_size}, args.golden_tail)
    dets_np, obs_np, _ = d.dem.compile_sampler(seed=args.seed).sample(args.batch_size)
    dets = torch.as_tensor(dets_np.astype(np.uint8), device=device)
    obs = torch.as_tensor(obs_np.astype(np.float32), device=device)
    obs_pred, _, _, _ = decode_config(wd, d, windows[c], tensors[c], dets, H_t, L_t, dump)
    dump.write()
    print(f"[golden] {config_label(*c)}: {int((obs_pred != obs).any(dim=1).sum())} logical errors "
          f"in {args.batch_size} samples")


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
            "final_commit": args.final_commit, "seed": args.seed, "dem_shape": list(d.H.shape)}
    if args.golden:
        return run_golden(args, d, configs, wd, windows, tensors, H_t, L_t, device, meta)
    keys = ("shots", "errors", "bp_conv", "bp_iter", "resid_nonzero", "time_s")
    stats = {c: dict.fromkeys(keys, 0) for c in configs}
    kept = {}  # configs of the result file this run does not decode, written back unchanged
    if os.path.exists(out_path):
        try:
            with open(out_path, encoding="utf-8") as f:
                prev = yaml.safe_load(f) or {}
        except yaml.YAMLError:
            prev = {}
        if not args.fresh and same_run(prev.get("meta") or {}, meta):
            by_label = {config_label(*c): c for c in configs}
            for label, r in (prev.get("configs") or {}).items():
                if label in by_label:
                    stats[by_label[label]] = {k: r[k] for k in keys}
                else:
                    kept[label] = r
            new = [label for label in by_label if label not in (prev.get("configs") or {})]
            print(f"[resume] {out_path}" + (f", new configs from shot 0: {', '.join(new)}" if new else ""))
        else:
            # other code, noise or decoder settings: keep the old result and start over
            backup = f"{out_path}.{time.strftime('%Y%m%d-%H%M%S')}.bak"
            os.replace(out_path, backup)
            reason = "--fresh" if args.fresh else "written with other settings"
            print(f"[fresh] {out_path} {reason}; kept as {os.path.basename(backup)}, starting over")

    def shots_done():
        return min(s["shots"] for s in stats.values())

    batch = 0
    while (min(s["errors"] for s in stats.values()) < args.target_error
           and (args.max_shots <= 0 or shots_done() < args.max_shots)):
        # The batch at shot offset o is sampled with seed + o, so a config added later
        # replays the batches the others already decoded (the same samples while the batch
        # size is unchanged) and the others wait until it has caught up.
        offset = shots_done()
        sampler = d.dem.compile_sampler(seed=args.seed + offset)
        dets_np, obs_np, _ = sampler.sample(args.batch_size)
        dets = torch.as_tensor(dets_np.astype(np.uint8), device=device)
        obs = torch.as_tensor(obs_np.astype(np.float32), device=device)
        for c in configs:
            if stats[c]["shots"] != offset:
                continue
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
        save(out_path, meta, stats, windows, d, kept)
        print(f"[batch {batch}] shots={shots_done()} " +
              " ".join(f"{config_label(*c)}={stats[c]['errors']}" for c in configs), flush=True)
    save(out_path, meta, stats, windows, d, kept)
    print(f"[saved] {out_path}")


# meta keys that only pick the config list; the rest must match for a result file to be resumed
CONFIG_LIST_KEYS = ("windows", "merge_window", "col_order", "final_commit")


def same_run(prev, meta):
    drop = lambda m: {k: v for k, v in m.items() if k not in CONFIG_LIST_KEYS}
    return drop(prev) == drop(meta)


def save(path, meta, stats, windows, d, kept=None):
    res = dict(kept or {})
    for c, s in stats.items():
        n = max(s["shots"], 1)
        ler = s["errors"] / n
        bulk = windows[c][len(windows[c]) // 2]
        res[config_label(*c)] = {
            "t_bp": c[0], "t_osd": c[1], "merge": c[2], "col_order": c[3], "flush": c[4], **s,
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
    ap.add_argument("-fc", "--final_commit", default="round",
                    help="end of run: round (slide to the end), all (flush the last full window), or round,all")
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
    ap.add_argument("--golden", default=None,
                    help="write OSD hardware test patterns of one batch (-bs samples, one config) to this "
                         "directory and stop; no result file is read or written")
    ap.add_argument("--golden_tail", action="store_true",
                    help="golden: also write the tail window, embedded in the bulk OSD H with invalid columns")
    return ap.parse_args()


def main():
    args = parse_commandline_args()
    d = Dem(load_circuit(args.interface_yaml))
    configs = parse_configs(args.windows, args.merge_window, args.col_order, args.final_commit)
    print(f"DEM={d.H.shape} rounds={d.rounds} configs={[config_label(*c) for c in configs]}")
    if args.check:
        run_check(d, configs, args.check_shots, args.seed)
    else:
        if not args.decoding_yaml:
            raise ValueError("decoding needs -d <decoding yaml>")
        run_decode(args, d, configs)


if __name__ == "__main__":
    main()
