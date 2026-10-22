"""Build a BB memory-X circuit (X detectors only) and split its DEM into H, L and priors.

Schedules: "7layer_x" (Stim-convention 7-layer circuit, X detectors only; default) or "ibm"
(IBM decoder_setup.py circuit and noise).

Run scripts from the repo root, or import after putting the repo root on sys.path.
"""
import os
import sys
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import stim

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools.bb_circuit_generator import (  # noqa: E402
    BB_CODE_DICT,
    generate_bb_interface_yaml,
    get_bb_matchings,
    load_bb_matrices,
)

X_ONLY_SCHEDULES = ("7layer_x", "ibm")

BB_SEARCH_BASE = os.path.join(REPO_ROOT, "examples", "txt", "BB")


@dataclass
class DemData:
    code: str
    p: float
    rounds: int
    num_checks: int          # X checks per round (72 for BB144)
    circuit: stim.Circuit
    dem: stim.DetectorErrorModel
    H: sp.csr_matrix         # detectors x error classes
    L: sp.csr_matrix         # observables x error classes
    prior: np.ndarray        # per class flip probability
    det_round: np.ndarray    # round of each detector, 0..rounds (rounds = final MX readout)
    det_check: np.ndarray    # X check index of each detector
    col_rmin: np.ndarray     # first round a class touches, -1 if it touches no detector
    col_rmax: np.ndarray     # last round a class touches, -1 if it touches no detector


def build_circuit(code: str = "144", p: float = 0.003, rounds: int = None,
                  schedule: str = "7layer_x") -> stim.Circuit:
    """Memory-X circuit with X detectors only; `rounds` defaults to the code distance."""
    if schedule not in X_ONLY_SCHEDULES:
        raise ValueError(f"schedule <{schedule}> is not X-detector-only; use one of {X_ONLY_SCHEDULES}")
    _, _, _, bb_key, dist = load_bb_matrices(code, BB_SEARCH_BASE)
    if bb_key not in BB_CODE_DICT:
        raise ValueError(f"code <{code}> has no polynomial parameters in BB_CODE_DICT")
    return stim.Circuit(generate_bb_interface_yaml(code, p, rounds if rounds else dist, schedule=schedule,
                                                   search_base=BB_SEARCH_BASE))


def build_dem(code: str = "144", p: float = 0.003, rounds: int = None, schedule: str = "7layer_x") -> DemData:
    _, _, _, bb_key, dist = load_bb_matrices(code, BB_SEARCH_BASE)
    rounds = rounds if rounds else dist
    ell, m = BB_CODE_DICT[bb_key][:2]
    circuit = build_circuit(code, p, rounds, schedule)
    dem = circuit.detector_error_model(decompose_errors=False)

    coords = circuit.get_detector_coordinates()
    num_det = circuit.num_detectors
    det_check = np.array([int(coords[i][0]) for i in range(num_det)])
    det_round = np.array([int(coords[i][1]) for i in range(num_det)])

    h_rows, h_cols, l_rows, l_cols, prior = [], [], [], [], []
    for inst in dem.flattened():
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
    H = sp.csr_matrix((np.ones(len(h_rows), dtype=np.uint8), (h_rows, h_cols)), shape=(num_det, n_col))
    L = sp.csr_matrix((np.ones(len(l_rows), dtype=np.uint8), (l_rows, l_cols)),
                      shape=(circuit.num_observables, n_col))

    col_rmin = np.full(n_col, -1)
    col_rmax = np.full(n_col, -1)
    Hc = H.tocsc()
    for c in range(n_col):
        dets = Hc.indices[Hc.indptr[c]:Hc.indptr[c + 1]]
        if dets.size:
            col_rmin[c] = det_round[dets].min()
            col_rmax[c] = det_round[dets].max()

    return DemData(code=str(code), p=p, rounds=rounds, num_checks=ell * m, circuit=circuit, dem=dem,
                   H=H, L=L, prior=np.array(prior), det_round=det_round, det_check=det_check,
                   col_rmin=col_rmin, col_rmax=col_rmax)
