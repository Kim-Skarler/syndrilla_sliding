"""P0 sanity check for the BB memory-X IBM circuit.

1. Full-length circuit: the DEM builds (no non-deterministic detector or observable),
   detector count is num_checks * (rounds + 1), observable count is k.
2. Circuit distance upper bound, IBM's heuristic (BivariateBicycleCodes/circuit_distance.py):
   for a random nonzero combination l of logical rows, BP-OSD-CS solves [H; l] e = [0; 1],
   and the smallest |e| found over all trials bounds the circuit distance from above.

Usage (repo root): python dem_sw/p0_sanity.py --code 144 --rounds 12 --dist-rounds 1 2 --trials 2000
"""
import argparse
import time

import numpy as np
import scipy.sparse as sp

from dem_tools import build_dem


def check_full(code, p, rounds, schedule):
    t = time.time()
    d = build_dem(code, p, rounds, schedule)  # raises if any detector/observable is non-deterministic
    expect_det = d.num_checks * (d.rounds + 1)
    print(f"[full] code={code} rounds={d.rounds} p={p}: detectors={d.H.shape[0]} (expect {expect_det}), "
          f"observables={d.L.shape[0]}, error classes={d.H.shape[1]} ({time.time() - t:.1f}s)")
    assert d.H.shape[0] == expect_det, "detector count mismatch"
    return d


def distance_upper_bound(code, p, rounds, trials, osd_order, seed, schedule):
    from ldpc import BpOsdDecoder

    d = build_dem(code, p, rounds, schedule)
    rng = np.random.default_rng(seed)
    L = d.L.toarray().astype(np.uint8)
    syndrome = np.zeros(d.H.shape[0] + 1, dtype=np.uint8)
    syndrome[-1] = 1
    best = d.H.shape[1]
    t = time.time()
    for _ in range(trials):
        coeff = rng.integers(0, 2, size=L.shape[0])
        if not coeff.any():
            continue
        logical_row = (coeff @ L) % 2
        H1 = sp.vstack([d.H, sp.csr_matrix(logical_row)]).tocsr()
        dec = BpOsdDecoder(H1, error_rate=0.01, max_iter=10000, bp_method="ms", ms_scaling_factor=0,
                           osd_method="osd_cs", osd_order=osd_order)
        e = dec.decode(syndrome)
        if ((H1 @ e) % 2 == syndrome).all():
            best = min(best, int(e.sum()))
    print(f"[distance] code={code} rounds={rounds}: upper bound d_C <= {best} "
          f"({trials} trials, osd_cs order {osd_order}, {time.time() - t:.1f}s)")
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--code", default="144")
    ap.add_argument("--schedule", default="7layer_x", choices=["7layer_x", "ibm"])
    ap.add_argument("-p", type=float, default=0.003, help="only sets priors; structure does not depend on p")
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--dist-rounds", type=int, nargs="*", default=[1, 2])
    ap.add_argument("--trials", type=int, default=2000)
    ap.add_argument("--osd-order", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    check_full(args.code, args.p, args.rounds, args.schedule)
    for r in args.dist_rounds:
        distance_upper_bound(args.code, args.p, r, args.trials, args.osd_order, args.seed, args.schedule)


if __name__ == "__main__":
    main()
