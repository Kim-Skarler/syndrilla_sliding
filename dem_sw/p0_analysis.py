"""P0 structure analysis of the BB memory-X IBM DEM (X detectors only).

Reports, per stim_dem_migration_plan.md §2.3 / §3:
- error classes per round (grouped by r_min), column weight, row degree, time span r_max - r_min
- classes with identical detectors but different observables; classes with observables but no detector
- time-translation invariance of the bulk rounds (detectors, observables and prior)
- Z_ell x Z_m symmetry: orbits of the detector sets under check-index shifts

Usage (repo root): python dem_sw/p0_analysis.py --code 144 --rounds 12 -p 0.003
"""
import argparse
from collections import Counter, defaultdict

import numpy as np

from dem_tools import BB_CODE_DICT, build_dem, get_bb_matchings, load_bb_matrices, BB_SEARCH_BASE


def fmt_hist(counter):
    return ", ".join(f"{k}:{v}" for k, v in sorted(counter.items()))


def column_supports(d):
    """Per class: (tuple of (check, round)), tuple of observables)."""
    Hc, Lc = d.H.tocsc(), d.L.tocsc()
    cols = []
    for c in range(d.H.shape[1]):
        dets = Hc.indices[Hc.indptr[c]:Hc.indptr[c + 1]]
        obs = Lc.indices[Lc.indptr[c]:Lc.indptr[c + 1]]
        cols.append((tuple(sorted((int(d.det_check[x]), int(d.det_round[x])) for x in dets)),
                     tuple(sorted(int(o) for o in obs))))
    return cols


def group_shift(ell, m):
    """shift[g][i]: check i moved by group element g = (da, db), with i = a*m + b."""
    a, b = np.divmod(np.arange(ell * m), m)
    return [((a + da) % ell) * m + (b + db) % m for da in range(ell) for db in range(m)]


def check_hx_invariant(code, ell, m):
    """Each matching (polynomial term) must commute with every group shift, on checks and data."""
    n2 = ell * m
    shifts = group_shift(ell, m)
    for edges in get_bb_matchings(*BB_CODE_DICT[code]):
        nb = {chk: dat for chk, dat in edges}
        for g in shifts:
            for chk, dat in edges:
                half, j = divmod(dat, n2)
                if nb[g[chk]] != half * n2 + g[j]:
                    return False
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--code", default="144")
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("-p", type=float, default=0.003)
    args = ap.parse_args()

    d = build_dem(args.code, args.p, args.rounds)
    _, _, _, bb_key, _ = load_bb_matrices(args.code, BB_SEARCH_BASE)
    ell, m = BB_CODE_DICT[bb_key][:2]
    R = d.rounds
    n_col = d.H.shape[1]
    cols = column_supports(d)
    weight = np.asarray(d.H.sum(axis=0)).ravel()
    row_deg = np.asarray(d.H.sum(axis=1)).ravel()
    span = d.col_rmax - d.col_rmin

    print(f"== code={args.code} rounds={R} p={args.p}: {d.H.shape[0]} detectors x {n_col} classes, "
          f"{d.L.shape[0]} observables")

    # --- per-round statistics ---
    print("\n-- per round (classes grouped by r_min) --")
    print("r_min  classes  /checks  weight(min/mean/max)  span hist")
    for r in range(R + 1):
        sel = d.col_rmin == r
        w = weight[sel]
        print(f"{r:5d}  {sel.sum():7d}  {sel.sum() / d.num_checks:7.2f}  "
              f"{w.min():3d}/{w.mean():5.2f}/{w.max():3d}         {fmt_hist(Counter(span[sel].tolist()))}")
    print(f"column weight hist (all): {fmt_hist(Counter(weight.tolist()))}")
    print(f"time span hist (all):     {fmt_hist(Counter(span[d.col_rmin >= 0].tolist()))}")
    print("row degree per detector round (min/mean/max):")
    for r in range(R + 1):
        deg = row_deg[d.det_round == r]
        print(f"  round {r:2d}: {deg.min()}/{deg.mean():.2f}/{deg.max()}")

    # --- observable bookkeeping ---
    by_dets = defaultdict(list)
    for c, (dets, obs) in enumerate(cols):
        by_dets[dets].append(c)
    conflicts = [cs for dets, cs in by_dets.items() if len(cs) > 1]
    undetectable = [c for c, (dets, obs) in enumerate(cols) if not dets and obs]
    empty = [c for c, (dets, obs) in enumerate(cols) if not dets and not obs]
    with_obs = sum(1 for _, obs in cols if obs)
    print("\n-- observables --")
    print(f"classes flipping >=1 observable: {with_obs}")
    print(f"detector sets shared by >1 class (same dets, different obs): {len(conflicts)} sets, "
          f"{sum(len(cs) for cs in conflicts)} classes; sizes {fmt_hist(Counter(len(cs) for cs in conflicts))}")
    print(f"classes with obs but no detector (must be 0): {len(undetectable)}")
    print(f"classes with neither (should be 0): {len(empty)}")
    assert not undetectable, "undetectable logical error class found"

    # --- time-translation invariance ---
    print("\n-- time-translation invariance --")
    def round_template(r):
        tpl = {}
        for c in np.nonzero(d.col_rmin == r)[0]:
            dets, obs = cols[c]
            key = (tuple((chk, rr - r) for chk, rr in dets), obs)
            tpl[key] = d.prior[c]
        return tpl
    templates = {r: round_template(r) for r in range(R + 1)}
    ref_r = R // 2
    ref = templates[ref_r]
    for r in range(R + 1):
        t = templates[r]
        same_keys = set(t) == set(ref)
        max_dp = max((abs(t[k] - ref[k]) / ref[k] for k in t if k in ref), default=float("nan"))
        print(f"  round {r:2d} vs round {ref_r}: classes {len(t):5d}, same support set={same_keys}, "
              f"common={len(set(t) & set(ref))}, max rel prior diff={max_dp:.2e}")

    # --- Z_ell x Z_m symmetry ---
    print("\n-- group symmetry Z_%d x Z_%d --" % (ell, m))
    print(f"H_X invariant under check/data shifts: {check_hx_invariant(bb_key, ell, m)}")
    shifts = group_shift(ell, m)
    det_sets = set(by_dets)
    seen, orbit_sizes, closed = set(), Counter(), True
    prior_by_dets = {dets: sorted(d.prior[c] for c in cs) for dets, cs in by_dets.items()}
    prior_mismatch = 0
    for dets in det_sets:
        if dets in seen:
            continue
        orbit = {tuple(sorted((int(g[chk]), rr) for chk, rr in dets)) for g in shifts}
        if not orbit <= det_sets:
            closed = False
        for o in orbit & det_sets:
            if not np.allclose(prior_by_dets.get(o, [])[:1], prior_by_dets[dets][:1], rtol=1e-9):
                prior_mismatch += 1
        seen |= orbit
        orbit_sizes[len(orbit)] += 1
    print(f"distinct detector sets: {len(det_sets)}; closed under the group: {closed}")
    print(f"orbit size hist: {fmt_hist(orbit_sizes)}  (total orbits {sum(orbit_sizes.values())})")
    print(f"detector sets whose prior differs from their orbit representative: {prior_mismatch}")
    bulk = d.col_rmin == ref_r
    print(f"bulk round {ref_r}: {bulk.sum()} classes -> {bulk.sum() / (ell * m):.2f} base types per round")
    print(f"distinct prior values (rel 1e-9): all={len(np.unique(np.round(np.log(d.prior), 9)))}, "
          f"bulk round={len(np.unique(np.round(np.log(d.prior[bulk]), 9)))}")


if __name__ == "__main__":
    main()
