import pytest
import stim
import torch

from syndrilla.decoder.relay_bp.relay_bp import create as create_relay_bp
from syndrilla.interface import create_interface

LEGS = 9
SOLUTIONS = 3
BATCH = 2


def _problem():
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_x", distance=3, rounds=3, after_clifford_depolarization=0.01,
        after_reset_flip_probability=0.01, before_measure_flip_probability=0.01,
    )
    interface = create_interface(cfg=dict(backend="stim", circuit=str(circuit)))
    llr0 = interface.error_model._prior_llr.unsqueeze(0).repeat(BATCH, 1)
    synd = torch.zeros(BATCH, circuit.num_detectors, dtype=torch.float64)
    return interface.matrix_bundle, dict(synd=synd, llr0=llr0)


def _cfg(device):
    return dict(dtype="float64", device={"device_type": device, "device_idx": 0}, check_type="hx",
                legs=LEGS, solution=SOLUTIONS, rebatch_opt=False)


def test_stops_after_the_requested_solutions():
    # the empty correction solves a zero syndrome in every leg
    bundle, io_dict = _problem()
    decoder = create_relay_bp(_cfg("cpu"), bundle=bundle)
    out = decoder(io_dict)
    assert decoder.r == SOLUTIONS
    assert out["converge"].tolist() == [1] * BATCH


def test_a_leg_that_finds_nothing_is_not_a_solution():
    bundle, io_dict = _problem()
    decoder = create_relay_bp(_cfg("cpu"), bundle=bundle)
    real = decoder.syndrome_estimation
    # only the first leg can converge, so the second and third solutions never come
    decoder.syndrome_estimation = lambda e_v: real(e_v) if decoder.r == 1 else real(e_v) + 1
    out = decoder(io_dict)
    assert decoder.r == LEGS
    assert out["converge"].tolist() == [1] * BATCH


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_leg_that_finds_nothing_is_not_a_solution():
    from syndrilla.decoder.relay_bp.relay_bp_cuda import create as create_relay_bp_cuda

    bundle, io_dict = _problem()
    decoder = create_relay_bp_cuda(_cfg("cuda"), bundle=bundle)
    ext = decoder._ext

    class FirstLegOnly:
        def __getattr__(self, name):
            return getattr(ext, name)

        def convergence_flag_update(self, mismatch, leg_iters, converges, i):
            if decoder.r > 1:
                mismatch.fill_(1)
            ext.convergence_flag_update(mismatch, leg_iters, converges, i)

    decoder._ext = FirstLegOnly()
    out = decoder(io_dict)
    assert decoder.r == LEGS
    assert out["converge"].tolist() == [1] * BATCH
