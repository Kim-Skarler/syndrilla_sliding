"""The sweep circuits against the sizes and priors of the Relay-BP reference circuits."""
import os
import sys

import pytest
import stim

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tools.bb_circuit_generator import generate_bb_interface_yaml, generate_surface_interface_yaml


def _columns(circuit):
    dem = circuit.detector_error_model(decompose_errors=False, flatten_loops=True)
    return dem.num_detectors, [inst.args_copy()[0] for inst in dem.flattened() if inst.type == "error"]


def _noise_counts(circuit):
    counts = {}
    for inst in circuit.flattened():
        counts[inst.name] = counts.get(inst.name, 0) + len(inst.targets_copy())
    return counts


def test_surface_noise_keeps_the_noiseless_circuit():
    p = 0.003
    noisy = stim.Circuit(generate_surface_interface_yaml(5, p))
    clean = stim.Circuit.generated("surface_code:rotated_memory_x", distance=5, rounds=5)
    assert str(noisy.without_noise().flattened()) == str(clean.flattened())


def test_surface_noise_matches_the_reference_circuit_d11():
    # official rotated surface memory_X d=11, p=0.003: 1320 detectors, 24485 columns, prior sum 23.94950
    rows, priors = _columns(stim.Circuit(generate_surface_interface_yaml(11, 0.003)))
    assert rows == 1320
    assert len(priors) == 24485
    assert sum(priors) == pytest.approx(23.94950, abs=1e-5)


def test_surface_noise_layout():
    circuit = stim.Circuit(generate_surface_interface_yaml(3, 0.003))
    h_noise = False
    flat = list(circuit.flattened())
    for prev, inst in zip(flat, flat[1:]):
        h_noise |= prev.name == "H" and inst.name == "DEPOLARIZE1" and inst.targets_copy() == prev.targets_copy()
    assert not h_noise
    assert all(inst.name != "X_ERROR" or prev.name == "R" for prev, inst in zip(flat, flat[1:]))
    final_readout = [inst for inst in flat if inst.name == "MX"]
    assert final_readout and all(not inst.gate_args_copy() for inst in final_readout)
    assert _noise_counts(circuit)["DEPOLARIZE1"] > 0


@pytest.mark.parametrize("code, rounds, rows, cols", [("72", 6, 432, 16200), ("144", 12, 1728, 67824)])
def test_ibm_xyz_has_the_size_of_the_reference_circuit(code, rounds, rows, cols):
    got_rows, priors = _columns(stim.Circuit(generate_bb_interface_yaml(code, 0.003, rounds, schedule="ibm_xyz")))
    assert (got_rows, len(priors)) == (rows, cols)


def test_ibm_xyz_only_adds_detectors_to_ibm():
    ibm = stim.Circuit(generate_bb_interface_yaml("72", 0.003, 6, schedule="ibm"))
    xyz = stim.Circuit(generate_bb_interface_yaml("72", 0.003, 6, schedule="ibm_xyz"))
    drop = lambda c: [inst for inst in c.flattened() if inst.name != "DETECTOR"]
    assert drop(ibm) == drop(xyz)
    assert xyz.num_detectors > ibm.num_detectors


def test_surface_stim_generated_noise_is_the_existing_results_noise():
    p = 0.003
    got = generate_surface_interface_yaml(5, p, noise="stim_generated")
    assert got == str(stim.Circuit.generated(
        "surface_code:rotated_memory_x", distance=5, rounds=5, after_clifford_depolarization=p,
        after_reset_flip_probability=p, before_measure_flip_probability=p, before_round_data_depolarization=p))
    with pytest.raises(ValueError):
        generate_surface_interface_yaml(5, p, noise="unknown")
