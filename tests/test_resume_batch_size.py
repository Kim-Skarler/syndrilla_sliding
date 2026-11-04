"""Resuming a checkpoint at another batch size, and run_sweep_v3 resuming after running out of memory."""
import glob
import os
import subprocess
import sys
from queue import Queue
from types import SimpleNamespace

import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import run_sweep_v3
from run_sweep_v3 import build_decoding_config, resolve_max_iter, set_checkpoint_target_error
from tools.bb_circuit_generator import generate_surface_interface_yaml


@pytest.fixture
def point(tmp_path):
    p = 0.02
    iface, deco = tmp_path / "interface.yaml", tmp_path / "decoding.yaml"
    generate_surface_interface_yaml(3, p, rounds=3, output_yaml_path=str(iface))
    cfg = build_decoding_config("relay_bp", resolve_max_iter("auto", "surface", 3), code_type="surface",
                                use_cpu=True, distance=3, error_rate=p)["decoding"]
    cfg["force_pytorch"] = True
    deco.write_text(yaml.safe_dump({"decoding": cfg, "decoder": cfg}, sort_keys=False))
    return SimpleNamespace(dir=tmp_path, iface=iface, deco=deco)


def _run(point, *flags):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONIOENCODING="utf-8")
    return subprocess.run([sys.executable, "-m", "syndrilla.main", f"-r={point.dir}", f"-i={point.iface}", f"-d={point.deco}", *flags],
                          env=env, capture_output=True, text=True, encoding="utf-8", errors="ignore")


def _result(point):
    path = glob.glob(str(point.dir / "result_phy_err_*.yaml"))[0]
    with open(path, encoding="utf-8") as f:
        return path, yaml.safe_load(f)


def test_resume_at_another_batch_size_keeps_the_counts(point):
    assert _run(point, "-bs=20", "-te=3").returncode == 0
    path, first = _result(point)
    batches = first["decoder_full"]["batch count"]
    errors = first["decoder_full"]["target error reached"]
    set_checkpoint_target_error(path, 8)

    assert _run(point, "-bs=5", "-te=8", f"-ckpt={path}").returncode != 0

    assert _run(point, "-bs=5", "-te=8", f"-ckpt={path}", "--allow-batch-size-change").returncode == 0
    _, second = _result(point)
    full, decoder = second["decoder_full"], second["decoder_0"]
    assert full["target error reached"] > errors
    assert full["batch count"] > batches
    assert decoder["sample count"] == 20 * batches + 5 * (full["batch count"] - batches)
    assert decoder["hx"]["logical error rate"] * decoder["sample count"] == pytest.approx(full["target error reached"])
    assert sum(decoder["iteration count"]) == decoder["sample count"]


def test_batch_size_change_is_refused_with_target_batch(point):
    got = _run(point, "-bs=5", "-tb=2", "--allow-batch-size-change")
    assert got.returncode != 0
    assert "--allow-batch-size-change cannot be combined with -tb" in got.stderr


def test_sweep_resumes_at_the_halved_batch_size_after_oom(tmp_path, monkeypatch):
    point_dir = tmp_path / "surface" / "d_3" / "bp" / "p_0.01"
    point_dir.mkdir(parents=True)
    ckpt = point_dir / "result_phy_err_0.01.yaml"
    ckpt.write_text(yaml.safe_dump({"decoder_full": {"batch size": 20, "target error": 5, "target error reached": 2}}))
    commands = []

    def fake_run(cmd, env=None, stdout=None, stderr=None):
        commands.append(cmd)
        if len(commands) == 1:
            stdout.write("CUDA out of memory\n")
            return SimpleNamespace(returncode=1)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(run_sweep_v3.subprocess, "run", fake_run)
    gpus = Queue()
    gpus.put("0")
    run_sweep_v3.worker_task(("surface", 3, 0.01, "bp", 20, 5, 9, str(tmp_path), False, "7layer", "stim_generated", None), gpus)

    assert "-bs=20" in commands[0] and "--allow-batch-size-change" not in commands[0]
    assert "-bs=10" in commands[1] and "--allow-batch-size-change" in commands[1]
    assert f"-ckpt={ckpt}" in commands[1]
    assert ckpt.exists() and not glob.glob(str(point_dir / "*.bak"))
