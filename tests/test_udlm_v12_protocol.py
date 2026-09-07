"""Small synthetic receipts exercise the fixed V12 acceptance/publication gate."""

import copy
import hashlib
import json
import shutil
from pathlib import Path

import pytest
import torch

import test_udlm_mask_rich_benchmark as prior_cases
from scripts.udlm import materialize_v12_protocol as materializer


def write(root, relative, value):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = materializer.engine.encode(value)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


@pytest.fixture
def study(tmp_path, monkeypatch):
    """Use real pinned design/history, but explicitly synthetic new receipts."""
    files = [materializer.DESIGN, materializer.TEMPLATE, materializer.FAILURE]
    files.extend(
        f"experiments/udlm/protocols/engineering_v12_configs/{name}.yaml"
        for name in materializer.CONFIG_SHAS
    )
    template = json.loads((materializer.ROOT / materializer.TEMPLATE).read_bytes())
    history = template["training"]
    files.extend(
        reference["relative_path"]
        for reference in (
            history["original_failed_campaign"],
            history["original_v8_protocol"],
            history["arms"]["CT"]["post_exit_audit"],
            history["arms"]["CT"]["terminal_receipt"],
        )
    )
    files.extend(spec["protocol"] for spec in materializer.ARMS.values())
    for relative in files:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(materializer.ROOT / relative, target)
    original = json.loads(
        (
            materializer.ROOT
            / materializer.ARMS["E_CE"]["directory"]
            / "terminal_manifest.json"
        ).read_bytes()
    )
    records, refs = {}, {}
    for arm, spec in materializer.ARMS.items():
        terminal = copy.deepcopy(original)
        plan = terminal["plan"]
        plan["protocol"] = json.loads((tmp_path / spec["protocol"]).read_bytes())
        plan["protocol_sha256"] = spec["protocol_sha256"]
        plan["config"]["callback"]["dirpath"] = str(
            tmp_path / spec["directory"] / "checkpoints"
        )
        if arm == "MASK_CE":
            plan["config"]["training"]["udlm"].update(
                prior_variant=spec["prior_variant"], mask_mixture_weight=0.9
            )
        plan["config_sha256"] = materializer.digest(plan["config"])
        terminal["source"] = {"head": spec["source"], "upstream": spec["source"]}
        terminal["checkpoint"].update(
            relative_path=spec["directory"] + "/checkpoints/1000.ckpt",
            sha256=("a" if arm == "E_CE" else "b") * 64,
            size_bytes=1234,
            finite_tensor_count=7,
        )
        if arm == "MASK_CE":
            terminal["checkpoint"]["udlm_prior_metadata_sha256"] = spec[
                "prior_metadata_sha256"
            ]
        request = {"plan": plan, "source": terminal["source"]}
        launch = {**request, "input_checkpoint": {"sha256": materializer.MDLM_SHA}}
        terminal["request_sha256"] = write(
            tmp_path, spec["directory"] + "/request_manifest.json", request
        )
        terminal["launch_sha256"] = write(
            tmp_path, spec["directory"] + "/launch_manifest.json", launch
        )
        refs[arm] = write(
            tmp_path, spec["directory"] + "/terminal_manifest.json", terminal
        )
        records[arm] = terminal
    monkeypatch.setattr(materializer, "CONTROL_CP_SHA", "a" * 64)
    monkeypatch.setattr(materializer, "CONTROL_TERMINAL_SHA", refs["E_CE"])

    def inspected(root, terminal, arm):
        spec = materializer.ARMS[arm]
        return {
            "diffusion_type": "udlm",
            "udlm_prior_variant": spec["prior_variant"],
            "udlm_prior_metadata": {"fixture": "synthetic prior identity"},
            "udlm_denoiser_metadata": dict(
                materializer.benchmark.UDLM_DENOISER_METADATA
            ),
        }

    monkeypatch.setattr(materializer, "inspect_checkpoint", inspected)
    return tmp_path, records, refs


def build(study):
    root, _, refs = study
    return materializer.build_protocol(
        input_root=root, config_root=root, v11b_terminal_sha256=refs["MASK_CE"]
    )


def test_fixed_prior_protocol_has_all_four_entries_and_no_ct_objective(study):
    protocol = build(study)
    assert protocol["seeds"] == [2000, 2001]
    assert protocol["nfe"] == 128 and protocol["num_samples"] == 100
    assert len(protocol["entries"]) == 4
    assert set(protocol["training"]["arms"]) == {"E_CE", "MASK_CE"}
    assert "prior_variant" not in protocol["training"]
    assert "prior_metadata_sha256" not in protocol["training"]
    assert "objective_comparison" not in protocol["design"]
    assert set(protocol["design"]["prior_comparison"]) == {"primary", "secondary"}
    assert {entry["parameterization"] for entry in protocol["entries"]} == {
        "x0_denoiser"
    }
    assert {entry["prior_variant"] for entry in protocol["entries"]} == {
        "empirical_frequency",
        "mask_rich_empirical",
    }
    assert (
        protocol["training"]["original_v11_failure"]["sha256"]
        == materializer.FAILURE_SHA
    )
    assert (
        protocol["training"]["original_v8_ct_context"]["controller_status"] == "failed"
    )
    assert not (study[0] / materializer.OUTPUT).exists()


@pytest.mark.parametrize(
    "mutation",
    [
        "failed",
        "no_checkpoint",
        "return_code",
        "source",
        "request_hash",
        "prior_hash",
        "step",
        "config",
    ],
)
def test_unaccepted_or_relabelled_treatment_never_reaches_checkpoint_inspection(
    study, monkeypatch, mutation
):
    root, records, refs = study
    terminal = records["MASK_CE"]
    if mutation == "failed":
        terminal["status"] = "failed"
    elif mutation == "no_checkpoint":
        terminal["checkpoint"] = None
    elif mutation == "return_code":
        terminal["training_return_code"] = 7
    elif mutation == "source":
        terminal["source"]["head"] = "c" * 40
    elif mutation == "request_hash":
        terminal["request_sha256"] = "c" * 64
    elif mutation == "prior_hash":
        terminal["checkpoint"]["udlm_prior_metadata_sha256"] = "c" * 64
    elif mutation == "step":
        terminal["checkpoint"]["global_step"] = 999
    else:
        terminal["plan"]["config"]["seed"] = 1501
    refs["MASK_CE"] = write(
        root,
        materializer.ARMS["MASK_CE"]["directory"] + "/terminal_manifest.json",
        terminal,
    )
    calls = []
    monkeypatch.setattr(materializer, "inspect_checkpoint", lambda *a: calls.append(a))
    with pytest.raises((ValueError, TypeError)):
        build(study)
    assert calls == []
    assert not (root / materializer.OUTPUT).exists()


def test_frozen_sampling_file_drift_is_rejected(study):
    path = (
        study[0]
        / "experiments/udlm/protocols/engineering_v12_configs/mask_ce_t100.yaml"
    )
    path.write_bytes(path.read_bytes().replace(b"num_steps: 128", b"num_steps: 512"))
    with pytest.raises(ValueError, match="digest mismatch"):
        build(study)


def test_pending_terminal_never_loads_checkpoints_or_publishes(study, monkeypatch):
    root = study[0]
    (
        root / materializer.ARMS["MASK_CE"]["directory"] / "terminal_manifest.json"
    ).unlink()
    calls = []
    monkeypatch.setattr(materializer, "inspect_checkpoint", lambda *a: calls.append(a))
    with pytest.raises(FileNotFoundError):
        build(study)
    assert calls == []
    assert not (root / materializer.OUTPUT).exists()


@pytest.mark.parametrize(
    "mutation",
    [None, "ce_marker", "prior_marker", "ema_updates", "ema_nan", "checkpoint_bytes"],
)
def test_actual_small_checkpoint_reuses_ce_prior_and_ema_validation(tmp_path, mutation):
    value = prior_cases.checkpoint(0.9, ce=True)
    value.update(
        global_step=1000,
        optimizer_states=[{"moment": torch.zeros(1)}],
        ema={"num_updates": 1000, "shadow_params": [torch.ones(1)]},
    )
    if mutation == "ce_marker":
        value["state_dict"]["_udlm_denoiser_ce_version"] = torch.tensor(1.0)
    elif mutation == "prior_marker":
        del value["state_dict"][materializer.benchmark.UDLM_MASK_RICH_STATE_KEY]
    elif mutation == "ema_updates":
        value["ema"]["num_updates"] = 999
    elif mutation == "ema_nan":
        value["ema"]["shadow_params"][0][0] = float("nan")
    path = tmp_path / "tiny.ckpt"
    torch.save(value, path)
    payload = path.read_bytes()
    terminal = {
        "plan": {"config": value["hyper_parameters"]["config"]},
        "checkpoint": {
            "relative_path": path.name,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
            "global_step": 1000,
            "finite_tensor_count": 7,
        },
    }
    if mutation == "checkpoint_bytes":
        path.write_bytes(payload + b"changed")
    previous = materializer.engine.ROOT
    if mutation is None:
        assert (
            materializer.inspect_checkpoint(tmp_path, terminal, "MASK_CE")[
                "global_step"
            ]
            == 1000
        )
    else:
        with pytest.raises((ValueError, RuntimeError)):
            materializer.inspect_checkpoint(tmp_path, terminal, "MASK_CE")
    assert materializer.engine.ROOT == previous


def test_publication_is_exclusive_and_acceptance_failure_emits_nothing(
    tmp_path, monkeypatch
):
    (tmp_path / Path(materializer.OUTPUT).parent).mkdir(parents=True)
    monkeypatch.setattr(materializer, "ROOT", tmp_path)
    monkeypatch.setattr(
        materializer, "build_protocol", lambda **kwargs: {"fixture": "accepted"}
    )
    assert materializer.main(["--v11b-terminal-sha256", "a" * 64]) == 0
    before = (tmp_path / materializer.OUTPUT).read_bytes()
    with pytest.raises(FileExistsError):
        materializer.main(["--v11b-terminal-sha256", "a" * 64])
    assert (tmp_path / materializer.OUTPUT).read_bytes() == before
    (tmp_path / materializer.OUTPUT).unlink()

    def rejected(**kwargs):
        raise ValueError("not completed")

    monkeypatch.setattr(materializer, "build_protocol", rejected)
    with pytest.raises(ValueError, match="not completed"):
        materializer.main(["--v11b-terminal-sha256", "a" * 64])
    assert not (tmp_path / materializer.OUTPUT).exists()
