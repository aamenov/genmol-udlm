"""Independent scalar laws and mocked diagnostic orchestration; no model calls."""

from copy import deepcopy
import hashlib
import json
import math

import pytest

from scripts.udlm import audit_near_absorbing_transfer as audit


def test_terminal_kl_matches_full_vocabulary_direct_sum():
    base = [0.02, 0.08, 0.3, 0.6]
    for mixture in audit.MIXTURES:
        prior = [(1 - mixture) * probability for probability in base]
        prior[0] += mixture
        for clean in range(len(prior)):
            for alpha in (0.001, 0.2, 0.9):
                marginal = [
                    (1 - alpha) * probability + alpha * (token == clean)
                    for token, probability in enumerate(prior)
                ]
                direct = math.fsum(
                    q * math.log(q / probability)
                    for q, probability in zip(marginal, prior)
                )
                assert audit.terminal_kl(prior[clean], alpha) == pytest.approx(
                    direct, rel=2e-12, abs=1e-14
                )


def test_tv_to_absorbing_is_independent_of_clean_token():
    base, mask = [0.02, 0.08, 0.3, 0.6], 0
    for mixture in audit.MIXTURES:
        prior = [(1 - mixture) * probability for probability in base]
        prior[mask] += mixture
        for time in audit.TIMES:
            alpha = 1 - 0.999 * time
            wanted = (1 - alpha) * (1 - mixture) * (1 - base[mask])
            for clean in range(len(prior)):
                actual = (
                    math.fsum(
                        abs(
                            alpha * (token == clean)
                            + (1 - alpha) * prior[token]
                            - alpha * (token == clean)
                            - (1 - alpha) * (token == mask)
                        )
                        for token in range(len(prior))
                    )
                    / 2
                )
                assert actual == pytest.approx(wanted, rel=2e-11, abs=2e-16)


def test_terminal_mismatch_grows_when_nonmask_prior_mass_vanishes():
    values = [audit.terminal_kl((1 - mixture) * 0.08) for mixture in audit.MIXTURES]
    assert all(right > left > 0 for left, right in zip(values, values[1:]))
    assert audit.terminal_kl(1e-100) > audit.terminal_kl(1e-10)


@pytest.mark.parametrize(
    "probability,alpha",
    [
        (0, 0.001),
        (1, 0.001),
        (-0.1, 0.001),
        (float("nan"), 0.001),
        (0.2, 0),
        (0.2, 1),
        (0.2, float("nan")),
        (0.2, float("inf")),
    ],
)
def test_terminal_kl_rejects_noninterior_laws(probability, alpha):
    with pytest.raises(ValueError, match="interior"):
        audit.terminal_kl(probability, alpha)


@pytest.fixture
def synthetic(tmp_path, monkeypatch):
    source = tmp_path / "scripts/udlm/audit_near_absorbing_transfer.py"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"# Synthetic diagnostic wrapper source\n")
    design = tmp_path / "experiments/udlm/designs/near_absorbing_transfer_cpu.md"
    design.parent.mkdir(parents=True)
    design.write_bytes(b"Synthetic design: no actual diagnostic outcomes.\n")
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    monkeypatch.setattr(audit, "__file__", str(source))
    rows = [
        {
            "source_index": index,
            "input_ids": [0, *([1, 2] * (25 if index < 15 else 32)), 3],
        }
        for index in range(16)
    ]
    panel = {"tokenizer": {"special_token_ids": [0, 3]}, "rows": rows}
    counts = [0, 7, 3, 0, *([1] * 1876)]
    data = {
        "schema_version": 1,
        "purpose": "Synthetic frozen token diagnostic only",
        "device": "cpu",
        "shape": [16, 66],
        "content_tokens": 814,
        "checkpoint_sha256": audit.original.CHECKPOINT_SHA,
        "panel_sha256": audit.original.PANEL_SHA,
        "frequency_sha256": audit.original.FREQUENCY_SHA,
        "uniform_floor": 0.0002,
        "noise_eps": 0.001,
        "inference_weights": {"source": "MDLM EMA", "ema_applied": True},
        "source_row_indices": list(range(16)),
        "source_hashes": {"scripts/udlm/audit_mask_rich_transfer.py": "a" * 64},
        "limitations": ["Synthetic fixture; no model forward was performed."],
        "results": [
            {
                "mask_mixture_weight": mixture,
                "time": time,
                "seed": audit.SEED + index,
                "groups": {"changed_nonmask": {"tokens": 0, "ce_mean": None}},
                "corrupted_token_ids_int64_bytes_sha256": f"{slot:064x}",
            }
            for slot, (mixture, index, time) in enumerate(
                (mixture, index, time)
                for mixture in audit.MIXTURES
                for index, time in enumerate(audit.TIMES)
            )
        ],
    }
    original_defaults = (
        audit.original.MIXTURES,
        audit.original.TIMES,
        audit.original.SEED,
    )
    calls, reads = [], []

    def main():
        calls.append(
            (audit.original.MIXTURES, audit.original.TIMES, audit.original.SEED)
        )
        print(json.dumps(data))

    def read_pinned(path, expected):
        reads.append((path, expected))
        if path == audit.original.PANEL:
            assert expected == audit.original.PANEL_SHA
            return deepcopy(panel)
        assert (
            path == audit.original.FREQUENCY
            and expected == audit.original.FREQUENCY_SHA
        )
        return {"counts_by_token_id": list(counts)}

    monkeypatch.setattr(audit.original, "main", main)
    monkeypatch.setattr(audit.original, "read_pinned", read_pinned)
    return {
        "data": data,
        "calls": calls,
        "reads": reads,
        "counts": counts,
        "defaults": original_defaults,
        "source": source,
        "design": design,
    }


def test_mocked_runner_enriches_all_conditions_and_preserves_raw_groups(
    synthetic, capsys
):
    audit.main()
    result = json.loads(capsys.readouterr().out)
    assert synthetic["calls"] == [(audit.MIXTURES, audit.TIMES, audit.SEED)]
    assert (
        audit.original.MIXTURES,
        audit.original.TIMES,
        audit.original.SEED,
    ) == synthetic["defaults"]
    assert len(synthetic["reads"]) == 2
    assert result["fixed_conditions"] == {
        "mixtures": list(audit.MIXTURES),
        "times": list(audit.TIMES),
        "base_seed": 2400,
        "forward_calls": 12,
        "rows_per_forward": 16,
    }
    assert (
        result["content_tokens"] == 814
        and result["study"] == "near_absorbing_frozen_transfer_cpu"
    )
    counts = synthetic["counts"]
    base = [
        (1 - 0.0002) * count / sum(counts) + 0.0002 / len(counts) for count in counts
    ]
    normalizer = math.fsum(base)
    base = [probability / normalizer for probability in base]
    for original, row in zip(synthetic["data"]["results"], result["results"]):
        assert {key: row[key] for key in original} == original
        expected = (
            0.999 * row["time"] * (1 - row["mask_mixture_weight"]) * (1 - base[0])
        )
        assert row["one_token_tv_to_absorbing_marginal"] == pytest.approx(expected)
        assert row["groups"]["changed_nonmask"]["ce_mean"] is None
    for row in result["terminal_kl_over_observed_clean_content"]:
        prior = [(1 - row["mask_mixture_weight"]) * probability for probability in base]
        prior[3] += row["mask_mixture_weight"]
        direct = []
        for clean in (1, 2):
            q = [
                (1 - 0.001) * p + 0.001 * (token == clean)
                for token, p in enumerate(prior)
            ]
            direct.append(
                math.fsum(value * math.log(value / p) for value, p in zip(q, prior))
            )
        assert row["alpha_1"] == 0.001
        assert row["mean_nats"] == pytest.approx(math.fsum(direct) / 2, abs=1e-14)
    assert (
        result["source_hashes"]["scripts/udlm/audit_mask_rich_transfer.py"] == "a" * 64
    )
    for key, file in (
        ("scripts/udlm/audit_near_absorbing_transfer.py", synthetic["source"]),
        (
            "experiments/udlm/designs/near_absorbing_transfer_cpu.md",
            synthetic["design"],
        ),
    ):
        assert (
            result["source_hashes"][key]
            == hashlib.sha256(file.read_bytes()).hexdigest()
        )


@pytest.mark.parametrize("failure", ["exception", "non_json_stdout"])
def test_original_module_defaults_restored_after_failure(
    synthetic, monkeypatch, failure
):
    def broken():
        if failure == "exception":
            raise RuntimeError("Synthetic original diagnostic failure")
        print("Unexpected library output before JSON")

    monkeypatch.setattr(audit.original, "main", broken)
    with pytest.raises((RuntimeError, json.JSONDecodeError)):
        audit.main()
    assert (
        audit.original.MIXTURES,
        audit.original.TIMES,
        audit.original.SEED,
    ) == synthetic["defaults"]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda data: data["results"].pop(),
        lambda data: data["results"][0].update(seed=999),
        lambda data: data["results"][0].update(mask_mixture_weight=0.8),
        lambda data: data.update(device="cuda"),
        lambda data: data.update(shape=[15, 66]),
        lambda data: data.update(content_tokens=813),
        lambda data: data.update(checkpoint_sha256="0" * 64),
        lambda data: data.update(panel_sha256="0" * 64),
        lambda data: data.update(frequency_sha256="0" * 64),
        lambda data: data.update(uniform_floor=0.001),
        lambda data: data.update(noise_eps=0.01),
    ],
)
def test_wrong_reused_result_cannot_be_relabeled_as_fixed_design(
    synthetic, capsys, mutation
):
    mutation(synthetic["data"])
    with pytest.raises((AssertionError, ValueError)):
        audit.main()
    assert capsys.readouterr().out == ""
