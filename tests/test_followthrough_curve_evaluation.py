import json

import pytest

from scripts.udlm.run_followthrough_curve_evaluation import config_for, training_receipts
from scripts.udlm.report_exploration import _sampling_budget


def receipt(tmp_path, arm, **updates):
    path = tmp_path / f"output/udlm/followthrough_learning_r2/{arm}_4000_b128_w2/terminal_manifest.json"
    path.parent.mkdir(parents=True)
    data = {"status": "completed", "training_return_code": 0,
            "completed_example_exposures": 512000, "leases_release_authorized": True}
    data.update(updates)
    path.write_text(json.dumps(data))


def test_missing_control_blocks_evaluation(tmp_path):
    receipt(tmp_path, "ct")
    assert training_receipts(tmp_path) is None
    receipt(tmp_path, "mdlm")
    assert set(training_receipts(tmp_path)) == {"ct", "mdlm"}


@pytest.mark.parametrize("change", [{"status": "failed"}, {"training_return_code": 1},
                                     {"completed_example_exposures": 128000},
                                     {"leases_release_authorized": False}])
def test_failed_or_incomplete_training_cannot_be_substituted(tmp_path, change):
    receipt(tmp_path, "ct", **change)
    with pytest.raises(RuntimeError):
        training_receipts(tmp_path)


def test_native_mdlm_sampling_is_preserved_and_its_compute_is_reported():
    mdlm, ct = config_for("mdlm", "test.ckpt"), config_for("ct", "test.ckpt")
    assert mdlm["randomness"] == 0.5 and "num_steps" not in mdlm
    assert ct["randomness"] == 0.0 and ct["num_steps"] == 128
    assert mdlm["softmax_temp"] == ct["softmax_temp"] == 0.5
    budget = _sampling_budget({"status": "completed", "generation_protocol": {
        "diffusion_type": "mdlm", "nfe": 53}})
    assert budget["nfe"] == budget["predictor_transitions_per_molecule"] == 53
    assert budget["corrector_steps_per_molecule"] == 0
