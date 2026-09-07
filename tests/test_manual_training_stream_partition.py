"""Exercise train() with real CPU iterable loaders and simulated DDP ranks."""

from types import SimpleNamespace

import datasets
import pytest
from omegaconf import OmegaConf

from genmol.utils import utils_data
from scripts import train as entrypoint


@pytest.fixture
def run_training(monkeypatch, tmp_path):
    monkeypatch.setattr(entrypoint, "_PILOT_CONTRACT", None)
    for name in (
        "_validate_and_record_pilot_config",
        "_screen_initialization_state_audit",
        "_reseed_training_rng_after_model_initialization",
        "_write_pilot_training_summary",
        "_launch_manifest_evidence",
    ):
        monkeypatch.setattr(entrypoint, name, lambda *args, **kwargs: None)
    monkeypatch.setattr(entrypoint, "_pilot_callbacks", lambda _config: [])
    monkeypatch.setattr(entrypoint, "_training_strategy", object)
    monkeypatch.setattr(entrypoint, "GenMol", lambda _config: object())
    monkeypatch.setattr(entrypoint, "get_last_checkpoint", lambda _path: None)
    monkeypatch.setattr(entrypoint.L, "seed_everything", lambda *args, **kwargs: None)
    monkeypatch.setattr(utils_data, "Collator", lambda _config: lambda rows: rows)
    monkeypatch.setattr(
        utils_data.datasets,
        "load_dataset",
        lambda *args, **kwargs: datasets.IterableDataset.from_generator(
            lambda: ({"safe": str(index)} for index in range(24))
        ),
    )
    shard_calls = []
    original_split = utils_data.split_dataset_by_node

    def record_split(dataset, *, rank, world_size):
        shard_calls.append((rank, world_size))
        return original_split(dataset, rank=rank, world_size=world_size)

    monkeypatch.setattr(utils_data, "split_dataset_by_node", record_split)

    def run(rank, world_size, *, data="safe"):
        config = OmegaConf.create(
            {
                "seed": 19,
                "data": data,
                "wandb": {"name": None},
                "training": {"use_bracket_safe": False},
                "callback": {"dirpath": str(tmp_path / "unused-checkpoints")},
                "trainer": {"test_trainer_marker": True},
                "loader": {"batch_size": 2, "num_workers": 0, "pin_memory": False},
            }
        )
        observed = []
        events = []
        trainer = SimpleNamespace(global_rank=rank, world_size=world_size, num_nodes=1)

        def fit(_model, loader, *, ckpt_path):
            assert ckpt_path is None
            observed.extend(
                row["safe" if data == "safe" else "input"]
                for batch in loader
                for row in batch
            )

        trainer.fit = fit

        def instantiate(value, **kwargs):
            if value.get("test_trainer_marker"):
                events.append("trainer")
                return trainer
            return object()

        def loader(*args, **kwargs):
            if data == "safe":
                assert events == [
                    "trainer"
                ], "hosted loader was constructed before Trainer"
            events.append("loader")
            return utils_data.get_dataloader(*args, **kwargs)

        monkeypatch.setattr(entrypoint.hydra.utils, "instantiate", instantiate)
        monkeypatch.setattr(entrypoint, "get_dataloader", loader)
        entrypoint.train.__wrapped__(config)
        assert events.count("loader") == 1
        return observed

    return run, shard_calls


def test_direct_two_rank_stream_has_disjoint_full_coverage(run_training):
    run, calls = run_training
    rank0 = run(0, 2)
    rank1 = run(1, 2)
    assert len(rank0) == len(rank1) == 12
    assert not set(rank0).intersection(rank1)
    assert set(rank0).union(rank1) == {str(index) for index in range(24)}
    assert calls == [(0, 2), (1, 2)]


def test_direct_single_rank_preserves_entire_order_without_sharding(run_training):
    run, calls = run_training
    assert run(0, 1) == [str(index) for index in range(24)]
    assert calls == []


def test_registered_pilot_still_uses_exact_existing_partition_once(
    run_training, monkeypatch
):
    run, calls = run_training
    monkeypatch.setattr(entrypoint, "_PILOT_CONTRACT", {"expected_world_size": 2})
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("NODE_RANK", "0")
    assert run(1, 2) == [str(index) for index in range(1, 24, 2)]
    assert calls == [(1, 2)]


def test_manual_user_dataset_keeps_original_loader_path(
    run_training, tmp_path, monkeypatch
):
    run, calls = run_training
    path = tmp_path / "local.safe"
    path.write_text("CC\nCO\nCN\n")
    # Isolate entrypoint routing from UserDataset's unrelated inherited
    # Hugging Face batched-indexing behavior; retain the real DataLoader.
    monkeypatch.setattr(
        utils_data,
        "UserDataset",
        lambda filename: [{"input": line} for line in path.read_text().splitlines()],
    )
    assert sorted(run(0, 1, data=str(path))) == ["CC", "CN", "CO"]
    assert calls == []


@pytest.mark.parametrize(
    "rank,world_size", [(True, 2), (-1, 2), (2, 2), (0, 0), (0, True)]
)
def test_manual_partition_rejects_invalid_trainer_identity(
    monkeypatch, rank, world_size
):
    monkeypatch.setattr(entrypoint, "_PILOT_CONTRACT", None)
    with pytest.raises(RuntimeError):
        entrypoint._training_streaming_partition(
            SimpleNamespace(global_rank=rank, world_size=world_size)
        )
