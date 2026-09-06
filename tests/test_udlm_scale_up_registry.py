from __future__ import annotations

import copy
import hashlib
import subprocess
from decimal import Decimal
from types import SimpleNamespace

import pytest

from scripts.udlm import launch_train_pilot as pilot
from scripts.udlm import prepare_scale_up_registry as prepare
from scripts.udlm import verify_scale_up_registry as scale


def _selected_config() -> dict[str, object]:
    return {
        "seed": 17,
        "trainer": {
            "devices": 1,
            "max_steps": 500,
            "accumulate_grad_batches": 8,
            "num_nodes": 1,
        },
        "loader": {
            "global_batch_size": 16,
            "batch_size": 2,
            "num_workers": 1,
        },
        "callback": {
            "dirpath": "/old/screen/checkpoints",
            "every_n_train_steps": 500,
            "filename": "{step}",
        },
        "training": {
            "reseed_after_model_initialization": True,
            "init_from_mdlm_checkpoint": "/project/50000.ckpt",
            "udlm": {
                "prior_variant": "empirical_frequency",
                "conditioning_variant": "film_adaln",
                "zero_init_conditioning": False,
                "exclude_special_tokens": False,
            },
        },
        "optim": {
            "scheduler": {
                "name": "half_cosine_with_linear_warmup_and_floor",
                "warmup_updates": 50,
                "horizon_updates": 1000,
                "decay_floor_lr": 0.000003,
            }
        },
    }


@pytest.mark.parametrize(
    ("world_size", "global_batch", "accumulation"),
    ((1, 16, 8), (2, 16, 4), (3, 18, 3), (4, 16, 2)),
)
def test_default_batch_plan_is_exact_for_all_supported_world_sizes(
    world_size: int, global_batch: int, accumulation: int
) -> None:
    assert scale.default_global_batch_size(world_size) == global_batch
    assert scale.exact_accumulation_steps(global_batch, 2, world_size) == accumulation


@pytest.mark.parametrize("world_size", (0, 5, True))
def test_world_size_outside_one_through_four_is_rejected(world_size: object) -> None:
    with pytest.raises(scale.ScaleUpValidationError):
        scale.validate_gpu_count(world_size)  # type: ignore[arg-type]


def test_r4_firewall_allows_exact_generation_readiness_paths_only() -> None:
    expected = {
        "scripts/exps/denovo/launch_benchmark.py",
        "scripts/exps/denovo/report.py",
        "scripts/exps/denovo/hparams_udlm_categorical_floor0002.yaml",
        "scripts/udlm/superiority_gate.py",
        "tests/test_denovo_benchmark.py",
        "tests/test_denovo_launcher.py",
        "tests/test_denovo_report.py",
        "tests/test_udlm_superiority_gate.py",
    }
    assert scale.GENERATION_READINESS_FRAMEWORK_PATHS == expected
    assert expected <= scale.FRAMEWORK_OPTIONAL_EXACT_PATHS
    assert all(scale._framework_path_allowed(path) for path in expected)
    assert {
        "scripts/exps/denovo/launch_benchmark.py",
        "scripts/exps/denovo/report.py",
        "scripts/exps/denovo/hparams_udlm_categorical_floor0002.yaml",
        "scripts/udlm/superiority_gate.py",
    } <= set(scale.SOURCE_PATHS)
    assert not scale._framework_path_allowed(
        "scripts/exps/denovo/unreviewed_generation_change.py"
    )


def test_r4_firewall_allows_exact_compatibility_repair_paths_only() -> None:
    expected = {
        "scripts/udlm/launch_health_panel.py",
        "scripts/udlm/validate_health_panel.py",
        "tests/test_checkpoint_io.py",
        "tests/test_udlm_health_panel_launcher.py",
        "tests/test_udlm_optimization_screen_launcher.py",
    }
    assert scale.R4_COMPATIBILITY_REPAIR_PATHS == expected
    assert expected <= scale.FRAMEWORK_OPTIONAL_EXACT_PATHS
    assert all(scale._framework_path_allowed(path) for path in expected)
    assert {
        "scripts/udlm/launch_health_panel.py",
        "scripts/udlm/validate_health_panel.py",
    } <= set(scale.SOURCE_PATHS)
    assert not scale._framework_path_allowed(
        "tests/test_udlm_unreviewed_compatibility_repair.py"
    )


def test_r4_publication_accepts_generation_readiness_set_and_rejects_extra() -> None:
    framework_paths = sorted(
        scale.FRAMEWORK_REQUIRED_PATHS
        | scale.GENERATION_READINESS_FRAMEWORK_PATHS
        | scale.R4_COMPATIBILITY_REPAIR_PATHS
    )
    config_paths = sorted(prepare._config_paths(1).values())
    publication = {
        "selection_revision": "a" * 40,
        "framework_revision": "b" * 40,
        "config_revision": "c" * 40,
        "framework_paths": framework_paths,
        "config_paths": config_paths,
    }
    common_kwargs = {
        "git_ancestor_checker": lambda parent, child: True,
        "git_sole_parent_checker": lambda child, parent: True,
        "git_tree_paths_loader": lambda revision, path: frozenset(),
        "git_pushed_checker": lambda revision: True,
        "git_diff_checker": lambda parent, child, paths: True,
        "changed_paths_loader": lambda parent, child: (
            frozenset(framework_paths)
            if parent == "a" * 40
            else frozenset(config_paths)
        ),
    }

    validated = scale._validate_publication(publication, **common_kwargs)
    assert validated["framework_paths"] == framework_paths

    publication_with_extra = copy.deepcopy(publication)
    publication_with_extra["framework_paths"].append(
        "scripts/exps/denovo/unreviewed_generation_change.py"
    )
    with pytest.raises(scale.ScaleUpValidationError, match="path set is invalid"):
        scale._validate_publication(publication_with_extra, **common_kwargs)


def test_derive_members_preserves_selected_design_and_masks_only_treatment(
    tmp_path,
) -> None:
    selected = _selected_config()
    snapshots = []
    for variant, slug in zip(
        scale.EXPECTED_VARIANTS, scale.EXPECTED_SLUGS, strict=True
    ):
        snapshots.append(
            scale.derive_scale_up_config(
                selected,
                training_variant=variant,
                gpu_count=3,
                global_batch_size=18,
                micro_batch_size=2,
                num_workers=1,
                output_directory=f"output/udlm/scale-{slug}",
                repository_root=tmp_path,
            )
        )
    assert selected == _selected_config(), "derivation must not mutate screen bytes"
    assert {item["trainer"]["accumulate_grad_batches"] for item in snapshots} == {3}
    assert {item["trainer"]["max_steps"] for item in snapshots} == {1000}
    assert {item["training"]["reseed_after_model_initialization"] for item in snapshots} == {
        True
    }
    assert {
        item["training"]["udlm"]["conditioning_variant"] for item in snapshots
    } == {"film_adaln"}
    assert {item["optim"]["scheduler"]["name"] for item in snapshots} == {
        "half_cosine_with_linear_warmup_and_floor"
    }
    assert {
        item["training"]["udlm"]["prior_variant"] for item in snapshots
    } == set(scale.EXPECTED_PRIORS.values())
    assert len({scale.matched_config_sha256(item) for item in snapshots}) == 1

    tampered = copy.deepcopy(snapshots[0])
    tampered["optim"]["scheduler"]["warmup_updates"] = 51
    assert scale.matched_config_sha256(tampered) != scale.matched_config_sha256(
        snapshots[0]
    )


def test_preparer_builds_exact_w3_member_names_and_matched_configs() -> None:
    revision = "a" * 40
    documents, members = prepare._candidate_documents(
        selected_config=_selected_config(),
        gpu_count=3,
        global_batch_size=18,
        micro_batch_size=2,
        num_workers=1,
        config_revision_for_names=revision,
    )
    assert [member["run_name"] for member in members] == [
        f"scaleup-w3-{slug}-{revision[:12]}" for slug in scale.EXPECTED_SLUGS
    ]
    assert [member["training_variant"] for member in members] == list(
        scale.EXPECTED_VARIANTS
    )
    assert len({scale.matched_config_sha256(item) for item in documents.values()}) == 1
    assert all(
        item["trainer"]["accumulate_grad_batches"] == 3
        and item["training"]["reseed_after_model_initialization"] is True
        for item in documents.values()
    )


def test_preparer_cli_exposes_world_size_but_no_scientific_overrides() -> None:
    parsed = prepare._parse_args(["materialize-configs", "--gpu-count", "4"])
    assert vars(parsed) == {"command": "materialize-configs", "gpu_count": 4}
    with pytest.raises(SystemExit):
        prepare._parse_args(
            [
                "materialize-configs",
                "--gpu-count",
                "1",
                "--micro-batch-size",
                "4",
            ]
        )


def test_committed_blob_reference_returns_exact_live_digest(
    tmp_path, monkeypatch
) -> None:
    payload = b"selection-bound scale-up source\n"
    relative_path = "scripts/udlm/example.py"
    live_path = tmp_path / relative_path
    live_path.parent.mkdir(parents=True)
    live_path.write_bytes(payload)
    monkeypatch.setattr(prepare, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(
        prepare,
        "_git_blob_bytes",
        lambda revision, path: payload
        if revision == "a" * 40 and path == relative_path
        else b"",
    )

    assert prepare._blob_reference_from_git("a" * 40, relative_path) == {
        "root": "repository",
        "relative_path": relative_path,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def test_exclusive_publisher_rejects_symlinked_parent(tmp_path, monkeypatch) -> None:
    repository = tmp_path / "repo"
    outside = tmp_path / "outside"
    repository.mkdir()
    outside.mkdir()
    (repository / "link").symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(prepare, "REPOSITORY_ROOT", repository)

    with pytest.raises(prepare.ScaleUpPreparationError, match="direct real directory"):
        prepare._publish_bytes_exclusive(
            repository / "link" / "escaped.json",
            b"{}\n",
            label="adversarial registry",
        )
    assert not (outside / "escaped.json").exists()


def test_exclusive_publisher_creates_direct_parents_without_overwrite(
    tmp_path, monkeypatch
) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    monkeypatch.setattr(prepare, "REPOSITORY_ROOT", repository)
    target = repository / "new" / "direct" / "artifact.json"

    prepare._publish_bytes_exclusive(target, b"{}\n", label="test registry")
    assert target.read_bytes() == b"{}\n"
    with pytest.raises(FileExistsError, match="refusing to replace"):
        prepare._publish_bytes_exclusive(target, b"changed\n", label="test registry")
    assert target.read_bytes() == b"{}\n"


def test_historical_absence_queries_fail_closed_on_indeterminate_git(
    monkeypatch,
) -> None:
    def unavailable_tree(revision, path):
        raise RuntimeError("object database unavailable")

    with pytest.raises(scale.ScaleUpValidationError, match="cannot prove"):
        scale._git_blob_absent(
            "a" * 40,
            "artifact.json",
            git_tree_paths_loader=unavailable_tree,
            label="adversarial artifact",
        )

    monkeypatch.setattr(
        prepare.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=args, returncode=128, stdout=b"", stderr=b"fatal"
        ),
    )
    with pytest.raises(prepare.ScaleUpPreparationError, match="indeterminate"):
        prepare._git_blob_absent("a" * 40, "artifact.json")


def test_registry_publication_requires_configs_absent_at_framework_revision() -> None:
    framework_paths = sorted(scale.FRAMEWORK_REQUIRED_PATHS)
    config_paths = sorted(prepare._config_paths(1).values())
    publication = {
        "selection_revision": "a" * 40,
        "framework_revision": "b" * 40,
        "config_revision": "c" * 40,
        "framework_paths": framework_paths,
        "config_paths": config_paths,
    }

    with pytest.raises(scale.ScaleUpValidationError, match="unexpectedly existed"):
        scale._validate_publication(
            publication,
            git_ancestor_checker=lambda parent, child: True,
            git_sole_parent_checker=lambda child, parent: True,
            git_tree_paths_loader=lambda revision, path: (
                frozenset() if revision == "a" * 40 else frozenset({path.as_posix()})
            ),
            git_pushed_checker=lambda revision: True,
            git_diff_checker=lambda parent, child, paths: True,
            changed_paths_loader=lambda parent, child: (
                frozenset(framework_paths)
                if parent == "a" * 40
                else frozenset(config_paths)
            ),
        )

    with pytest.raises(
        scale.ScaleUpValidationError, match="scale-up framework path unexpectedly"
    ):
        scale._validate_publication(
            publication,
            git_ancestor_checker=lambda parent, child: True,
            git_sole_parent_checker=lambda child, parent: True,
            git_tree_paths_loader=lambda revision, path: (
                frozenset({path.as_posix()})
                if revision == "a" * 40
                else frozenset()
            ),
            git_pushed_checker=lambda revision: True,
            git_diff_checker=lambda parent, child, paths: True,
            changed_paths_loader=lambda parent, child: (
                frozenset(framework_paths)
                if parent == "a" * 40
                else frozenset(config_paths)
            ),
        )


@pytest.mark.parametrize(
    "preexisting_path",
    (
        "experiments/udlm/protocols/selection_bound_scale_up_configs_gpu2",
        "experiments/udlm/protocols/selection_bound_scale_up_registry_gpu3.json",
    ),
)
def test_registry_publication_rejects_unselected_preexisting_scale_namespaces(
    preexisting_path: str,
) -> None:
    framework_paths = sorted(scale.FRAMEWORK_REQUIRED_PATHS)
    config_paths = sorted(prepare._config_paths(1).values())
    publication = {
        "selection_revision": "a" * 40,
        "framework_revision": "b" * 40,
        "config_revision": "c" * 40,
        "framework_paths": framework_paths,
        "config_paths": config_paths,
    }

    with pytest.raises(scale.ScaleUpValidationError, match="family.*unexpectedly"):
        scale._validate_publication(
            publication,
            git_ancestor_checker=lambda parent, child: True,
            git_sole_parent_checker=lambda child, parent: True,
            git_tree_paths_loader=lambda revision, path: (
                frozenset({preexisting_path})
                if revision == "b" * 40 and path.as_posix() == preexisting_path
                else frozenset()
            ),
            git_pushed_checker=lambda revision: True,
            git_diff_checker=lambda parent, child, paths: True,
            changed_paths_loader=lambda parent, child: (
                frozenset(framework_paths)
                if parent == "a" * 40
                else frozenset(config_paths)
            ),
        )

def test_registry_builder_normalizes_strict_decimal_leaves(monkeypatch) -> None:
    documents, members = prepare._candidate_documents(
        selected_config=_selected_config(),
        gpu_count=1,
        global_batch_size=16,
        micro_batch_size=2,
        num_workers=1,
        config_revision_for_names="b" * 40,
    )
    registry = SimpleNamespace(
        data={
            "stages": [
                {
                    "arms": [
                        {
                            "arm_id": "E-L1",
                            "scheduler": {
                                "name": "half_cosine_with_linear_warmup_and_floor",
                                "decay_floor_lr": Decimal("0.000003"),
                            },
                        }
                    ]
                },
                {
                    "arms": [
                        {
                            "arm_id": "E-A1",
                            "conditioner": {"kind": "film_adaln"},
                            "resolved_configs": [
                                {
                                    "scheduler_arm_id": "E-L1",
                                    "config": {
                                        "root": "repository",
                                        "relative_path": "screen.json",
                                        "sha256": "1" * 64,
                                        "size_bytes": 1,
                                        "canonical_sha256": "2" * 64,
                                    },
                                }
                            ],
                        }
                    ]
                },
            ],
            "common_training": {
                "initialization": {"checkpoint": {"root": "project"}}
            },
        }
    )
    monkeypatch.setattr(
        prepare,
        "_blob_reference_from_git",
        lambda revision, path: {
            "root": "repository",
            "relative_path": path,
            "sha256": "3" * 64,
            "size_bytes": 1,
        },
    )

    candidate = prepare._build_registry_document(
        selection_revision="a" * 40,
        framework_revision="b" * 40,
        config_revision="c" * 40,
        framework_paths=sorted(scale.FRAMEWORK_REQUIRED_PATHS),
        config_paths=sorted(prepare._config_paths(1).values()),
        screen_authority={},
        screen_registry=registry,  # type: ignore[arg-type]
        scheduler_selection={"selected_arm_id": "E-L1"},
        conditioning_selection={"selected_arm_id": "E-A1"},
        selected_config=_selected_config(),
        gpu_count=1,
        global_batch_size=16,
        micro_batch_size=2,
        num_workers=1,
        documents=documents,
        members=members,
    )

    floor = candidate["selected_design"]["scheduler"]["decay_floor_lr"]
    assert isinstance(floor, float)
    assert floor == 3e-6
    assert scale.json_bytes(candidate)


def test_scale_registry_rejects_noncanonical_json_serialization() -> None:
    payload = b'{  "schema_version" : 1 }\n'
    parsed = scale.screen.strict_json_loads(payload, label="adversarial registry")
    with pytest.raises(
        scale.ScaleUpValidationError,
        match="deterministic scale-up JSON serialization",
    ):
        scale.load_validated_registry(
            payload,
            relative_path=(
                "experiments/udlm/protocols/"
                "selection_bound_scale_up_registry_gpu1.json"
            ),
            expected_raw_sha256=hashlib.sha256(payload).hexdigest(),
            expected_canonical_sha256=scale.canonical_json_sha256(parsed),
        )


def _registry_header() -> dict[str, object]:
    return {
        "schema_version": scale.REGISTRY_SCHEMA_VERSION,
        "registry_id": scale.EXPECTED_REGISTRY_ID,
        "status": scale.EXPECTED_STATUS,
        "claim_scope": scale.EXPECTED_CLAIM_SCOPE,
        "firewall": {
            "generation_metrics_allowed": False,
            "superiority_evidence_eligible": False,
            "candidate_lock_eligible": False,
            "failed_or_missing_receipt_policy": "incomplete_no_panel",
            "unregistered_attempts_allowed": False,
        },
        "publication": None,
        "source": None,
        "screen_authority": None,
        "selected_design": None,
        "common_training": None,
        "members": None,
    }


def test_registry_rejects_boolean_schema_and_numeric_false_firewall() -> None:
    registry = _registry_header()
    registry["schema_version"] = True
    with pytest.raises(scale.ScaleUpValidationError, match="schema_version"):
        scale.validate_registry(registry)

    registry = _registry_header()
    registry["firewall"]["generation_metrics_allowed"] = 0
    with pytest.raises(scale.ScaleUpValidationError, match="firewall"):
        scale.validate_registry(registry)


def test_json_reference_rejects_boolean_schema_before_loading() -> None:
    reference = {
        "root": "repository",
        "relative_path": "evidence.json",
        "sha256": "5" * 64,
        "size_bytes": 1,
        "schema_version": True,
        "canonical_sha256": "6" * 64,
    }
    with pytest.raises(scale.ScaleUpValidationError, match="schema.*integer"):
        scale._load_reference(
            reference,
            label="adversarial evidence",
            revision=None,
            loader=lambda root, path: b"{}",
            git_blob_loader=lambda revision, path: b"{}",
            json_schema=1,
            require_canonical=True,
        )

def _common_training() -> tuple[dict[str, object], SimpleNamespace]:
    initialization = {"checkpoint": {"root": "project"}}
    common = {
        "optimizer_updates": 1_000,
        "training_seed": 17,
        "gpu_count": 1,
        "global_batch_size": 16,
        "micro_batch_size_per_process": 2,
        "accumulate_grad_batches": 8,
        "effective_global_batch_size": 16,
        "num_workers": 1,
        "exclude_special_tokens": False,
        "reseed_after_model_initialization": True,
        "initialization": initialization,
        "gpu_safety_policy": {
            "max_utilization_percent": 10,
            "utilization_comparison": "strictly_less_than",
            "min_free_memory_mib": 30_000,
            "active_compute_processes_allowed": True,
            "compute_mode_prohibited_allowed": False,
        },
        "common_resolved_config_sha256": "4" * 64,
        "artifact_schema_versions": {
            "launch_manifest": 2,
            "runtime_config": 2,
            "training_summary": 5,
            "exit_receipt": 5,
            "predecessor_receipt_binding": 1,
            "matched_panel": 2,
        },
    }
    registry = SimpleNamespace(
        data={"common_training": {"initialization": copy.deepcopy(initialization)}}
    )
    return common, registry


@pytest.mark.parametrize(
    ("field", "value", "error"),
    (
        ("optimizer_updates", True, "optimizer updates"),
        ("training_seed", True, "training seed"),
        ("accumulate_grad_batches", True, "accumulation steps"),
        ("effective_global_batch_size", True, "effective global batch"),
    ),
)
def test_common_training_rejects_boolean_integer_fields(
    field: str, value: object, error: str
) -> None:
    common, registry = _common_training()
    common[field] = value
    with pytest.raises(scale.ScaleUpValidationError, match=error):
        scale._validate_common_training(common, screen_registry=registry)


def test_common_training_rejects_numeric_boolean_and_boolean_schema() -> None:
    common, registry = _common_training()
    common["gpu_safety_policy"]["compute_mode_prohibited_allowed"] = 0
    with pytest.raises(scale.ScaleUpValidationError, match="GPU safety policy"):
        scale._validate_common_training(common, screen_registry=registry)

    common, registry = _common_training()
    common["artifact_schema_versions"]["launch_manifest"] = True
    with pytest.raises(scale.ScaleUpValidationError, match="artifact schema"):
        scale._validate_common_training(common, screen_registry=registry)


def test_members_reject_boolean_position_before_loading_configs() -> None:
    members = []
    for position, (variant, slug) in enumerate(
        zip(scale.EXPECTED_VARIANTS, scale.EXPECTED_SLUGS, strict=True)
    ):
        members.append(
            {
                "position": True if position == 0 else position,
                "slug": slug,
                "training_variant": variant,
                "prior_variant": scale.EXPECTED_PRIORS[variant],
                "comparison_role": scale.EXPECTED_ROLES[variant],
                "run_name": f"scaleup-w1-{slug}-{'b' * 12}",
                "output_directory": f"output/udlm/scaleup-w1-{slug}-{'b' * 12}",
                "config": {},
            }
        )
    with pytest.raises(scale.ScaleUpValidationError, match="member 0 position"):
        scale._validate_members(
            members,
            selected_config=_selected_config(),
            common={"gpu_count": 1},
            publication={
                "config_revision": "c" * 40,
                "framework_revision": "b" * 40,
                "config_paths": list(prepare._config_paths(1).values()),
            },
            loader=lambda root, path: b"",
            git_blob_loader=lambda revision, path: b"",
        )


def _json_reference(name: str, *, schema: int = 1) -> dict[str, object]:
    return {
        "root": "repository",
        "relative_path": f"experiments/udlm/screens/{name}.json",
        "sha256": "1" * 64,
        "size_bytes": 123,
        "schema_version": schema,
        "canonical_sha256": "2" * 64,
    }


def _validated_registry() -> scale.ValidatedScaleUpRegistry:
    members = []
    for position, (variant, slug) in enumerate(
        zip(scale.EXPECTED_VARIANTS, scale.EXPECTED_SLUGS, strict=True)
    ):
        members.append(
            {
                "position": position,
                "slug": slug,
                "training_variant": variant,
                "prior_variant": scale.EXPECTED_PRIORS[variant],
                "comparison_role": scale.EXPECTED_ROLES[variant],
                "run_name": f"scale-{slug}",
                "output_directory": f"output/udlm/scale-{slug}",
                "config": {
                    "root": "repository",
                    "relative_path": f"experiments/udlm/{slug}.json",
                    "sha256": str(position + 3) * 64,
                    "size_bytes": 456,
                    "canonical_sha256": str(position + 6) * 64,
                },
            }
        )
    authority = {
        "screen_registry": _json_reference("registry", schema=2),
        "scheduler_evidence": _json_reference("scheduler_evidence"),
        "scheduler_selection": _json_reference("scheduler_selection"),
        "conditioning_evidence": _json_reference("conditioning_evidence"),
        "conditioning_selection": _json_reference("conditioning_selection"),
    }
    return scale.ValidatedScaleUpRegistry(
        data={
            "publication": {"config_revision": "a" * 40},
            "screen_authority": authority,
            "selected_design": {
                "scheduler_arm_id": "E-L0",
                "conditioning_arm_id": "E-A1",
            },
            "members": members,
            "common_training": {"gpu_count": 3},
        },
        relative_path=scale.PurePosixPath(
            "experiments/udlm/protocols/selection_bound_scale_up_registry_gpu3.json"
        ),
        raw_sha256="b" * 64,
        raw_size_bytes=789,
        canonical_sha256="c" * 64,
        screen_registry=None,  # type: ignore[arg-type]
        scheduler_selection={"selected_arm_id": "E-L0"},
        conditioning_selection={"selected_arm_id": "E-A1"},
    )


def test_manifest_binding_is_exact_and_accepted_by_pilot_compatibility() -> None:
    registry = _validated_registry()
    binding = scale.expected_manifest_binding(registry, position=1)
    assert scale.validate_manifest_binding(binding, registry=registry, position=1) == binding
    assert (
        pilot.validate_selection_bound_scale_up(
            binding,
            expected_training_variant="schedule_uniform",
            expected_position=1,
            expected_world_size=3,
            expected_resolved_config_sha256="7" * 64,
        )
        == binding
    )
    tampered = copy.deepcopy(binding)
    tampered["selected_design"]["scheduler_arm_id"] = "E-L1"
    with pytest.raises(scale.ScaleUpValidationError):
        scale.validate_manifest_binding(tampered, registry=registry, position=1)

    tampered = copy.deepcopy(binding)
    tampered["member"]["position"] = True
    with pytest.raises(scale.ScaleUpValidationError):
        scale.validate_manifest_binding(tampered, registry=registry, position=1)
