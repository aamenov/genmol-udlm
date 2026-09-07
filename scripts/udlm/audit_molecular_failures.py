"""CPU audit of frozen V5/V6 molecule artifacts; no sampling or model loads.

By default write JSON to stdout only. --output-directory publishes audit.json
and configurations.csv exclusively into a fresh directory. Recorded QED/SA and
decode results come from upstream exact independent rescoring; only lexical
syntax, selected-molecule size and auxiliary raw RDKit failures are new here.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import logging
from pathlib import Path
import re
import statistics

from rdkit import Chem, rdBase
from rdkit.Chem import Descriptors


REPORT_SHA256 = {
    5: "f05c0a16e07696a91df80fe4f60fa6ed9011de19c67323b90614c764216b1092",
    6: "36675c3743d8cee7073f31d21ed048c4342ae60f9f693475aac4147a5d4b0792",
}
COUNT_FIELDS = (
    "n",
    "strict_valid",
    "repaired_valid",
    "strict_failure",
    "odd_ring",
    "bad_parentheses",
    "both_flags",
    "syntax_flag",
    "syntax_strict_failure",
    "flagged_strict_valid",
    "recovered",
    "largest_component",
    "released_unique",
    "released_quality",
    "strict_unique",
    "strict_quality",
    "released_low_qed_only",
    "released_high_sa_only",
    "released_both_quality_fail",
    "strict_low_qed_only",
    "strict_high_sa_only",
    "strict_both_quality_fail",
    "raw_safe_error",
)
DEFINITIONS = {
    "pooled_scope": "Counts across 18 configurations and 36 seed runs are a heterogeneous diagnostic census, not a single-model benchmark estimate or independent repeated molecules.",
    "ring_parity": "Remove complete bracket-atom substrings; recognize %(integer), %NN and single-digit ring labels; normalize label integers and flag any odd count. Even counts permit label reuse and do not establish chemical validity.",
    "parentheses": "After removing bracket atoms and %(integer) ring labels, flag negative running parenthesis depth or nonzero final depth.",
    "syntax_scope": "Necessary lexical diagnostics, not a complete parser. Counts overlap. Zero flagged strict-valid rows is an empirical observation on this frozen sample.",
    "strict": "Saved exact-rescored SAFE fix=False decode plus RDKit validation, with no repair or largest-component selection.",
    "repaired": "Saved exact-rescored SAFE fix=True decode with released largest-component selection by SMILES string length.",
    "quality": "Distinct valid molecules with QED>=0.6 and SA<=4 divided by requests; uniqueness is evaluated within each seed, not globally across configurations. QED is drug-likeness; lower SA means easier synthetic accessibility.",
    "size": "RDKit MolWt and heavy-atom count of saved released selected SMILES, among first-unique molecules per seed. Associations with repair/recovery are not causal effects.",
    "auxiliary_rdkit": "Direct RDKit MolFromSmiles(raw_safe) for strict failures without lexical flags; classify first matching logged error category. This auxiliary parser call is not a redefinition of upstream strict SAFE decoding.",
    "control_audit": "Sum upstream final_sampled_editable_positions control counts; editable count is the sampler-input MASK count. This does not inspect unrecorded intermediate diffusion states.",
    "comparison": "Two seeds and128 requests/configuration, selected from multiple settings. Within-arm Gibbs flag counts are descriptive and do not establish structural improvement; V5 andV6 have different seeds.",
}


def lexical_flags(raw_safe):
    text = re.sub(r"\[[^\]]*\]", "", raw_safe)
    labels = re.findall(r"%\(\d+\)|%\d{2}|\d", text)
    counts = Counter(int(re.sub(r"\D", "", label)) for label in labels)
    odd_ring = any(count % 2 for count in counts.values())
    depth, negative_prefix = 0, False
    for character in re.sub(r"%\(\d+\)", "", text):
        depth += (character == "(") - (character == ")")
        negative_prefix |= depth < 0
    return {"odd_ring": odd_ring, "bad_parentheses": negative_prefix or depth != 0}


def row_diagnostics(row):
    flags = lexical_flags(row["raw_safe"])
    strict, repaired = bool(row["strict_smiles"]), bool(row["released_smiles"])
    syntax = flags["odd_ring"] or flags["bad_parentheses"]
    result = dict(
        flags,
        n=1,
        strict_valid=strict,
        repaired_valid=repaired,
        strict_failure=not strict,
        syntax_flag=syntax,
        syntax_strict_failure=syntax and not strict,
        flagged_strict_valid=syntax and strict,
        both_flags=flags["odd_ring"] and flags["bad_parentheses"],
        recovered=row["released_was_recovered"] == "True",
        largest_component=row["released_largest_component_applied"] == "True",
        raw_safe_error=bool(row["raw_safe_error"]),
    )
    for prefix, valid in (("strict", strict), ("released", repaired)):
        unique = valid and row[prefix + "_is_first_unique"] == "True"
        low = unique and float(row[prefix + "_qed"]) < 0.6
        high = unique and float(row[prefix + "_sa"]) > 4
        quality = unique and not low and not high
        if quality != (row[prefix + "_quality_counted"] == "True"):
            raise ValueError("Saved quality flag contradicts QED/SA/uniqueness")
        result.update(
            {
                prefix + "_unique": unique,
                prefix + "_quality": quality,
                prefix + "_low_qed_only": low and not high,
                prefix + "_high_sa_only": high and not low,
                prefix + "_both_quality_fail": low and high,
            }
        )
    return result


def tally(rows):
    return {field: sum(int(row[field]) for row in rows) for field in COUNT_FIELDS}


class Inputs:
    def __init__(self, workspace):
        self.workspace = workspace.resolve()
        self.payloads = {}

    def read(self, path, *, expected=None, role):
        path = path.resolve()
        if not path.is_relative_to(self.workspace):
            raise ValueError(f"Input outside declared workspace: {path}")
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if expected is not None and digest != expected:
            raise ValueError(f"Input hash mismatch: {path}")
        self.payloads[path] = (payload, role)
        return payload

    def manifest(self):
        result = []
        for path, (payload, role) in sorted(self.payloads.items()):
            if path.read_bytes() != payload:
                raise ValueError(f"Input changed during audit: {path}")
            result.append(
                {
                    "workspace_relative_path": str(path.relative_to(self.workspace)),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "size_bytes": len(payload),
                    "role": role,
                }
            )
        return result


def parse_rows(payload, expected_count):
    rows = list(csv.DictReader(io.StringIO(payload.decode())))
    if len(rows) != expected_count or [
        int(row["sample_index"]) for row in rows
    ] != list(range(expected_count)):
        raise ValueError("Unexpected raw CSV row count or sample indices")
    return rows


def size_statistics(rows):
    if not rows:
        return {"n": 0}
    return {
        "n": len(rows),
        "low_qed": sum(row["qed"] < 0.6 for row in rows),
        "high_sa": sum(row["sa"] > 4 for row in rows),
        "quality": sum(row["qed"] >= 0.6 and row["sa"] <= 4 for row in rows),
        **{
            f"median_{key}": statistics.median(row[key] for row in rows)
            for key in ("qed", "mw", "heavy_atoms")
        },
    }


def residual_parse_categories(raw_strings):
    counts = Counter()
    logger = logging.getLogger("rdkit")
    handlers, level, propagate = logger.handlers, logger.level, logger.propagate
    rdBase.LogToPythonLogger()
    logger.handlers, logger.propagate = [], False
    logger.setLevel(logging.ERROR)
    try:
        for raw in raw_strings:
            stream = io.StringIO()
            handler = logging.StreamHandler(stream)
            logger.addHandler(handler)
            molecule = Chem.MolFromSmiles(raw)
            logger.removeHandler(handler)
            message = stream.getvalue().lower()
            if molecule is not None:
                category = "raw_rdkit_accepts"
            elif "kekul" in message:
                category = "kekulization"
            elif "valence" in message:
                category = "valence"
            elif "non-ring atom" in message:
                category = "nonring_aromatic"
            elif "already exists" in message or "duplicate" in message:
                category = "duplicate_bond"
            else:
                category = "other_parse"
            counts[category] += 1
    finally:
        logger.handlers, logger.level, logger.propagate = handlers, level, propagate
        rdBase.LogToCppStreams()
    return dict(sorted(counts.items()))


def audit(artifact_root, workspace):
    artifact_root, workspace = artifact_root.resolve(), workspace.resolve()
    inputs = Inputs(workspace)
    inputs.read(Path(__file__), role="audit generator source")
    records, per_seed, residual = [], [], []
    size_groups = defaultdict(list)
    controls, editable_positions = Counter(), 0
    reports = {}
    for version in (5, 6):
        report_path = (
            artifact_root
            / f"output/udlm/engineering_v{version}_reports/complete/report.json"
        )
        report = json.loads(
            inputs.read(
                report_path,
                expected=REPORT_SHA256[version],
                role=f"frozen V{version} independent report",
            )
        )
        reports[version] = report
        expected_runs = 24 if version == 5 else 12
        if (
            report["status"] != "complete"
            or report["superiority_established"] is not False
            or report["accounting"]["status_counts"] != {"completed": expected_runs}
            or report["accounting"]["independently_rescored_requests"]
            != expected_runs * 64
        ):
            raise ValueError(
                "Expected complete independently rescored engineering report"
            )
        for run in report["runs"]:
            if run["independent_rescore"]["status"] != "exact_match":
                raise ValueError("Run lacks exact independent rescoring")
            for name, artifact in run["artifacts"].items():
                payload = inputs.read(
                    artifact_root / artifact["relative_path"],
                    expected=artifact["sha256"],
                    role=f"V{version} {name}",
                )
                if name == "raw_samples.csv":
                    rows = parse_rows(payload, 64)
            control = run["independent_rescore"]["identity"][
                "sampled_token_control_audit"
            ]
            controls.update(
                control["control_token_counts"]["final_sampled_editable_positions"]
            )
            editable_positions += control["control_token_counts"][
                "sampler_input_all_positions"
            ]["mask"]
            run_records = []
            for row in rows:
                record = dict(
                    row_diagnostics(row),
                    version=version,
                    arm=run["arm_id"],
                    config=run["config_id"],
                    seed=run["seed"],
                    temperature=run["generation_protocol"]["temperature"],
                    gibbs_corrector=run["config"]["effective"].get(
                        "gibbs_corrector", False
                    ),
                )
                run_records.append(record)
                if record["strict_failure"] and not record["syntax_flag"]:
                    residual.append(row["raw_safe"])
                if record["released_unique"]:
                    molecule = Chem.MolFromSmiles(row["released_smiles"])
                    if molecule is None:
                        raise ValueError(
                            "Saved released-valid SMILES fails RDKit parsing"
                        )
                    size = {
                        "qed": float(row["released_qed"]),
                        "sa": float(row["released_sa"]),
                        "mw": Descriptors.MolWt(molecule),
                        "heavy_atoms": molecule.GetNumHeavyAtoms(),
                    }
                    keys = [
                        "all_unique",
                        (
                            "largest_component"
                            if record["largest_component"]
                            else "no_largest_component"
                        ),
                        "recovered" if record["recovered"] else "strict_already_valid",
                    ]
                    if version == 6 and run["config_id"] == "s_t050_gibbs":
                        keys.append("v6_best_s_gibbs")
                    for key in keys:
                        size_groups[key].append(size)
            counts = tally(run_records)
            for branch, prefix in (
                ("released_comparable", "released"),
                ("strict", "strict"),
            ):
                metrics = run["metrics"][branch]
                valid_key = "repaired_valid" if prefix == "released" else "strict_valid"
                if (
                    counts[valid_key] != metrics["valid_count"]
                    or counts[prefix + "_unique"] != metrics["unique_count"]
                    or counts[prefix + "_quality"] != metrics["quality_count"]
                ):
                    raise ValueError("CSV counts disagree with report metrics")
            per_seed.append(
                dict(
                    version=version,
                    arm=run["arm_id"],
                    config=run["config_id"],
                    seed=run["seed"],
                    **counts,
                )
            )
            records.extend(run_records)
    baseline_path = workspace / "output/benchmarks/denovo_50000/report/aggregate.json"
    baseline = json.loads(
        inputs.read(baseline_path, role="local MDLM baseline aggregate")
    )
    baseline_records = []
    for run in baseline["seed_runs"]:
        payload = inputs.read(
            Path(run["raw_samples_path"]),
            expected=run["raw_samples_sha256"],
            role=f"baseline seed {run['seed']} raw CSV",
        )
        baseline_records.extend(
            row_diagnostics(row) for row in parse_rows(payload, 1000)
        )
    if baseline["status"] != "completed" or len(baseline_records) != 3000:
        raise ValueError("Unexpected baseline population")
    configurations = []
    for version, name in sorted({(r["version"], r["config"]) for r in records}):
        selected = [
            r for r in records if (r["version"], r["config"]) == (version, name)
        ]
        first = selected[0]
        configurations.append(
            dict(
                version=version,
                arm=first["arm"],
                config=name,
                temperature=first["temperature"],
                gibbs_corrector=first["gibbs_corrector"],
                seeds=sorted({r["seed"] for r in selected}),
                **tally(selected),
            )
        )
    training = {}
    for arm in ("R", "S", "E"):
        entry = next(
            e
            for e in reports[6]["protocol"]["configuration"]["entries"]
            if e["arm_id"] == arm
        )
        path = (
            artifact_root
            / Path(entry["checkpoint"]).parent.parent
            / "training_summary.json"
        )
        summary = json.loads(
            inputs.read(path, role=f"{arm} historical adaptation budget")
        )
        if (
            summary["status"] != "completed"
            or summary["final_checkpoint"]["sha256"] != entry["checkpoint_sha256"]
        ):
            raise ValueError("Training summary/checkpoint identity mismatch")
        training[arm] = summary["training_accounting"]
    official_root = artifact_root / "tmp/official_udlm_edb0f8c"
    for name in ("scripts/train_qm9_no-guidance.sh", "diffusion.py"):
        inputs.read(official_root / name, role="pinned official UDLM reference code")
    totals = tally(records)
    return {
        "schema_version": 1,
        "status": "complete",
        "definitions": DEFINITIONS,
        "software": {"rdkit": rdBase.rdkitVersion},
        "pooled_diagnostic": totals,
        "syntax_fraction_of_strict_failures": totals["syntax_strict_failure"]
        / totals["strict_failure"],
        "by_version": {
            str(v): tally([r for r in records if r["version"] == v]) for v in (5, 6)
        },
        "v5_by_temperature": {
            str(t): tally(
                [r for r in records if r["version"] == 5 and r["temperature"] == t]
            )
            for t in (0.5, 0.7, 0.85, 1.0)
        },
        "configurations": configurations,
        "per_seed": per_seed,
        "control_token_audit": {
            "final_editable_counts": dict(controls),
            "editable_token_count": editable_positions,
            "mean_editable_length": editable_positions / len(records),
        },
        "selected_molecule_size_associations": {
            k: size_statistics(v) for k, v in sorted(size_groups.items())
        },
        "unflagged_strict_failure_direct_rdkit": {
            "n": len(residual),
            "categories": residual_parse_categories(residual),
        },
        "local_baseline_diagnostic": tally(baseline_records),
        "historical_training": training,
        "inputs": inputs.manifest(),
        "superiority_established": False,
        "new_training_or_generation": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path)
    args = parser.parse_args()
    result = audit(args.artifact_root, args.workspace_root)
    encoded = (
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    if args.output_directory is None:
        print(encoded.decode(), end="")
        return
    destination = args.output_directory.resolve()
    if (
        not destination.is_relative_to(args.workspace_root.resolve())
        or destination.exists()
    ):
        raise ValueError(
            "Output must be a fresh directory inside the declared workspace"
        )
    csv_output = io.StringIO()
    rows = [
        dict(row, seeds="/".join(map(str, row["seeds"])))
        for row in result["configurations"]
    ]
    writer = csv.DictWriter(csv_output, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    destination.mkdir(parents=True, exist_ok=False)
    for name, payload in (
        ("audit.json", encoded),
        ("configurations.csv", csv_output.getvalue().encode()),
    ):
        with (destination / name).open("xb") as stream:
            stream.write(payload)
    print(
        json.dumps(
            {
                "output_directory": str(destination),
                "rows": result["pooled_diagnostic"]["n"],
                "syntax_fraction_of_strict_failures": result[
                    "syntax_fraction_of_strict_failures"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
