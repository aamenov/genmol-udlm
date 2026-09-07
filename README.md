<h1 align="center">GenMol-UDLM</h1>

This repository is an independent research reimplementation and extension of
[NVIDIA GenMol](https://github.com/NVIDIA-BioNeMo/genmol). It replaces the
absorbing masked-diffusion path with Uniform Diffusion Language Model (UDLM)
controls and molecular variants derived from the
[UDLM paper](https://arxiv.org/abs/2412.10193) and its
[reference implementation](https://github.com/kuleshov-group/discrete-diffusion-guidance).
It is not NVIDIA's official repository and is not affiliated with NVIDIA or the
UDLM authors.

> **Research status — 2026-09-07.** Temperature, fixed-budget Gibbs and training-objective
> studies completed 44 runs and independently rescored all 3,104 requested
> samples. Best selected pilot quality is **57.03%**, versus **85.8%** for the
> local MDLM baseline. **No superiority has been established.** Each exploratory
> configuration used only two seeds of 64 or 100 requests; the baseline used
> three seeds of 1,000. Final UDLM evaluation seeds remain reserved.
>
> See the [43-page study report](output/udlm/study_overview_v9_20260907/study_overview.pdf),
> [temperature results](experiments/udlm/results/engineering_v5.md), and
> [Gibbs results](experiments/udlm/results/engineering_v6.md). All R/S/E
> checkpoints received 1,000 batch-16 updates from the common MDLM EMA.
> An opt-in [CE clean-denoiser adaptation](docs/udlm_ce_implementation.md) is
> implemented and CPU-tested; its matched molecular comparison is complete. The
> [batch-128 resource pilot](experiments/udlm/results/engineering_v7_throughput.md)
> passed. The prospective [matched CT/CE study](experiments/udlm/protocols/engineering_v8_objectives.json)
> specifies 1,000 batch-128 updates per arm from the same MDLM EMA. CT reached
> step 1,000, but a [process-exit incident](experiments/udlm/results/engineering_v8_ct_exit_incident.md)
> stopped the controller before CE; its checkpoint passed a separate CPU audit.
> The distinct [CE follow-up completed](experiments/udlm/results/engineering_v8b_ce_training.md)
> all 1,000 updates with the original matched training setup and passed checkpoint
> validation. The paired [V9 molecular evaluation](experiments/udlm/protocols/engineering_v9_objectives.json)
> compared CT/CE at temperatures 1.0/0.5, using 800 requests and 128 predictor
> evaluations per molecule. [V9 results](experiments/udlm/results/engineering_v9.md)
> reached 54.5% best quality; CE changed quality by -5.5pp at T1 and -2.0pp at T0.5.
> The [paired report](output/udlm/engineering_v9_reports/paired_complete/report.pdf)
> includes both decoding rules and all signed contrasts. A fresh
> [128-versus-512 predictor comparison](experiments/udlm/protocols/engineering_v10_resolution.json)
> is ready to launch. Current authorization is
> at most two dynamically selected GPUs below 10% utilization.

The main educational implementation is
[`genmol_from_scratch.ipynb`](genmol_from_scratch.ipynb); exact experiment
state, hashes, caveats, and the next safe action are in
[`PROJECT_CONTEXT.md`](PROJECT_CONTEXT.md). The standalone repository is
[aamenov/genmol-udlm](https://github.com/aamenov/genmol-udlm).

## Archived v4 generation plan

The D diagnostic generated 32 rows and then failed completion validation due
to a missing normalized top-p default in the expected configuration. Its
failure and raw rows are preserved; no ranked stage followed. The design below
is historical. V5 and V6 were separate engineering studies, not a reclassification
or continuation of that failed campaign.

Protocol v4 binds three immutable W=1 checkpoints: release-compatible uniform
R (`d0310d2e…`), schedule-consistent uniform S (`100f467b…`), and
schedule-consistent empirical-prior E (`dce870e8…`). All three restart from the
same verified MDLM EMA, use the selected E-L1 schedule and E-A1 conditioner,
and complete 1,000 optimizer updates. This is training/provenance evidence, not
a molecule benchmark. The frozen
`experiments/udlm/protocols/de_novo_superiority_v4.json` has raw SHA-256
`9432360dad30a01de7ededf62db77470af9a0b8297fc78f06d330dbc73e826b7`
and canonical SHA-256
`4edb0d193fcedc76913905220f3431fed8f0dd416f900c8da5f433a6071d4bfa`.

The prospective sampling universe is 36 settings:

- checkpoint arm in `{R, S, E}`;
- raw-LOO softmax temperature in `{0.50, 0.70, 0.85, 1.00}`; and
- `raw_loo_top_p` in `{1.00, 0.98, 0.95}`.

Temperature and stable nucleus filtering act on the active-alphabet raw LOO
probabilities before the exact UDLM reverse bridge. The crossing token is
retained, ties use ascending active token ID, and the final reverse posterior is
never truncated. `raw_loo_top_p=1.0` takes the literal pre-v4 path; uniform and
categorical posterior tensors and cloned-RNG samples must be exactly equal to
the old behavior.

The campaign is deliberately small-first. D runs one structural-only
seed-1100 × 32 diagnostic. A screens 12 temperature settings at seed 1101 × 32;
B tests 18 promoted temperature/top-p interactions at seed 1102 × 64; C checks
six survivors at held-out engineering seed 1103 × 96. The eligible stage alone
uses seeds 1000 and 1001 × 256 for the three arm survivors. One global winner is
then frozen before final seeds 0, 1, and 2 × 1,000. Ranking uses raw unrounded
released-compatible quality, then diversity, then ASCII config and attempt IDs.
Every scheduled child must terminate; failures remain disclosed and unrankable,
with no retry, substitution, or cross-stage score pooling.
Before any ranked stage advances, its raw, unrounded metrics are recomputed by
the fresh CPU independent rescore. Every non-D child starts strictly after its
predecessor decision completes, every child terminates before its own decision,
and the eligible decision completes before the candidate lock is built.

Candidate benchmark schema 8 embeds bounded sampler-input IDs, final sampled
IDs, editable bits, and recomputed control-token counts in `summary.json`, then
checks that tokenizer decoding exactly matches CSV raw text. Publication is
exclusive and completion-last: `raw_samples.csv` is linked before
`summary.json`, with identity-owned rollback, a retained output-directory
descriptor, and a repository-global generation lease.

At every real launch, the full GPU inventory is inspected and selected UUIDs
are immediately re-probed. A card is eligible only below 10% utilization
(exactly 10% is not idle), with at least 30,000 MiB free and non-prohibited
compute mode. Active processes are recorded and never interrupted. The archived
plan previously allowed three GPUs; current authorization is at most two GPUs.
Physical GPU 0 is never assumed, and long jobs use named detached `tmux` sessions
with logs under
`output/logs/`.

The publication firewall is acyclic: F contains framework/tests/docs/notebook
and v4; clean pushed C adds exactly 33 new YAMLs while reusing three historical
identity YAMLs; clean pushed G adds only the config registry, and every GPU
child binds exact G. After all 43 pre-final children terminate, a CPU
materializer independently validates and rescores all outcomes, publishes 43
tracked envelopes plus a completion-last manifest, and force-adds exactly that
manifest-derived closure. The addition-only EVIDENCE commit is the exact sole
child of G. It is followed by decision-only, deterministic-ledger-only, and
deterministically-built candidate-lock-only commits. There is no hand-authored
lock draft. Final seeds are forbidden until the lock is clean and pushed.

## Exact post-G operational handoff

This sequence must run in the original artifact-bearing worktree. A fresh Git
clone does not contain the ignored terminal checkpoints, child outputs, live
envelopes, or stage decisions; it is valid only if those artifacts were
restored byte-for-byte and independently pass every check. Replace the two
digest marker strings below with the raw and canonical registry SHA-256 values
verified at clean pushed G. `G` itself is captured from that clean pushed
revision.

```bash
cd /home/aidar.alimbayev/Documents/genmolv2/run_sources/udlm_genmol_worktree
mkdir -p output/logs
G="$(git rev-parse HEAD)"
REGISTRY_RAW='REPLACE_WITH_REGISTRY_RAW_SHA256'
REGISTRY_CANONICAL='REPLACE_WITH_REGISTRY_CANONICAL_SHA256'
```

Authorize one bounded prefix at a time. D is internally fixed to concurrency
one; the archived plan allowed three children, but current authorization caps
all work for this goal at two GPUs. Do not launch A until D's
decision is terminal, B until A is terminal, C until B is terminal, or
`eligible` until C is terminal.

```bash
tmux new-session -d -s genmol-udlm-v4-d "cd /home/aidar.alimbayev/Documents/genmolv2/run_sources/udlm_genmol_worktree && env -u CUDA_VISIBLE_DEVICES /home/aidar.alimbayev/Documents/genmolv2/.venv/bin/python scripts/udlm/launch_candidate_campaign.py --registry experiments/udlm/protocols/de_novo_candidate_config_registry_v1.json --expected-registry-sha256 $REGISTRY_RAW --expected-registry-canonical-sha256 $REGISTRY_CANONICAL --through-stage D > output/logs/v4-campaign-D-controller.log 2>&1"
tmux new-session -d -s genmol-udlm-v4-a "cd /home/aidar.alimbayev/Documents/genmolv2/run_sources/udlm_genmol_worktree && env -u CUDA_VISIBLE_DEVICES /home/aidar.alimbayev/Documents/genmolv2/.venv/bin/python scripts/udlm/launch_candidate_campaign.py --registry experiments/udlm/protocols/de_novo_candidate_config_registry_v1.json --expected-registry-sha256 $REGISTRY_RAW --expected-registry-canonical-sha256 $REGISTRY_CANONICAL --through-stage A > output/logs/v4-campaign-A-controller.log 2>&1"
tmux new-session -d -s genmol-udlm-v4-b "cd /home/aidar.alimbayev/Documents/genmolv2/run_sources/udlm_genmol_worktree && env -u CUDA_VISIBLE_DEVICES /home/aidar.alimbayev/Documents/genmolv2/.venv/bin/python scripts/udlm/launch_candidate_campaign.py --registry experiments/udlm/protocols/de_novo_candidate_config_registry_v1.json --expected-registry-sha256 $REGISTRY_RAW --expected-registry-canonical-sha256 $REGISTRY_CANONICAL --through-stage B > output/logs/v4-campaign-B-controller.log 2>&1"
tmux new-session -d -s genmol-udlm-v4-c "cd /home/aidar.alimbayev/Documents/genmolv2/run_sources/udlm_genmol_worktree && env -u CUDA_VISIBLE_DEVICES /home/aidar.alimbayev/Documents/genmolv2/.venv/bin/python scripts/udlm/launch_candidate_campaign.py --registry experiments/udlm/protocols/de_novo_candidate_config_registry_v1.json --expected-registry-sha256 $REGISTRY_RAW --expected-registry-canonical-sha256 $REGISTRY_CANONICAL --through-stage C > output/logs/v4-campaign-C-controller.log 2>&1"
tmux new-session -d -s genmol-udlm-v4-eligible "cd /home/aidar.alimbayev/Documents/genmolv2/run_sources/udlm_genmol_worktree && env -u CUDA_VISIBLE_DEVICES /home/aidar.alimbayev/Documents/genmolv2/.venv/bin/python scripts/udlm/launch_candidate_campaign.py --registry experiments/udlm/protocols/de_novo_candidate_config_registry_v1.json --expected-registry-sha256 $REGISTRY_RAW --expected-registry-canonical-sha256 $REGISTRY_CANONICAL --through-stage eligible > output/logs/v4-campaign-eligible-controller.log 2>&1"
```

After all five stage decisions are terminal, stay at exact G and materialize,
then stage, the evidence closure. The staging flag belongs to the materializer,
uses exact `git add -f` paths derived from the manifest, verifies the index, and
does not commit or push.

```bash
/home/aidar.alimbayev/Documents/genmolv2/.venv/bin/python scripts/udlm/materialize_candidate_evidence.py --expected-source-revision "$G" --expected-registry-sha256 "$REGISTRY_RAW" --expected-registry-canonical-sha256 "$REGISTRY_CANONICAL"
/home/aidar.alimbayev/Documents/genmolv2/.venv/bin/python scripts/udlm/materialize_candidate_evidence.py --expected-source-revision "$G" --expected-registry-sha256 "$REGISTRY_RAW" --expected-registry-canonical-sha256 "$REGISTRY_CANONICAL" --stage-published-evidence
git commit -m 'Publish registered v4 campaign evidence'
git push --no-thin origin codex/udlm-genmol-scale-retry1
EVIDENCE="$(git rev-parse HEAD)"
```

Build each authority artifact only from its clean pushed predecessor. The lock
phase is the deterministic schema-2 builder; do not create a draft manually.

```bash
/home/aidar.alimbayev/Documents/genmolv2/.venv/bin/python scripts/udlm/prepare_candidate_authority.py --phase decision --expected-source-revision "$EVIDENCE" --expected-registry-sha256 "$REGISTRY_RAW" --expected-registry-canonical-sha256 "$REGISTRY_CANONICAL" --registry-revision "$G"
git add experiments/udlm/candidates/candidate_decision.json
git commit -m 'Publish registered v4 candidate decision'
git push --no-thin origin codex/udlm-genmol-scale-retry1
DECISION="$(git rev-parse HEAD)"
/home/aidar.alimbayev/Documents/genmolv2/.venv/bin/python scripts/udlm/prepare_candidate_authority.py --phase ledger --expected-source-revision "$DECISION"
git add experiments/udlm/candidates/candidate_ledger.json
git commit -m 'Publish deterministic v4 candidate ledger'
git push --no-thin origin codex/udlm-genmol-scale-retry1
LEDGER="$(git rev-parse HEAD)"
/home/aidar.alimbayev/Documents/genmolv2/.venv/bin/python scripts/udlm/prepare_candidate_authority.py --phase lock --expected-source-revision "$LEDGER"
git add experiments/udlm/candidates/candidate_lock.json
git commit -m 'Lock registered v4 candidate before final evaluation'
git push --no-thin origin codex/udlm-genmol-scale-retry1
LOCK="$(git rev-parse HEAD)"
```

Finally, read `LOCK_CHECKPOINT` from `training.checkpoint.relative_path`,
`LOCK_CONFIG` from `inference.evaluation_config_relative_path`, and
`LOCK_OUTPUT_ROOT` as the common parent of the three
`inference.final_run_directories_by_seed` paths in the committed lock. Set
`GPU_COUNT` to 1, 2, or 3. The launcher requires every supplied field to equal
the lock and validates the committed schema-2 lock before mutation or GPU work.

```bash
LOCK_CHECKPOINT='REPLACE_WITH_EXACT_LOCK_CHECKPOINT_PATH'
LOCK_CONFIG='REPLACE_WITH_EXACT_LOCK_CONFIG_PATH'
LOCK_OUTPUT_ROOT='REPLACE_WITH_EXACT_LOCK_OUTPUT_ROOT'
GPU_COUNT='REPLACE_WITH_INTEGER_1_TO_3'
tmux new-session -d -s genmol-udlm-v4-final "cd /home/aidar.alimbayev/Documents/genmolv2/run_sources/udlm_genmol_worktree && env -u CUDA_VISIBLE_DEVICES /home/aidar.alimbayev/Documents/genmolv2/.venv/bin/python scripts/exps/denovo/launch_benchmark.py --checkpoint $LOCK_CHECKPOINT --config $LOCK_CONFIG --num-samples 1000 --seeds 0 1 2 --output-root $LOCK_OUTPUT_ROOT --candidate-lock experiments/udlm/candidates/candidate_lock.json --gpu-count $GPU_COUNT --max-utilization-percent 10 --min-free-memory-mib 30000 --log-root output/logs > output/logs/v4-final-controller.log 2>&1"
```

For every launch, utilization must be strictly below 10% (10% is busy), free
memory must be at least 30,000 MiB, physical IDs are never assumed, selected
UUIDs are immediately re-probed, and active processes are recorded and never
interrupted.

## Upstream GenMol reference documentation

The remainder of this README is retained from NVIDIA's released GenMol
documentation for scientific comparison. First-person claims below describe
the original GenMol work, not this experimental fork.

<p align="center">
    <img width="750" src="assets/concept.png"/>
</p>


## Contribution
+ We introduce GenMol, a model for unified and versatile molecule generation by building masked discrete diffusion that generates SAFE molecular sequences.
+ We propose fragment remasking, an effective strategy for exploring chemical space using molecular fragments as the unit of exploration.
+ We propose molecular context guidance (MCG), a guidance scheme for GenMol to effectively utilize molecular context information.
+ We validate the efficacy and versatility of GenMol on a wide range of drug discovery tasks.

## 🚀 News

#### 2025/10/15
We introduce GenMol V2, trained with an extended SAFE syntax, demonstrating improved performance in *de novo* and fragment-constrained generation. Please refer to the section below: [GenMol V2: GenMol with Extended SAFE Syntax](#-genmol-v2-genmol-with-extended-safe-syntax).

## Table of Contents
- [Installation](#installation)
- [GenMol V1](#genmol-v1)
  - [Training](#training)
  - [Training with User-defined Dataset](#optional-training-with-user-defined-dataset)
  - [*De Novo* Generation](#de-novo-generation)
  - [Fragment-constrained Generation](#fragment-constrained-generation)
  - [Goal-directed Hit Generation (PMO Benchmark)](#goal-directed-hit-generation-pmo-benchmark)
  - [Goal-directed Lead Optimization](#goal-directed-lead-optimization)
- [GenMol V2: GenMol with Extended SAFE Syntax](#-genmol-v2-genmol-with-extended-safe-syntax)
  - [Summary](#summary)
  - [Introduction](#introduction)
  - [Benchmarks](#benchmarks)
  - [Training](#training-1)
  - [*De Novo* Generation](#de-novo-generation-1)
  - [Fragment-constrained Generation](#fragment-constrained-generation-1)
- [License](#license)
- [Citation](#citation)

## 📦 Installation
Clone this repository:
```bash
git clone https://github.com/aamenov/genmol-udlm.git
cd genmol-udlm
```

Run the following command to install the dependencies:
```bash
bash env/setup.sh
```

<details>
<summary>Troubleshooting: ImportError: libXrender.so.1</summary>

Run the following command:
```bash
apt update && apt install -y libsm6 libxext6 && apt-get install -y libxrender-dev
```
</details>

<details>
<summary>Troubleshooting: ImportError: cannot import name '_CONFIG_FOR_DOC' from 'transformers.models.gpt2.modeling_gpt2'</summary>

Run the following command:
```bash
#!/bin/bash

# Use CONDA_PREFIX which points to current active environment
if [ -z "$CONDA_PREFIX" ]; then
    echo "Error: No conda environment is currently active"
    exit 1
fi

# Comment out all lines in the safe package __init__.py
sed -i 's/^/# /' "$CONDA_PREFIX/lib/python3.10/site-packages/safe/__init__.py"

# Import required packages
echo "from .converter import SAFEConverter, decode, encode" >> "$CONDA_PREFIX/lib/python3.10/site-packages/safe/__init__.py"

echo "Fixed safe package in environment: $CONDA_PREFIX"
```
</details>

## 🔬 GenMol V1
### Training
We provide the pretrained [checkpoint](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara/resources/genmol_v1). Place `model.ckpt` in the checkpoints directory and set the correct information in ./configs/base.yaml.

(Optional) To train GenMol from scratch, run the following command:
```bash
torchrun --nproc_per_node ${num_gpu} scripts/train.py hydra.run.dir=${save_dir} wandb.name=${exp_name}
```
Other hyperparameters can be adjusted in `configs/base.yaml`.<br>
The training used 8 NVIDIA A100 GPUs and took ~5 hours.

We suggest the use of [SAFE dataset V2](https://huggingface.co/datasets/datamol-io/safe-drugs) to train GenMol (Note V2 removes some invalid molecules/corrupted SAFE strings. The original data that was used by GenMol is available here [SAFE dataset V1](https://huggingface.co/datasets/datamol-io/safe-gpt/tree/b83175cd7394).

### (Optional) Training with User-defined Dataset
To use your own training dataset, first convert your SMILES dataset into SAFE by running the following command:
```bash
python scripts/preprocess_data.py ${input_path} ${data_path}
```
`${input_path}` is the path to the dataset file with a SMILES in each row. For example,
```
CCS(=O)(=O)N1CC(CC#N)(n2cc(-c3ncnc4[nH]ccc34)cn2)C1
NS(=O)(=O)c1cc2c(cc1Cl)NC(C1CC3C=CC1C3)NS2(=O)=O
...
```
`${data_path}` is the path of the processed dataset.

Then, set `data` in `base.yaml` to `${data_path}`.

### *De Novo* Generation
Run the following command to perform *de novo* generation:
```bash
python scripts/exps/denovo/run.py
```

<details>
<summary>Troubleshooting: _pickle.UnpicklingError: invalid load key, '<'</summary>

If you see this error, it is likely coming from `/miniconda3/envs/genmol/lib/python3.10/site-packages/tdc/chem_utils/oracle/oracle.py`, line 347, in readFragmentScores `_fscores = pickle.load(f)`

The root cause is a corrupted or incompletely downloaded pkl file for the SA score. The fix is simple: grab the correct files from the official RDKit repository:
https://github.com/rdkit/rdkit/tree/master/Contrib/SA_Score/fpscores.pkl.gz

Extract the downloaded file into the `genmol/oracle` directory.
</details>

The experiment in the paper used 1 NVIDIA A100 GPU.

### Fragment-constrained Generation
Run the following command to perform fragment-constrained generation:
```bash
python scripts/exps/frag/run.py
```

The experiment in the paper used 1 NVIDIA A100 GPU.

### Goal-directed Hit Generation (PMO Benchmark)

We provide the fragment vocabularies in the folder `scripts/exps/pmo/vocab`.

(Optional) Place [zinc250k.csv](https://www.kaggle.com/datasets/basu369victor/zinc250k) in the `data` folder, then run the following command to construct the fragment vocabularies and label the molecules with property labels:
```bash
python scripts/exps/pmo/get_vocab.py
```

Run the following command to perform goal-directed hit generation:
```bash
python scripts/exps/pmo/run.py -o ${oracle_name}
```
The generated molecules will be saved in `scripts/exps/pmo/main/genmol/results`.

Run the following command to evaluate the result:
```bash
python scripts/exps/pmo/eval.py ${file_name}
# e.g., python scripts/exps/pmo/eval.py scripts/exps/pmo/main/genmol/results/albuterol_similarity_0.csv
```

The experiment in the paper used 1 NVIDIA A100 GPU and took ~2-4 hours for each task.

### Goal-directed Lead Optimization
Run the following command to perform goal-directed lead optimization:
```bash
python scripts/exps/lead/run.py -o ${oracle_name} -i ${start_mol_idx} -d ${sim_threshold}
```
The generated molecules will be saved in `scripts/exps/lead/results`.

Run the following command to evaluate the result:
```bash
python scripts/exps/lead/eval.py ${file_name}
# e.g., python scripts/exps/lead/eval.py scripts/exps/lead/results/parp1_id0_thr0.4_0.csv
```

The experiment in the paper used 1 NVIDIA A100 GPU and took ~10 min for each task.

## 🚀 GenMol V2: GenMol with Extended SAFE Syntax (Angle-Brackets for Inter-Fragment Attachment Points)
### Summary: 
GenMol V2 introduces Extended SAFE Syntax, which uses *angle-brackets* for Inter-Fragment Attachment Points. This change improves performance for specific tasks, particularly one-step linker design.

### Introduction:
Following SAFE-GPT, GenMol performs two-step linker design in fragment-constrained generation, i.e., two molecules are respectively generated given each of two fragments and then combined later as a single molecule. However, users may prefer one-step generation that can condition the context of both fragments at the same time.

While GenMol shows versatile performance on various tasks, it shows low validity in some tasks, especially in one-step linker design. We attribute this to the standard SAFE syntax, which considers the `intra-fragment` (linked atoms are in the same fragment) and `inter-fragment` (links between two fragments) attachment points are not easy to distinguish. 

To this end, we propose an extended SAFE syntax that uses angle-brackets to distinguish intra-fragment attachment points from inter-fragment attachment points.

For example, a SAFE string
```
X1XXX1X2.X2X3XXXX3X4.X5XX5X4
```
has 1, 3, 5 as its intra-fragment attachment points, while 2 and 4 are inter-fragment attachment points. With the extended syntax it becomes:
```
X1XXX1X<1>.X<1>X1XXXX1X<2>.X1XX1X<2>
```
In this way, the links within a fragment (i.e., 1, 2, ...) are independent to links crossing fragments (i.e., <1>, <2>, ...) and the model can learn how to complete a SAFE more efficiently.

GenMol V2 trained with the extended SAFE syntax actually shows significantly improved performance on *de novo* and fragment-constrained generation! On goal-directed hit generation and lead optimization, GenMol V2 performs slightly worse than GenMol. This is because GenMol performs fragment remasking in these tasks, which changes only a small part of the entire molecular sequence, and therefore does not benefit from the extended SAFE syntax.

### Benchmarks

<h4 align="center">Table. De Novo Generation</h4>

| Model | Validity (%) | Uniqueness (%) | Quality (%) | Diversity |
| --- | --- | --- | --- | --- |
| GenMol | 100.0 | 99.7 | 84.6 | 0.818 |
| GenMol V2 | 100.0 | 97.8 | 89.7 | 0.830 |

<h4 align="center">Table. Fragment-constrained Generation</h4>

| Model | Task | Validity (%) | Uniqueness (%) | Quality (%) | Diversity | Distance |
| --- | --- | --- | --- | --- | --- | --- |
| GenMol | Linker design (1-step) | 16.7 | 93.9 | 4.3 | 0.529 | 0.573 |
| | Linker design | 100.0 | 83.7 | 21.9 | 0.547 | 0.563 |
| | Motif extension | 82.9 | 77.5 | 30.1 | 0.617 | 0.682 |
| | Scaffold decoration | 96.6 | 82.7 | 31.8 | 0.591 | 0.651 |
| | Superstructure generation | 97.5 | 83.6 | 34.8 | 0.599 | 0.762 |
| GenMol V2 | Linker design (1-step) | 81.8 | 87.1 | 28.6 | 0.566 | 0.545 |
| | Linker design | 100.0 | 76.6 | 18.4 | 0.512 | 0.539 |
| | Motif extension | 99.4 | 84.5 | 49.0 | 0.626 | 0.659 |
| | Scaffold decoration | 99.2 | 90.5 | 39.7 | 0.571 | 0.604 |
| | Superstructure generation | 99.7 | 89.8 | 39.0 | 0.551 | 0.769 |

<h4 align="center">Table. Goal-directed Hit Generation</h4>

| Model | PMO Sum Score |
| --- | --- |
| GenMol | 18.362 |
| GenMol V2 | 17.943 |

<h4 align="center">Table. Goal-directed Lead Optimization</h4>

| Model | Success rate (%) |
| --- | --- |
| GenMol | 86.7 |
| GenMol V2 | 80.0 |

### Training
We provide the trained GenMol V2 [checkpoint](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara/resources/genmol_v2?version=1.0). Place `model_v2.ckpt` in the checkpoints directory and set the correct information in ./configs/base.yaml.

(Optional) To train GenMol V2 from scratch, run the following command:
```bash
torchrun --nproc_per_node ${num_gpu} scripts/train.py hydra.run.dir=${save_dir} wandb.name=${exp_name} loader.global_batch_size=1024 training.use_bracket_safe=true
```
The training used 8 NVIDIA A100 GPUs.

### *De Novo* Generation
Run the following command to perform *de novo* generation using GenMol V2:
```bash
python scripts/exps/denovo/run.py -c scripts/exps/frag/hparams_v2.yaml
```

### Fragment-constrained Generation
Run the following command to perform fragment-constrained generation using GenMol V2:
```bash
python scripts/exps/frag/run.py -c scripts/exps/frag/hparams_v2.yaml
```

## License
Copyright @ 2025, NVIDIA Corporation. All rights reserved.<br>
The source code is made available under Apache-2.0.<br>
The model weights are made available under the [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/).

## 📝 Citation
This fork builds on the original GenMol and discrete-diffusion-guidance works;
please cite those papers when using their ideas or released code.
```BibTex
@article{lee2025genmol,
  title     = {GenMol: A Drug Discovery Generalist with Discrete Diffusion},
  author    = {Lee, Seul and Kreis, Karsten and Veccham, Srimukh Prasad and Liu, Meng and Reidenbach, Danny and Peng, Yuxing and Paliwal, Saee and Nie, Weili and Vahdat, Arash},
  journal   = {International Conference on Machine Learning},
  year      = {2025}
}

@article{schiff2024discreteguidance,
  title     = {Simple Guidance Mechanisms for Discrete Diffusion Models},
  author    = {Schiff, Yair and Sahoo, Subham Sekhar and Phung, Hao and Wang, Guanghan and Boshar, Sam and Dalla-torre, Hugo and de Almeida, Bernardo P and Rush, Alexander and Pierrot, Thomas and Kuleshov, Volodymyr},
  journal   = {arXiv preprint arXiv:2412.10193},
  year      = {2024}
}
```
