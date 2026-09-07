"""One manually invoked post-exit audit; preserve V8 failure and exact stale leases."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict

ROOT = Path('/home/aidar.alimbayev/Documents/genmolv2/run_sources/udlm_genmol_worktree')
for item in (ROOT, ROOT / 'src'):
    sys.path.insert(0, str(item))
from scripts import artifact_io
from scripts.udlm import launch_engineering_training as engine
from scripts.exps.denovo import benchmark

base = 'output/udlm/engineering_v8/ct_ce_e_1000_b128_w2'
terminal_path = f'{base}/ct/terminal_manifest.json'
campaign_path = f'{base}/campaign_terminal.json'
terminal_claim, payload = artifact_io.snapshot_file(ROOT, terminal_path, capture_bytes=True)
terminal = json.loads(payload)
campaign_claim, campaign_bytes = artifact_io.snapshot_file(ROOT, campaign_path, capture_bytes=True)
campaign = json.loads(campaign_bytes)
assert terminal['status'] == campaign['status'] == 'failed'
assert terminal['training_return_code'] == 0
assert terminal['error'] == 'RuntimeError: child exited but its process group remains; leases retained'
assert terminal['completed_example_exposures'] is None
assert not (ROOT / base / 'ce').exists()
assert engine.benchmark._require_clean_pushed_source() == terminal['source']
claims = [artifact_io.FileClaim(**record) for record in terminal['leases']]
engine.check_leases(ROOT, claims)
lease_bytes = [(claim, artifact_io.snapshot_file(ROOT, claim.relative_path, capture_bytes=True)[1]) for claim in claims]
owners = {json.loads(data)['pid'] for _, data in lease_bytes}

def absence_probe():
    for pid in owners:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            pass
        else:
            raise RuntimeError(f'lease owner {pid} still exists')
    if engine.process_group_exists(terminal['training_pid']):
        raise RuntimeError('training process group still exists')
    return {'at': engine.stamp(), 'lease_owner_pids_absent': sorted(owners), 'training_process_group_absent': terminal['training_pid']}

before = absence_probe()
checkpoint_relative = f'{base}/ct/checkpoints/1000.ckpt'
expected_hash = terminal['artifact_hashes'][checkpoint_relative]
checkpoint = engine.validate_checkpoint_output(ROOT / checkpoint_relative, terminal['plan']['config'], expected_steps=1000)
assert checkpoint['sha256'] == expected_hash
metadata = benchmark.checkpoint_metadata(ROOT / checkpoint_relative, expected_sha256=expected_hash)
assert metadata['global_step'] == 1000 and metadata['diffusion_type'] == 'udlm'
assert 'udlm_denoiser_metadata' not in metadata
assert terminal['plan']['config']['training']['udlm']['parameterization'] == 'raw_loo'
after = absence_probe()
assert artifact_io.snapshot_file(ROOT, terminal_path)[0] == terminal_claim
assert artifact_io.snapshot_file(ROOT, campaign_path)[0] == campaign_claim
engine.check_leases(ROOT, claims)
assert engine.benchmark._require_clean_pushed_source() == terminal['source']
process_listing = subprocess.check_output(['ps', '-eo', 'pid,ppid,pgid,stat,etimes,args'], text=True)
gpu_processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory', '--format=csv,noheader,nounits'], text=True)
for line in gpu_processes.splitlines():
    fields = [value.strip() for value in line.split(',')]
    assert int(fields[1]) not in owners | {terminal['training_pid'], 407026}

output = f'{base}/post_exit_audit'
artifact_io.create_directory_exclusive(ROOT, output)
inputs = {}
for name, content in [('original_terminal.json', payload), ('original_campaign_terminal.json', campaign_bytes), ('process_snapshot.txt', process_listing.encode()), ('gpu_processes.csv', gpu_processes.encode()), *[(f'original_lease_{i}.json', data) for i, (_, data) in enumerate(lease_bytes)]]:
    inputs[name] = asdict(artifact_io.publish_bytes_exclusive(ROOT, f'{output}/{name}', content))
training_sources = [ROOT / 'scripts/train.py', *sorted((ROOT / 'src').rglob('*.py'))]
source_hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in training_sources}
receipt = {
    'schema_version': 1,
    'kind': 'manual_post_exit_checkpoint_audit_and_owned_stale_lease_release',
    'source': terminal['source'],
    'original_campaign_status': 'failed',
    'original_terminal_sha256': terminal_claim.sha256,
    'original_campaign_terminal_sha256': campaign_claim.sha256,
    'original_training_return_code': 0,
    'original_training_subprocess_seconds': terminal['training_subprocess_seconds'],
    'checkpoint_status': 'separately_validated_after_controller_failure',
    'checkpoint': checkpoint,
    'checkpoint_metadata': metadata,
    'configured_example_exposures_supported_by_step_and_batch': 128000,
    'exposure_caveat': 'Configured example exposures, not a distinct-molecule census or independent live row trace; original completion fields remain null.',
    'training_implementation_sha256': source_hashes,
    'probes': [before, after],
    'archived_inputs': inputs,
    'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'interpretation': 'Observed transient process-group teardown after successful training, consistent with an immediate-check race. Exact transient members were not captured. This separate audit does not change original failure status or start CE.',
    'manual_action': 'Release only the two exact retained leases after confirming their controller and training process group are absent; do not signal any process.',
}
absence_probe()
engine.release_leases(ROOT, claims)
assert all(not (ROOT / claim.relative_path).exists() for claim in claims)
receipt['leases_released'] = True
receipt['finished_at'] = engine.stamp()
published = artifact_io.publish_bytes_exclusive(ROOT, f'{output}/post_exit_audit.json', engine.encode(receipt))
print(json.dumps({'audit': asdict(published), 'checkpoint_sha256': expected_hash, 'leases_released': True}, indent=2))
