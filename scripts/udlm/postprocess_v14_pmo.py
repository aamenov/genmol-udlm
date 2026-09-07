"""One-shot CPU-only V14 acceptance after the frozen controller has exited.

Run in a named tmux session. Never resumes, retries an oracle, or changes the
optimization protocol. Scientific acceptance comes from the saved report status.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

WORKSPACE = Path('/home/aidar.alimbayev/Documents/genmolv2')
SOURCES = WORKSPACE / 'run_sources'
ROOT = SOURCES / 'udlm_genmol_worktree'
PYTHON = WORKSPACE / '.venv/bin/python'
PANEL = 'experiments/udlm/protocols/engineering_v14_pmo.json'
PANEL_SHA = '8f42ce172b392d27260317f3db7e07ce5c5cbe528eb448234388fcc34e0f4fda'
CAMPAIGN = ROOT / 'output/udlm/engineering_v14_pmo'
OUTPUT = ROOT / 'output/udlm/engineering_v14_pmo_postprocessing_20260907'
PINS = {
    'udlm_genmol_worktree': '48473d4febbd06d9bc07986ca96ceb93926c91ec',
    'udlm_pmo_rescore_worktree': '3e2796e0e2d890cb33a2e29c58efec46575290af',
    'udlm_pmo_report_worktree': 'f7548f04be55384b0cf47157e63ea0395279f5cd',
}


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(directory, *args):
    return subprocess.check_output(['git', *args], cwd=directory, text=True).strip()


def save(path, record):
    with Path(path).open('x') as stream:
        json.dump(record, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def check_sources(pins):
    for name, expected in pins.items():
        directory = SOURCES / name
        if (git(directory, 'rev-parse', 'HEAD') != expected
                or git(directory, 'rev-parse', '@{upstream}') != expected
                or git(directory, 'status', '--porcelain', '--untracked-files=no')):
            raise ValueError(f'Source changed, dirty, or not pushed: {name}')
    if sha(ROOT / PANEL) != PANEL_SHA:
        raise ValueError('Fixed V14 panel changed')


def execute(label, worktree, arguments, record, pins):
    check_sources(pins)
    if sha(CAMPAIGN / 'terminal_manifest.json') != record['controller_terminal_sha256']:
        raise ValueError('Controller terminal changed before postprocessing step')
    command = [str(PYTHON), '-u', *map(str, arguments)]
    overrides = dict(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                     OPENBLAS_NUM_THREADS='1', PYTHONHASHSEED='0', TOKENIZERS_PARALLELISM='false',
                     PYTHONPATH=f'{worktree}/src:{worktree}', TMPDIR=str(ROOT / 'output/tmp'))
    step = {'label': label, 'command': command, 'cwd': str(worktree), 'started_at': now(),
            'environment_overrides': overrides, 'timeout_seconds': 3600}
    record['steps'].append(step)
    save(OUTPUT / f'{label}.intent.json', step)
    env = {**os.environ, **overrides}
    with (OUTPUT / f'{label}.log').open('x') as stream:
        result = subprocess.run(command, cwd=worktree, env=env, stdout=stream,
                                stderr=subprocess.STDOUT, timeout=3600)
    step.update(return_code=result.returncode, finished_at=now())
    save(OUTPUT / f'{label}.exit.json', step)
    check_sources(pins)
    if sha(CAMPAIGN / 'terminal_manifest.json') != record['controller_terminal_sha256']:
        raise ValueError('Controller terminal changed during postprocessing step')
    print(json.dumps(step), flush=True)
    return result.returncode


def main():
    if Path(sys.executable).resolve() != PYTHON.resolve():
        raise ValueError('Use the project virtual environment')
    own = Path(__file__).resolve().parents[2]
    pins = {**PINS, own.name: git(own, 'rev-parse', 'HEAD')}
    check_sources(pins)
    OUTPUT.mkdir(parents=True, exist_ok=False)
    record = {'schema_version': 1, 'started_at': now(), 'source_commits': pins,
              'wrapper_sha256': sha(__file__), 'panel_sha256': PANEL_SHA, 'steps': [],
              'oracle_scope': 'fresh postoptimization verification only; outside online budget',
              'retry_policy': 'none; existing namespace is an error', 'status': 'waiting'}
    save(OUTPUT / 'request.json', record)
    code = 1
    try:
        pipeline_path = ROOT / 'output/logs/engineering-v14-pmo-pipeline-status.json'
        terminal_path = CAMPAIGN / 'terminal_manifest.json'
        deadline = time.monotonic() + 12 * 3600
        while not (pipeline_path.exists() and terminal_path.exists()):
            if time.monotonic() >= deadline:
                raise TimeoutError('Controller did not finish within the 12-hour watcher limit')
            time.sleep(15)
        # The pipeline completion is written only after subprocess.run has returned.
        time.sleep(1)
        pipeline = json.loads(pipeline_path.read_bytes())
        terminal = json.loads(terminal_path.read_bytes())
        if pipeline['source'] != PINS['udlm_genmol_worktree']:
            raise ValueError('Wrong generation pipeline source')
        if os.path.lexists(ROOT / 'output/.single_generation_job.lock'):
            raise ValueError('Generation lease remains after controller process exit')
        check_sources(pins)
        terminal_sha = sha(terminal_path)
        record.update(controller_status=terminal['status'], controller_terminal_sha256=terminal_sha,
                      pipeline_status_sha256=sha(pipeline_path), controller_return_code=pipeline['return_code'])
        panel = json.loads((ROOT / PANEL).read_bytes())
        receipts = []
        accepted_controller = (pipeline['return_code'] == 0 and terminal['status'] == 'completed'
                               and terminal.get('final_input_validation') == 'unchanged'
                               and terminal.get('lease_release_authorized') is True)
        if accepted_controller:
            for entry in panel['entries']:
                identifier = entry['id']
                receipt = OUTPUT / 'verification' / f'{identifier}.json'
                directory = CAMPAIGN / 'runs' / identifier / 'fexofenadine_mpo/released' / f"seed_{entry['seed']}"
                rc = execute(identifier, SOURCES / 'udlm_pmo_rescore_worktree',
                             ['scripts/udlm/rescore_pmo_run.py', '--input-root', ROOT,
                              '--protocol', PANEL, '--entry-id', identifier,
                              '--run-directory', directory, '--output', receipt], record, pins)
                saved = json.loads(receipt.read_bytes())
                if saved['run']['entry_id'] != identifier or (rc == 0) != (saved['status'] == 'verified'):
                    raise ValueError('Verifier exit/receipt identity differs')
                claim = saved['inputs'].get('controller_terminal')
                if claim is not None and claim['sha256'] != terminal_sha:
                    raise ValueError('Verifier used a different controller terminal')
                # Failed receipts remain evidence; all six predeclared checks run once.
                receipts.extend(['--verification', identifier, receipt, sha(receipt)])
        else:
            record['verification_skipped'] = 'Entire campaign was not accepted; no verification oracle calls'
        report_directory = OUTPUT / 'report'
        rc = execute('report', SOURCES / 'udlm_pmo_report_worktree',
                     ['scripts/udlm/report_pmo_pilot.py', '--input-root', ROOT, '--panel', PANEL,
                      '--panel-sha256', PANEL_SHA, '--terminal-sha256', terminal_sha,
                      *receipts, '--output', report_directory], record, pins)
        if rc:
            raise RuntimeError(f'Report failed with return code {rc}; retain all original evidence')
        report = json.loads((report_directory / 'report.json').read_bytes())
        record['scientific_report_status'] = report['status']
        record['outside_budget_verification_calls'] = report['outside_budget_verification_calls']
        rc = execute('combine', own,
                     ['scripts/udlm/combine_pmo_study_report.py', '--base-directory',
                      ROOT / 'output/udlm/study_overview_v13_20260907',
                      '--pmo-directory', report_directory, '--pmo-manifest-sha256',
                      sha(report_directory / 'manifest.json'), '--output', OUTPUT / 'combined'], record, pins)
        if rc:
            raise RuntimeError(f'Combined PDF failed with return code {rc}')
        record['status'] = 'published_locally'
        code = 0 if report['status'] == 'verified_complete' else 2
    except Exception as error:
        record.update(status='failed', error=f'{type(error).__name__}: {error}')
    finally:
        record.update(finished_at=now(), return_code=code)
        save(OUTPUT / 'terminal.json', record)
        print(json.dumps(record), flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
