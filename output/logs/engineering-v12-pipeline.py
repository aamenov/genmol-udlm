import datetime, json, os, pathlib, subprocess, sys
root = pathlib.Path(__file__).resolve().parents[2]
os.chdir(root)
started = datetime.datetime.now(datetime.timezone.utc).isoformat()
common = ['--protocol', 'experiments/udlm/protocols/engineering_v12_mask_prior.json', '--output-root', 'output/udlm/engineering_v12']
launch = subprocess.run([sys.executable, '-u', 'scripts/udlm/launch_exploration.py', *common, '--log-root', 'output/logs/engineering_v12'])
print(json.dumps({'event': 'generation_terminal', 'returncode': launch.returncode}), flush=True)
report_env = os.environ.copy()
report_env['CUDA_VISIBLE_DEVICES'] = ''
report = subprocess.run([sys.executable, '-u', 'scripts/udlm/report_exploration.py', *common, '--report-dir', 'output/udlm/engineering_v12_reports/complete'], env=report_env)
status = {'started_at_utc': started, 'finished_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'generation_returncode': launch.returncode, 'report_returncode': report.returncode}
with open('output/logs/engineering-v12-pipeline-status.json', 'x') as stream:
    json.dump(status, stream, indent=2)
    stream.write('\n')
print(json.dumps(status), flush=True)
sys.exit(1 if launch.returncode or report.returncode else 0)
