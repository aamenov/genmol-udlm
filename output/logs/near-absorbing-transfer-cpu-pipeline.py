from pathlib import Path
from datetime import datetime, timezone
import hashlib,json,os,subprocess,sys,time
root=Path(__file__).resolve().parents[2]
output=root/'output/udlm/near_absorbing_transfer_cpu_20260907'
output.mkdir(parents=True,exist_ok=False)
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
def guard():
 assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()==head
 assert subprocess.check_output(['git','rev-parse','@{upstream}'],cwd=root,text=True).strip()==head
 assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=root,text=True).strip()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):
 with p.open('x') as f:json.dump(d,f,sort_keys=True,indent=2);f.write('\n')
command=[sys.executable,'-u','scripts/udlm/audit_near_absorbing_transfer.py']
record={'started_at':datetime.now(timezone.utc).isoformat(),'source':head,'command':command,'device':'cpu','wrapper_sha256':sha(Path(__file__)),'script_sha256':sha(root/command[-1]),'design_sha256':sha(root/'experiments/udlm/designs/near_absorbing_transfer_cpu.md'),'gpu_jobs':0,'optimization_calls':0,'forward_calls_planned':12}
guard();save(output/'request.json',record)
env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','PYTHONHASHSEED':'0','HF_HUB_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false','PYTHONDONTWRITEBYTECODE':'1'}
start=time.monotonic()
with (output/'result.json').open('x') as stdout,(output/'stderr.log').open('x') as stderr:
 run=subprocess.run(command,cwd=root,env=env,stdout=stdout,stderr=stderr)
record.update(return_code=run.returncode,finished_at=datetime.now(timezone.utc).isoformat(),wall_seconds=time.monotonic()-start,result_sha256=sha(output/'result.json'),stderr_sha256=sha(output/'stderr.log'))
try:
 guard()
 if run.returncode==0:
  result=json.loads((output/'result.json').read_text());assert len(result['results'])==12 and result['device']=='cpu'
  record['status']='completed'
 else:record['status']='failed'
except Exception as e:record.update(status='failed',error=f'{type(e).__name__}: {e}')
save(output/'terminal.json',record)
print(json.dumps(record),flush=True)
sys.exit(0 if record['status']=='completed' else 1)
