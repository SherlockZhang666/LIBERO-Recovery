import datetime,json,pathlib,subprocess,time,sys
r=pathlib.Path(__file__).resolve().parents[1]
pids=[int(sys.argv[1])] if len(sys.argv)>1 else []
for p in pathlib.Path('/proc').glob('[0-9]*/cmdline'):
 try:
  args=p.read_bytes().split(b'\0')
  if len(sys.argv)==1 and len(args)>1 and args[-2]==b'scripts/serve.py':pids.append(int(p.parent.name))
 except (OSError,PermissionError):pass
if len(pids)!=1:raise SystemExit(f'Expected one owned policyserver, found{pids}')
pid=pids[0];out=r/'logs'/f'policy-memory_{pid}.jsonl';print('Monitoring',pid,out,flush=True)
with out.open('a') as f:
 while pathlib.Path(f'/proc/{pid}').exists():
  record={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':pid}
  try:
   record['process_memory']={a:b.strip() for a,b in (x.split(':',1) for x in pathlib.Path(f'/proc/{pid}/status').read_text().splitlines() if ':' in x) if a in ['VmRSS','VmPeak','VmSwap']}
   record['host_memory']={a:b.strip() for a,b in (x.split(':',1) for x in pathlib.Path('/proc/meminfo').read_text().splitlines()) if a in ['MemAvailable','SwapFree']}
   record['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.free,utilization.gpu','--format=csv,noheader'],text=True).strip()
  except Exception as e:record['error']=str(e)
  f.write(json.dumps(record)+'\n');f.flush();time.sleep(5)
