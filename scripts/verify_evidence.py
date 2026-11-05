"""CPU-only consistency audit of actual saved policy/recovery trajectories."""
import argparse,hashlib,json,pathlib
import numpy as np
r=pathlib.Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--run',default='20261105T095735Z_acceptance_*');args=p.parse_args();rows=[]
for run in sorted((r/'runs').glob(args.run)):
 for f in sorted(run.glob('episode_*/result.json')):
  x=json.loads(f.read_text());ep=f.parent
  if not x['completed']:
   rows.append({'episode':str(ep.relative_to(r)),'completed':False,'exception':x['exception']});continue
  a=np.load(ep/'trajectory.npz');n=x['steps'];acts=a['actions'];states=a['observations'];sim=a['sim_states'];end=np.load(ep/'control_final_state.npy');start=np.load(ep/'control_start_state.npy')
  assert acts.shape==(n,7) and states.shape==(n,8) and sim.shape[0]==n
  assert np.isfinite(acts).all() and np.isfinite(states).all() and np.isfinite(sim).all() and np.isfinite(end).all()
  assert np.max(np.abs(acts))>0 and np.max(np.std(states,axis=0))>0
  assert a['agentview'].shape==(n,256,256,3) and a['wrist'].shape==(n,256,256,3)
  assert a['agentview'].std()>10 and a['wrist'].std()>10
  assert np.allclose(np.diff(sim[:,0]),.05,atol=1e-8) and np.isclose(end[0]-start[0],n*.05)
  config=json.loads((run/'config.json').read_text());replan=config['replan_steps'];chunks=sorted(ep.glob('chunk_*.npy'));assert len(chunks)==x['policy_queries']
  for i,chunk in enumerate(chunks):
   c=np.load(chunk);assert c.shape==(10,7) and np.isfinite(c).all();length=min(replan,n-i*replan);assert np.allclose(acts[i*replan:i*replan+length],c[:length],rtol=1e-6,atol=1e-6)
  assert (ep/'agentview.mp4').stat().st_size>0 and (ep/'wrist.mp4').stat().st_size>0
  if x['mode']=='recovery':
   source=r/x['scene'];statefile=source.with_name(source.stem+'_sim_state.npy');assert hashlib.sha256(statefile.read_bytes()).hexdigest()==x['state_sha256'];original=np.load(ep/'original_failure_state.npy');assert np.array_equal(original,np.load(statefile))
   init=np.load(ep/'initial_perturbed_state.npy');expected=original.copy()
   for v in x['obj_deltas'].values():
    k=v['flattened_address'];expected[k]+=v['dx'];expected[k+1]+=v['dy']
   assert np.array_equal(init,expected)
   assert not x['native_success_at_failure_restore'] and not x['initial_native_success']
  inf=json.loads((ep/'inferences.json').read_text())
  rows.append({'episode':str(ep.relative_to(r)),'completed':True,'steps':n,'success':x['success'],'recovery_success':x.get('recovery_success'),'finite':True,'nonzero_learned_actions':True,'learned_chunks_match_applied_actions':True,'simulated_seconds':float(end[0]-start[0]),'physics_step_seconds':.05,'first_inference_seconds':inf[0]['seconds'],'query_index_range':[inf[0]['server_query_index'],inf[-1]['server_query_index']],'agentview_std':float(a['agentview'].std()),'wrist_std':float(a['wrist'].std())})
(r/'logs/evidence-audit.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
