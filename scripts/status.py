import argparse,datetime,json,pathlib,subprocess
r=pathlib.Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--write',action='store_true');a=p.parse_args()
results=[]
for f in sorted((r/'runs').glob('*/episode_*/result.json')):
    x=json.loads(f.read_text());x['evidence']=str(f.relative_to(r));results.append(x)
def gate(ok,evidence=None):
    return {'status':'passed' if ok else 'not_verified','evidence':evidence or []}
idx=[json.loads(x) for x in (r/'benchmark_failure/index/selection.jsonl').read_text().splitlines()] if (r/'benchmark_failure/index/selection.jsonl').exists() else []
assets={'expected_scene_pairs':len(idx),'downloaded_scene_pairs':sum((r/x['bddl_file']).exists() and (r/x['sim_state_file']).exists() for x in idx),'downloaded_annotations':len(list((r/'benchmark_failure/annotations/scenes').glob('*/*.json'))),'checkpoint_hash_verified':(r/'models/pi05_libero/verified-hashes.json').exists()}
assets['complete']=bool(idx) and assets['downloaded_scene_pairs']==len(idx) and assets['downloaded_annotations']>=len(idx)
def select(pred):return [x['evidence'] for x in results if pred(x)]
inference=select(lambda x:x.get('policy_queries',0)>0)
rollout=select(lambda x:x.get('completed',False) and x.get('steps',0)>0)
control=select(lambda x:x.get('steps',0)>0 and x.get('policy_queries',0)>0)
task_success=select(lambda x:x.get('success',False) and not x.get('initial_native_success',True))
recovery=select(lambda x:x['mode']=='recovery' and x.get('steps',0)>0 and not x.get('initial_native_success',True) and not x.get('native_success_at_failure_restore',True))
recovery_success=select(lambda x:bool(x.get('recovery_success')))
doctor=json.loads((r/'logs/doctor.json').read_text()) if (r/'logs/doctor.json').exists() else {}
scene_file=r/'logs/scene-validation.json'
scenes=json.loads(scene_file.read_text()) if scene_file.exists() else []
gates={
 'environment':gate(bool(doctor),['logs/doctor.json']),
 'assets':gate(assets['complete'] and assets['checkpoint_hash_verified'],['benchmark_failure/index/selection.jsonl','models/pi05_libero/verified-hashes.json']),
 'scene_loading':gate(bool(scenes) and all(x['finite'] and x['state_length']==x['expected_state_length'] for x in scenes),['logs/scene-validation.json']),
 'physics':gate(doctor.get('finite_sim_state',False) and doctor.get('render_nonblank',False),['logs/doctor.json','logs/doctor.png']),
 'robot_control':gate(bool(control),control),
 'policy_inference':gate(bool(inference),inference),
 'policy_rollout':gate(bool(rollout),rollout),
 'task_success':gate(bool(task_success),task_success),
 'recovery_replay':gate(bool(scenes),['logs/scene-validation.json']),
 'recovery_execution':gate(bool(recovery),recovery),
 'recovery_success':gate(bool(recovery_success),recovery_success),
}
s={'benchmark':'LIBERO-Recovery','updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=r,text=True).strip(),'policy_backend':{'name':'official openpi pi05_libero','adapted_from_author_backend':True,'author_checkpoint':'not supplied by official source','source_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=r/'vendor/openpi',text=True).strip(),'checkpoint':'gs://openpi-assets/checkpoints/pi05_libero','object_versions':'deployment/pi05-libero-files.json','hashes':'models/pi05_libero/verified-hashes.json'},'scope':'Selected scenes and tasks only; not the full benchmark score','assets':assets,'gates':gates,'episodes':results}
if a.write:(r/'deployment-status.json').write_text(json.dumps(s,indent=2))
print(json.dumps(s,indent=2))
