import argparse,collections,json,pathlib
r=pathlib.Path(__file__).resolve().parents[1];p=argparse.ArgumentParser();p.add_argument('--suite',default='libero_goal');p.add_argument('--level',default='any');p.add_argument('--task',default='');a=p.parse_args()
for line in (r/'benchmark_failure/index/selection.jsonl').read_text().splitlines():
 x=json.loads(line)
 if a.suite!='any' and x['source']!=a.suite:continue
 if a.task not in x['task'].replace(' ','_'):continue
 ann=r/'benchmark_failure/annotations/scenes'/x['source']/(x['scene_id']+'.json');d=json.loads(ann.read_text()) if ann.exists() else {};level=x.get('level',d.get('level','unavailable'))
 if a.level!='any' and level!=a.level:continue
 print(json.dumps({'scene_id':x['scene_id'],'task':x['task'],'level':level,'stage':d.get('stage'),'downloaded':(r/x['bddl_file']).exists() and (r/x['sim_state_file']).exists(),'bddl':x['bddl_file']}))
