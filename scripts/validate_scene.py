import json,pathlib,numpy as np
from rollout import make_env,reset,noisy_state,reanchor
r=pathlib.Path(__file__).resolve().parents[1];results=[]
for ep in ['019','020']:
 b=r/f'benchmark_failure/scenes/libero_goal/open_the_middle_drawer_of_the_cabinet/episode_{ep}/step_0080.bddl'
 env=make_env(b,7);reset(env,7);original=np.load(b.with_name(b.stem+'_sim_state.npy'));untouched=original.copy();perturbed,deltas=noisy_state(env,original,np.random.RandomState(7),.02)
 addresses=set()
 for v in deltas.values():addresses.update([v['flattened_address'],v['flattened_address']+1])
 changed=set(np.flatnonzero(original!=perturbed).tolist());assert changed==addresses;assert np.array_equal(original,untouched);assert original[0]==perturbed[0]
 obs=env.set_init_state(perturbed);reanchor(env)
 item={'episode':ep,'state_length':len(original),'expected_state_length':1+env.env.sim.model.nq+env.env.sim.model.nv,'changed_indices':sorted(changed),'object_deltas':deltas,'native_success_at_restore':bool(env.env._check_success()),'finite':bool(np.isfinite(env.env.sim.get_state().flatten()).all()),'time_preserved':float(original[0]),'original_untouched':bool(np.array_equal(original,untouched))};assert item['state_length']==item['expected_state_length'];assert item['finite'];results.append(item);env.close()
(r/'logs/scene-validation.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))
