"""Real public openpi policy rollouts; reconstruction only before control starts."""
import argparse, collections, hashlib, json, pathlib, re, time, traceback
import imageio.v2 as imageio
import numpy as np
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools, websocket_client_policy, msgpack_numpy
import websockets.sync.client
from robosuite.utils.errors import RandomizationError
ROOT=pathlib.Path(__file__).resolve().parents[1]
class CompileTolerantClient(websocket_client_policy.WebsocketClientPolicy):
    # First JAX compilation can exceed the websockets15 default20s ping timeout.
    # Inference payloads and model responses remain the official openpi protocol.
    def _wait_for_server(self):
        conn=websockets.sync.client.connect(self._uri,compression=None,max_size=None,ping_interval=None,ping_timeout=None)
        return conn,msgpack_numpy.unpackb(conn.recv())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def quat_axis(q):
    q=np.asarray(q).copy();q[3]=np.clip(q[3],-1,1);d=np.sqrt(1-q[3]**2)
    return np.zeros(3) if d<1e-7 else q[:3]*2*np.arccos(q[3])/d
def state(obs):return np.concatenate([obs['robot0_eef_pos'],quat_axis(obs['robot0_eef_quat']),obs['robot0_gripper_qpos']]).astype(np.float32)
def reanchor(env):
    for r in env.robots:r.controller.update();r.controller.update_initial_joints(r.controller.joint_pos);r.controller.reset_goal()
def make_env(bddl,seed):
    for attempt in range(12):
        np.random.seed(seed+attempt)
        try:
            env=OffScreenRenderEnv(bddl_file_name=str(bddl),camera_heights=256,camera_widths=256)
            env.seed(seed);return env
        except RandomizationError:continue
    raise RuntimeError('Placement failed after 12 initialization attempts')
def reset(env,seed):
    for attempt in range(12):
        np.random.seed(seed+attempt)
        try:return env.reset()
        except RandomizationError:continue
    raise RuntimeError('Placement failed after 12 reset attempts')
def noisy_state(env,original,rng,noise):
    # MuJoCo get_state().flatten(): time, qpos, qvel, act. qpos starts at1.
    sim=env.env.sim; nq=sim.model.nq;nv=sim.model.nv
    assert original.size>=1+nq+nv,(original.size,nq,nv)
    copy=np.asarray(original,dtype=np.float64).copy();deltas={}
    for name,bodyid in env.env.obj_body_id.items():
        jid=int(sim.model.body_jntadr[bodyid])
        if jid<0 or int(sim.model.jnt_type[jid])!=0:continue
        qadr=int(sim.model.jnt_qposadr[jid]);xy=rng.uniform(-noise,noise,size=2)
        copy[1+qadr:1+qadr+2]+=xy
        deltas[name]={'dx':float(xy[0]),'dy':float(xy[1]),'qpos_address':qadr,'flattened_address':qadr+1}
    return copy,deltas

def run(args):
    out=ROOT/'runs'/args.run_id;out.mkdir(parents=True,exist_ok=False)
    (out/'config.json').write_text(json.dumps(vars(args),indent=2));rng=np.random.RandomState(args.seed)
    import shutil,subprocess
    provenance={'benchmark_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'policy_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT/'vendor/openpi',text=True).strip(),'checkpoint':'gs://openpi-assets/checkpoints/pi05_libero','dataset':'ataier/LIBERO_Recovery_FailureScenes','dataset_revision':'95acf98dd41a4965018069e7e4f27aac1fc69328'}
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2))
    for manifest in ['source-files.json','verified-hashes.json']:
        if (ROOT/'models/pi05_libero'/manifest).exists():shutil.copyfile(ROOT/'models/pi05_libero'/manifest,out/manifest)
    client=CompileTolerantClient('127.0.0.1',args.port)
    if args.mode=='policy':
        native_root=ROOT/'LIBERO/libero/libero/bddl_files'
        bddls=sorted(native_root.glob(('*' if args.suite=='any' else args.suite)+'/'+('*.bddl' if args.task=='any' else args.task+'.bddl')))
        trials=[('native',False)]*args.trials
    else:
        index=[json.loads(x) for x in (ROOT/'benchmark_failure/index/selection.jsonl').read_text().splitlines()]
        selected=[]
        for x in index:
            if args.suite!='any' and x['source']!=args.suite:continue
            if args.task!='any' and args.task.lower() not in x['task'].lower().replace(' ','_'):continue
            ann=ROOT/'benchmark_failure/annotations/scenes'/x['source']/(x['scene_id']+'.json')
            annotation=json.loads(ann.read_text()) if ann.exists() else {}
            level=x.get('level',annotation.get('level'))
            if args.level!='any' and level!=args.level:continue
            b=ROOT/x['bddl_file']
            if b.exists() and (ROOT/x['sim_state_file']).exists():selected.append((x,annotation))
        if args.scene_id:selected=[x for x in selected if x[0]['scene_id']==args.scene_id]
        selected=sorted(selected,key=lambda x:x[0]['scene_id'])
        if args.scenes>0 and len(selected)>args.scenes:
            chosen=sorted(np.random.RandomState(args.seed).choice(len(selected),args.scenes,replace=False))
            selected=[selected[i] for i in chosen]
        if not selected:raise RuntimeError('No downloaded matching scenes; run scripts/bench list')
        (out/'selected-scenes.json').write_text(json.dumps(selected,indent=2))
        bddls=[ROOT/x[0]['bddl_file'] for x in selected]
        trials=[('close',True)]*args.close_trials+[('open',False)]*args.open_trials
    results=[]
    for si,bddl in enumerate(bddls):
        bddl_prompt=re.search(r'\(:language\s+(.+?)\s*\)',bddl.read_text()).group(1).strip()
        prompt=bddl.stem.replace('_',' ') if args.mode=='policy' else bddl_prompt
        env=make_env(bddl,args.seed+1000001*(si+1))
        for ti,(grip,closed) in enumerate(trials):
            ep=out/f'episode_{si:03d}_{ti:03d}_{grip}';ep.mkdir()
            frames=[];wrist=[];actions=[];states=[];simstates=[];inferences=[]
            result={'mode':args.mode,'scene':str(bddl.relative_to(ROOT)),'scene_sha256':sha(bddl),'prompt':prompt,'bddl_language':bddl_prompt,'seed':args.seed,'trial':ti,'gripper_init':grip,'policy':'official openpi pi05_libero','policy_rng_seed':0,'recovery_type':'learned_policy_continuation' if args.mode=='recovery' else None,'success':False,'completed':False,'exception':None}
            try:
                obs=reset(env,args.seed+1000001*(si+1)+10001*(ti+1))
                obj_deltas={}
                if args.mode=='recovery':
                    sfile=bddl.with_name(bddl.stem+'_sim_state.npy');original=np.load(sfile)
                    init,obj_deltas=noisy_state(env,original,rng,args.noise)
                    np.save(ep/'original_failure_state.npy',original);np.save(ep/'initial_perturbed_state.npy',init)
                    obs=env.set_init_state(init);reanchor(env);result['native_success_at_failure_restore']=bool(env.env._check_success())
                    result.update({'source_scene_id':selected[si][0]['scene_id'],'annotation':selected[si][1],'state_sha256':sha(sfile),'initialization':'recorded failure state copied; free-joint xy perturbation applied before rollout; controller goal reanchored','obj_deltas':obj_deltas,'flat_qpos_offset':1})
                    # Physical actuator control, with fresh observation after the final settle step.
                    for _ in range(args.settle_steps):obs,_,_,_=env.step([0.]*6+[1. if closed else -1.])
                else:
                    for _ in range(10):obs,_,_,_=env.step([0.]*6+[-1.])
                result['initial_native_success']=bool(env.env._check_success())
                np.save(ep/'control_start_state.npy',env.env.sim.get_state().flatten())
                if args.mode=='recovery' and result['initial_native_success']:
                    result['invalid_recovery']='scene already satisfies native predicate before policy'
                queue=collections.deque();start=time.monotonic();baseline_force=None;first_contact_step=None;first_gripper_change_step=None;prev_gripper=None
                for step in range(args.max_steps):
                    img=np.ascontiguousarray(obs['agentview_image'][::-1,::-1]);wr=np.ascontiguousarray(obs['robot0_eye_in_hand_image'][::-1,::-1])
                    frames.append(img);wrist.append(wr);s=state(obs);states.append(s);simstates.append(env.env.sim.get_state().flatten())
                    if not queue:
                        inp={'observation/image':image_tools.convert_to_uint8(image_tools.resize_with_pad(img,224,224)),'observation/wrist_image':image_tools.convert_to_uint8(image_tools.resize_with_pad(wr,224,224)),'observation/state':s,'prompt':prompt}
                        if not inferences:
                            np.savez_compressed(ep/'first_policy_input.npz',image=inp['observation/image'],wrist_image=inp['observation/wrist_image'],state=s,prompt=prompt)
                            result['input_shapes']={k:list(np.shape(v)) for k,v in inp.items()}
                        t=time.monotonic();response=client.infer(inp);chunk=np.asarray(response['actions']);latency=time.monotonic()-t
                        assert chunk.ndim==2 and chunk.shape[1]==7 and len(chunk)>=args.replan_steps,chunk.shape
                        np.save(ep/f'chunk_{len(inferences):04d}.npy',chunk)
                        inferences.append({'step':step,'seconds':latency,'shape':list(chunk.shape),'server_query_index':response.get('deployment_query_index')});queue.extend(chunk[:args.replan_steps])
                    action=np.asarray(queue.popleft(),dtype=np.float32)
                    if not np.all(np.isfinite(action)):raise ValueError('non-finite learned action')
                    # pi05 LIBERO transform already returns native OSC delta and native gripper sign.
                    obs,reward,done,info=env.step(action.tolist());actions.append(action)
                    force=env.env.robots[0].ee_force.copy()
                    if baseline_force is None:baseline_force=force
                    elif first_contact_step is None and np.linalg.norm(force-baseline_force)>2.:first_contact_step=step
                    current_gripper=int(action[-1]>0)
                    if prev_gripper is not None and first_gripper_change_step is None and current_gripper!=prev_gripper:first_gripper_change_step=step
                    prev_gripper=current_gripper
                    if step%20==0:print(json.dumps({'episode':str(ep),'step':step,'done':bool(done),'action':action.tolist()}),flush=True)
                    if done:result['success']=True;break
                np.save(ep/'control_final_state.npy',env.env.sim.get_state().flatten())
                imageio.imwrite(ep/'final_after_action.png',np.ascontiguousarray(obs['agentview_image'][::-1,::-1]))
                result.update(first_contact_step=first_contact_step,first_gripper_change_step=first_gripper_change_step,completed=True,steps=len(actions),wall_seconds=time.monotonic()-start,native_predicate=bool(env.env._check_success()),policy_queries=len(inferences))
                result['environment_done']=bool(result['success'])
                result['success']=bool(result['environment_done'] and result['native_predicate'])
                result['recovery_success']=bool(result['success'] and result['native_predicate'] and not result['initial_native_success'] and not result['native_success_at_failure_restore']) if args.mode=='recovery' else None
            except Exception:
                result['exception']=traceback.format_exc();print(result['exception'],flush=True)
            finally:
                result['policy_queries']=len(inferences);result['steps']=len(actions)
                if frames:
                    imageio.mimwrite(ep/'agentview.mp4',frames,fps=20);imageio.mimwrite(ep/'wrist.mp4',wrist,fps=20)
                    imageio.imwrite(ep/'initial.png',frames[0]);imageio.imwrite(ep/'final.png',frames[-1])
                    np.savez_compressed(ep/'trajectory.npz',observations=np.stack(states),actions=np.asarray(actions),sim_states=np.stack(simstates),agentview=np.stack(frames),wrist=np.stack(wrist))
                (ep/'inferences.json').write_text(json.dumps(inferences,indent=2));(ep/'result.json').write_text(json.dumps(result,indent=2));results.append(result)
                (out/'results.json').write_text(json.dumps(results,indent=2));print('RESULT',json.dumps(result),flush=True)
        env.close()
    summary={'episodes':len(results),'complete':sum(x['completed'] for x in results),'successes':sum(x['success'] for x in results),'recovery_successes':sum(bool(x.get('recovery_success')) for x in results),'open':{'episodes':sum(x['gripper_init']=='open' for x in results),'successes':sum(x['success'] and x['gripper_init']=='open' for x in results)},'close':{'episodes':sum(x['gripper_init']=='close' for x in results),'successes':sum(x['success'] and x['gripper_init']=='close' for x in results)},'scope':'selected smoke evaluation; not a full benchmark score'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print('SUMMARY',json.dumps(summary),flush=True)
    if summary['complete']!=summary['episodes']:raise SystemExit(1)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['policy','recovery'],required=True);p.add_argument('--run-id',required=True);p.add_argument('--port',type=int,default=18061);p.add_argument('--suite',default='libero_goal');p.add_argument('--task',default='open_the_middle_drawer_of_the_cabinet');p.add_argument('--trials',type=int,default=1);p.add_argument('--scenes',type=int,default=1);p.add_argument('--scene-id',default='');p.add_argument('--level',default='L1');p.add_argument('--close-trials',type=int,default=1);p.add_argument('--open-trials',type=int,default=1);p.add_argument('--noise',type=float,default=.02);p.add_argument('--settle-steps',type=int,default=20);p.add_argument('--max-steps',type=int,default=520);p.add_argument('--replan-steps',type=int,default=5);p.add_argument('--seed',type=int,default=7);run(p.parse_args())
