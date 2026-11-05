import json,pathlib,platform,subprocess,time
import imageio.v2 as imageio
import numpy as np
import mujoco
from libero.libero.envs import OffScreenRenderEnv
r=pathlib.Path(__file__).resolve().parents[1]
b=r/'LIBERO/libero/libero/bddl_files/libero_goal/open_the_middle_drawer_of_the_cabinet.bddl'
e=OffScreenRenderEnv(bddl_file_name=str(b),camera_heights=256,camera_widths=256);o=e.reset();before=e.env.sim.get_state().flatten()
for _ in range(10):o,_,_,_=e.step([0]*6+[-1])
f=o['agentview_image'][::-1,::-1];imageio.imwrite(r/'logs/doctor.png',f)
d={'python':platform.python_version(),'mujoco':mujoco.__version__,'render_shape':list(f.shape),'render_std':float(f.std()),'render_nonblank':bool(f.std()>10),'finite_sim_state':bool(np.isfinite(e.env.sim.get_state().flatten()).all()),'native_success':bool(e.env._check_success()),'state_size':len(before),'nq':e.env.sim.model.nq,'nv':e.env.sim.model.nv,'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],text=True).strip()};e.close();assert d['render_nonblank'] and d['finite_sim_state'];(r/'logs/doctor.json').write_text(json.dumps(d,indent=2));print(json.dumps(d,indent=2))
