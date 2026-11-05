"""Official openpi policy creation, loopback-only websocket transport."""
import logging,pathlib,os,json,subprocess,importlib.metadata
import jax
import websockets.asyncio.server as ws_server
from openpi.policies import policy_config
from openpi.training import config
from openpi.serving import websocket_policy_server
logging.basicConfig(level=logging.INFO,force=True)
r=pathlib.Path(__file__).resolve().parents[1]
runtime={'jax':jax.__version__,'devices':[str(d) for d in jax.devices()],'torch':importlib.metadata.version('torch'),'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],text=True).strip()}
(r/'logs'/('policy-runtime_'+os.environ.get('BENCH_RUN_ID',str(os.getpid()))+'.json')).write_text(json.dumps(runtime,indent=2))
logging.info('Actual policy runtime: %s',runtime)
p=policy_config.create_trained_policy(config.get_config('pi05_libero'),r/'models/pi05_libero')
class RecordedPolicy:
    def __init__(self,policy):self.policy=policy;self.query_index=0
    def infer(self,obs):
        response=self.policy.infer(obs)
        response['deployment_query_index']=self.query_index
        self.query_index+=1
        return response
class LocalPolicyServer(websocket_policy_server.WebsocketPolicyServer):
    async def run(self):
        async with ws_server.serve(self._handler,self._host,self._port,compression=None,max_size=None,process_request=websocket_policy_server._health_check,ping_interval=None,ping_timeout=None) as server:
            await server.serve_forever()
LocalPolicyServer(policy=RecordedPolicy(p),host='127.0.0.1',port=18061,metadata=p.metadata).serve_forever()
