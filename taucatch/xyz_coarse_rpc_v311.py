"""Distinct condition-only RPC and fresh fast RPC with actual clock proof."""
import io
import json
import time
from multiprocessing.connection import Client
import numpy as np

from .action_link import AUTHKEY
from .xyz_candidate_rpc_v163 import no_delay
from .xyz_coarse_contract_v308 import XYZ_CONTRACT
from .xyz_coarse_policy_v310 import KIND, BEHAVIOR, NAMESPACE
from .xyz_context_softsign_v273 import CONTEXT_CONTRACT
from .xyz_coarse_contract_v308 import XYZ_CONTRACT
from .xyz_coarse_policy_v310 import OBSERVATION


def pack(sample, mode, op, episode):
    b = io.BytesIO()
    np.savez(b, **sample, mode=mode, op=op, episode=episode)
    return b.getvalue()


def unpack(payload):
    with np.load(io.BytesIO(payload), allow_pickle=False) as z:
        mode, op, episode = str(z['mode']), str(z['op']), str(z['episode'])
        if mode not in ['mean','zero','sample'] or not episode:
            raise ValueError('Unsupported sampled prerelease mode')
        fields = ({'over','mosaic','capture_ns','observed_ns','exposure_start_ns','readout_ns'}
                  if op == 'condition' else
                  {'state','events','context','observed_t','release_ns','preparation_complete_ns'}
                  if op == 'fast' else set())
        if not fields or set(z.files) != fields | {'mode','op','episode'}:
            raise ValueError('Input whitelist mismatch')
        s = {k:z[k].copy() for k in fields}
    if op == 'condition':
        for k in ['over','mosaic']:
            if s[k].shape != (224,224,3) or s[k].dtype != np.uint8:
                raise ValueError('Invalid RGB')
        for k in fields - {'over','mosaic'}:
            s[k] = int(s[k])
        if not (0 < s['exposure_start_ns'] < s['capture_ns'] and
                s['capture_ns'] + s['readout_ns'] <= s['observed_ns'] and
                s['readout_ns'] == 3000000):
            raise ValueError('Invalid actual camera exposure/readout')
    else:
        for k,shape in [('state',(8,)),('events',(2,6,60,80)),('context',(22,))]:
            if s[k].shape != shape or not np.isfinite(s[k]).all():
                raise ValueError('Invalid fresh '+k)
        s['observed_t'] = float(s['observed_t'])
        for k in ['release_ns','preparation_complete_ns']:
            s[k] = int(s[k])
        if not np.isfinite(s['observed_t']) or s['observed_t'] < .020:
            raise ValueError('Invalid flight time')
    return s, mode, op, episode


class PrethrowClient:
    def __init__(self, address, base_sha, artifact_sha, mode):
        self.conn=Client(address,authkey=AUTHKEY);no_delay(self.conn)
        self.base_sha,self.artifact_sha,self.mode=base_sha,artifact_sha,mode

    def _receive(self):
        o=json.loads(self.conn.recv_bytes())
        if 'error' in o: raise RuntimeError(o['error'])
        if (o.get('kind')!=KIND or o.get('behavior_contract')!=BEHAVIOR or
            o.get('observation_contract')!=OBSERVATION or o.get('behavior_namespace')!=NAMESPACE or o.get('context_contract')!=CONTEXT_CONTRACT or
            o.get('base_checkpoint_sha256')!=self.base_sha or o.get('artifact_sha256')!=self.artifact_sha):
            raise ValueError('Wrong prerelease runtime identity')
        return o

    def control(self,op,text=None):
        start=time.perf_counter_ns()
        self.conn.send_bytes(json.dumps(dict(op=op,**({}if text is None else {'text':text}))).encode())
        o=self._receive();o['roundtrip_ms']=(time.perf_counter_ns()-start)/1e6
        return o

    def predict(self,sample,op,episode):
        start=time.perf_counter_ns();self.conn.send_bytes(pack(sample,self.mode,op,episode))
        o=self._receive()
        if o['op']!=op or o['episode']!=episode or o['mode']!=self.mode:
            raise ValueError('Reply mismatch')
        if op=='condition':
            if any(k in o for k in ['actions','mean','latent','log_prob']):
                raise ValueError('Condition preparation must not produce actions')
        else:
            mean,action=np.asarray(o['mean'],np.float32),np.asarray(o['actions'],np.float32)
            if mean.shape!=(12,3) or not np.isfinite(mean).all() or np.max(abs(mean))>1.00001:
                raise ValueError('Malformed XYZ mean')
            if self.mode=='sample':
                latent=np.asarray(o['latent'],np.float32)
                if (latent.shape!=(12,3) or not np.isfinite(latent).all() or
                    not np.array_equal(action,np.clip(latent,-1,1)) or
                    not np.isfinite(o['log_prob']) or type(o['behavior_index'])is not int or o['behavior_index']<0):
                    raise ValueError('Malformed sampled latent/action/density')
            else:
                if not np.array_equal(action,mean if self.mode=='mean' else np.zeros((12,3),np.float32)):
                    raise ValueError('Wrong deterministic XYZ action')
                if o['latent']is not None or o['log_prob']is not None or o['behavior_index']is not None:
                    raise ValueError('Deterministic request consumed sampling identity')
        end=time.perf_counter_ns();o.update(roundtrip_ms=(end-start)/1e6,client_started_ns=start,client_complete_ns=end)
        return o

    def close(self): self.conn.close()
