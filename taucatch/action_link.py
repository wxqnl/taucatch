"""Version-independent local action RPC: RGB, raw DVS, robot state in; 12x7 actions out."""
import io
import json
import time
from multiprocessing.connection import Client
import numpy as np

ADDRESS=('127.0.0.1',6012)
AUTHKEY=b'taucatch-native-action'


def pack_action_input(text,sample):
    buf=io.BytesIO()
    np.savez(buf,text=np.array(text),**{k:sample[k] for k in ('over','mosaic','state','events')})
    return buf.getvalue()


def unpack_action_input(payload):
    with np.load(io.BytesIO(payload),allow_pickle=False) as z:
        sample={k:z[k].copy() for k in ('over','mosaic','state','events')}
        text=str(z['text'])
    if sample['over'].shape!=(224,224,3) or sample['mosaic'].shape!=(224,224,3):
        raise ValueError('Action RGB views must be 224x224x3')
    if sample['state'].shape!=(8,) or sample['events'].shape!=(2,6,60,80):
        raise ValueError('Action RPC requires robot state8 and stereo DVS6x60x80')
    if not np.isfinite(sample['state']).all() or not np.isfinite(sample['events']).all():
        raise ValueError('Nonfinite policy observation')
    return text,sample


class ActionClient:
    def __init__(self,address=ADDRESS):self.conn=Client(address,authkey=AUTHKEY)
    def _call(self,payload):
        t0=time.perf_counter();self.conn.send_bytes(payload)
        out=json.loads(self.conn.recv_bytes());out['roundtrip_ms']=1000*(time.perf_counter()-t0)
        if 'error' in out:raise RuntimeError(out['error'])
        return out
    def predict(self,text,sample):
        # Full request latency includes packing, transfer, lock wait and response parsing.
        t0=time.perf_counter();out=self._call(pack_action_input(text,sample))
        out['roundtrip_ms']=1000*(time.perf_counter()-t0)
        a=np.asarray(out['actions'],np.float32)
        if a.shape!=(12,7) or not np.isfinite(a).all():raise ValueError('Invalid native action reply')
        return out
    def prepare(self,text):return self._call(json.dumps(dict(op='prepare',text=text)).encode())
    def clear_text_cache(self):return self._call(json.dumps(dict(op='clear_text_cache')).encode())
    def close(self):self.conn.close()
