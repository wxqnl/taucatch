"""Identity-checked XYZ/context RPC, complete client serialization/validation billed."""
import io
import json
from multiprocessing.connection import Client
import socket
import time

import numpy as np

from .action_link import AUTHKEY
from .xyz_context_v134 import CONTEXT_CONTRACT
from .xyz_learning_v158 import KIND, XYZ_CONTRACT, BEHAVIOR_CONTRACT

MODES = ('mean', 'zero')
KEYS = ('over', 'mosaic', 'state', 'events', 'context')


def no_delay(conn):
    sock = socket.socket(fileno=conn.fileno())
    try:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    finally:
        sock.detach()


def validate(out, base, artifact, runtime):
    if out.get('base_checkpoint_sha256') != base or out.get('artifact_sha256') != artifact or out.get('runtime_options_sha256') != runtime:
        raise ValueError('Wrong XYZ encoder/policy/runtime identity')
    if out.get('behavior_contract') != BEHAVIOR_CONTRACT or out.get('policy_version') != 1 or out.get('optimization_steps',0) < 1:
        raise ValueError('Wrong trained candidate behavior/version identity')
    if out.get('kind') != KIND or out.get('xyz_contract') != XYZ_CONTRACT or out.get('context_contract') != CONTEXT_CONTRACT:
        raise ValueError('Wrong XYZ/context execution ABI')


def pack(text, sample, mode):
    if mode not in MODES:
        raise ValueError('Unknown XYZ mode')
    buffer = io.BytesIO()
    np.savez(buffer, text=np.array(text), mode=np.array(mode), observed_t=float(sample['observed_t']),
             **{k: sample[k] for k in KEYS})
    return buffer.getvalue()


def unpack(payload):
    with np.load(io.BytesIO(payload), allow_pickle=False) as z:
        if set(z.files) != set(KEYS) | {'text', 'mode', 'observed_t'}:
            raise ValueError('Unexpected XYZ sensor payload')
        sample = {k: z[k].copy() for k in KEYS}
        sample['observed_t'] = float(z['observed_t'])
        text, mode = str(z['text']), str(z['mode'])
    if mode not in MODES or sample['over'].shape != (224,224,3) or sample['mosaic'].shape != (224,224,3) or sample['state'].shape != (8,) or sample['events'].shape != (2,6,60,80) or sample['context'].shape != (22,):
        raise ValueError('Wrong new XYZ input shape')
    if sample['over'].dtype != np.uint8 or sample['mosaic'].dtype != np.uint8 or any(not np.isfinite(sample[k]).all() for k in ['state', 'events', 'context']) or not np.isfinite(sample['observed_t']):
        raise ValueError('Nonfinite or malformed XYZ sensor input')
    return text, sample, mode


class XYZClient:
    def __init__(self, address, base, artifact, runtime, mode):
        if mode not in MODES:
            raise ValueError(mode)
        self.conn = Client(address, authkey=AUTHKEY)
        no_delay(self.conn)
        self.base, self.artifact, self.runtime, self.mode = base, artifact, runtime, mode

    def _control(self, op, text=None):
        start = time.perf_counter()
        self.conn.send_bytes(json.dumps(dict(op=op, **({} if text is None else dict(text=text)))).encode())
        out = json.loads(self.conn.recv_bytes())
        if 'error' in out:
            raise RuntimeError(out['error'])
        validate(out, self.base, self.artifact, self.runtime)
        out['roundtrip_ms'] = 1000*(time.perf_counter()-start)
        return out

    def clear_text_cache(self):
        return self._control('clear_text_cache')

    def prepare(self, text):
        return self._control('prepare', text)

    def predict(self, text, sample):
        start = time.perf_counter()
        self.conn.send_bytes(pack(text, sample, self.mode))
        out = json.loads(self.conn.recv_bytes())
        if 'error' in out:
            raise RuntimeError(out['error'])
        validate(out, self.base, self.artifact, self.runtime)
        action, mean = np.asarray(out['actions'], np.float32), np.asarray(out['mean'], np.float32)
        if action.shape != (12,3) or mean.shape != (12,3) or not np.isfinite(action).all() or not np.isfinite(mean).all() or np.max(np.abs(action)) > 1.00001 or out['mode'] != self.mode or out['observed_t'] != sample['observed_t']:
            raise ValueError('Wrong XYZ reply')
        expected=mean if self.mode=='mean' else np.zeros((12,3),np.float32)
        if not np.array_equal(action,expected) or out['latent'] is not None or out['log_prob'] is not None:
            raise ValueError('Wrong deterministic candidate action')
        if not out['trained'] or out['learned_output_executed']!=(self.mode=='mean'):
            raise ValueError('Candidate execution attribution mismatch')
        out['client_validation_included'] = True
        out['roundtrip_ms'] = 1000*(time.perf_counter()-start)
        return out

    def close(self):
        self.conn.close()
