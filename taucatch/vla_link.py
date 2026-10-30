"""Link between the simulation (`taucatch` env) and the chooser VLA (`cts` env, GPU 6), plan §5.10.

The two environments have different PyTorch and Python versions, so the chooser runs as a
server (scripts/chooser_server.py) and the simulation calls it over a local socket. Messages are
.npz bytes (no pickling across versions): overview, mosaic (uint8 224 x 224 x 3), state (8,) and
the instruction; the answer is the five scores and the server's compute time.
"""
import io
import json
import time
from multiprocessing.connection import Client

import numpy as np

ADDRESS = ("127.0.0.1", 6011)
AUTHKEY = b"taucatch-chooser"


def pack(text, over, mosaic, state):
    buf = io.BytesIO()
    np.savez(buf, over=over, mosaic=mosaic, state=np.asarray(state, np.float32), text=np.array(text))
    return buf.getvalue()


def unpack(data):
    z = np.load(io.BytesIO(data))
    return str(z["text"]), z["over"], z["mosaic"], z["state"]


class ChooserClient:
    def __init__(self, address=ADDRESS, authkey=AUTHKEY):
        self.conn = Client(address, authkey=authkey)

    def choose(self, text, over, mosaic, state):
        """Scores for marks 1-4 and "none", and the server's compute time (ms)."""
        out = self.choose_detailed(text, over, mosaic, state)
        return np.array(out["scores"]), out["ms"]

    def choose_detailed(self, text, over, mosaic, state):
        """Diagnostic response; roundtrip_ms includes packing, transfer, lock wait and parsing.

        New servers report actual text-cache counter deltas. Older servers leave these
        fields absent; absence is never interpreted as a hit or a cold encoding.
        """
        start = time.perf_counter()
        self.conn.send_bytes(pack(text, over, mosaic, state))
        out = json.loads(self.conn.recv_bytes())
        out["roundtrip_ms"] = 1000 * (time.perf_counter() - start)
        return out

    def clear_text_cache(self):
        """Explicit diagnostics only: clears shared server text entries, preserving graphs.

        Use between isolated experiments, never during a concurrent evaluation queue.
        Does not restart the server, reset neurons or clear CUDA graph state.
        """
        self.conn.send_bytes(json.dumps(dict(op="clear_text_cache")).encode())
        out = json.loads(self.conn.recv_bytes())
        if "error" in out:
            raise RuntimeError(out["error"])
        return out

    def prepare(self, text):
        """Encode a known instruction before release; report its separate wall time (ms)."""
        return self.prepare_detailed(text)["prepared_ms"]

    def prepare_detailed(self, text):
        """Preparation timing and cache evidence, separate from any in-flight request."""
        start = time.perf_counter()
        self.conn.send_bytes(json.dumps(dict(op="prepare", text=text)).encode())
        out = json.loads(self.conn.recv_bytes())
        out["roundtrip_ms"] = 1000 * (time.perf_counter() - start)
        if "error" in out:
            raise RuntimeError(out["error"])
        return out

    def close(self):
        self.conn.close()
