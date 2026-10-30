"""Serve chooser scores to the simulation. Run in the cts env on GPU 6.

Default cached backend reuses frozen BERT features across frames. For the optimized CUDA
candidate use --backend compile_graph --spike-mode mixed_fp16_ext. Measure its choice
agreement separately; it is not assumed to have the standard action head's latency/quality.
"""
import argparse
import json
import signal
import threading
import time
from multiprocessing.connection import Listener

import numpy as np
import torch

from taucatch.chooser import build_chooser, chooser_runtime, to_pixels
from taucatch.vla_link import ADDRESS, AUTHKEY, unpack

LOCK = threading.Lock()
TIMES = []


def accept_until_stopped(listener, handle_connection, stop):
    """Keep the main thread interruptible even inside socket accept/authentication.

    Listener.accept also authenticates a client and can block in a C socket call.
    The daemon accept thread owns this blocking operation; the main thread waits
    on an Event so SIGINT/SIGTERM can request cleanup without a terminate fallback.
    """
    errors = []
    def accept_loop():
        try:
            while not stop.is_set():
                conn = listener.accept()
                if stop.is_set():
                    conn.close()
                    return
                threading.Thread(target=handle_connection, args=(conn,), daemon=True).start()
        except Exception as error:
            if not stop.is_set():
                errors.append(error)
                stop.set()
    thread = threading.Thread(target=accept_loop, daemon=True)
    thread.start()
    while not stop.wait(.2):
        pass
    if errors:
        raise errors[0]


def install_stop_handlers(stop):
    # Install after runtime compilation too: external libraries may alter signals.
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda signum, frame: stop.set())


def canonical_device(device):
    """Match cache identity to actual tensor devices (cuda:0, rather than cuda)."""
    device = torch.device(device)
    if device.type == 'cuda' and device.index is None:
        return torch.device('cuda', torch.cuda.current_device())
    return device


def infer(model, prepare, device, text, over, mosaic, state, runtime=None):
    zoom = getattr(model, 'input_mosaic_zoom', 1)
    if zoom != 1:
        from taucatch.choice_zoom import mosaic_candidate_count, zoom_mosaic
        mosaic = zoom_mosaic(mosaic, mosaic_candidate_count(mosaic), zoom)
    if runtime is not None and runtime.gpu_preprocess:
        pixels = torch.from_numpy(np.stack((over, mosaic)))[None].contiguous().to(device)
    else:
        pixels = to_pixels(over[None], mosaic[None], prepare, device)
    state = torch.from_numpy(state[None]).to(device)
    return (model([text], pixels, state) if runtime is None else
            runtime([text], {"pixel_values": pixels}, state))[0]


def cache_counts(runtime):
    if runtime is None or not all(hasattr(runtime, k) for k in ('text_hits', 'text_misses')):
        return None
    return dict(hits=runtime.text_hits, misses=runtime.text_misses)


def cache_delta(before, runtime):
    after = cache_counts(runtime)
    return None if before is None or after is None else {k: after[k] - before[k] for k in before}


def serve(conn, model, prepare, device, runtime=None):
    with conn, torch.inference_mode():
        while True:
            try:
                msg = conn.recv_bytes()
            except (EOFError, ConnectionResetError):
                return
            if msg.startswith(b"{"):
                request = json.loads(msg)
                if request.get("op") == "clear_text_cache":
                    if runtime is None or not hasattr(runtime, '_text'):
                        conn.send_bytes(json.dumps(dict(error="text cache unavailable")).encode())
                        continue
                    # This explicit diagnostic operation clears the shared text cache only.
                    # Call between isolated experiments, never inside concurrent volley rows.
                    with LOCK:
                        entries = len(runtime._text)
                        runtime._text.clear()
                    conn.send_bytes(json.dumps(dict(cleared_entries=entries, scope="server shared text cache")).encode())
                    continue
                if request.get("op") != "prepare" or runtime is None:
                    conn.send_bytes(json.dumps(dict(error="prepare requires a cached runtime backend")).encode())
                    continue
                wait_start = time.perf_counter()
                with LOCK:
                    lock_wait_ms = 1000 * (time.perf_counter() - wait_start)
                    before = cache_counts(runtime)
                    torch.cuda.synchronize(device)
                    t0 = time.perf_counter()
                    runtime._features([request["text"]], canonical_device(device))
                    torch.cuda.synchronize(device)
                    ms = 1000 * (time.perf_counter() - t0)
                    cache = cache_delta(before, runtime)
                conn.send_bytes(json.dumps(dict(prepared_ms=ms, text_cache=cache, lock_wait_ms=lock_wait_ms)).encode())
                continue
            text, over, mosaic, state = unpack(msg)
            wait_start = time.perf_counter()
            with LOCK:
                lock_wait_ms = 1000 * (time.perf_counter() - wait_start)
                before = cache_counts(runtime)
                torch.cuda.synchronize(device)
                t0 = time.perf_counter()
                scores = infer(model, prepare, device, text, over, mosaic, state, runtime).float().cpu().numpy()
                ms = 1000 * (time.perf_counter() - t0)
                TIMES.append(ms)
                cache = cache_delta(before, runtime)
            conn.send_bytes(json.dumps(dict(scores=scores.tolist(), ms=ms,
                                            text_cache=cache, lock_wait_ms=lock_wait_ms)).encode())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default="outputs/models/chooser_v1.pt")
    ap.add_argument("--port", type=int, default=ADDRESS[1])
    ap.add_argument("--backend", choices=["eager", "cached", "cuda_graph", "compile", "compile_graph"], default="cached")
    ap.add_argument("--spike-mode", choices=["none", "mixed_fp16", "mixed_fp16_ext"], default="none")
    a = ap.parse_args()
    if not torch.cuda.is_available():
        ap.error("CUDA is unavailable; this server requires GPU 6")
    if a.backend == "eager" and a.spike_mode != "none":
        ap.error("spike precision options require a runtime backend")
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # Background tmux parents may inherit SIGINT ignored; allow owned clean shutdown.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    device = canonical_device("cuda")
    model, prepare = build_chooser()
    checkpoint = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state"])
    model.input_mosaic_zoom = checkpoint.get('input_transform', {}).get('mosaic_zoom', 1)
    if model.input_mosaic_zoom not in (1, 2, 4):
        ap.error('Unsupported checkpoint mosaic zoom')
    model.to(device).eval()
    runtime = None if a.backend == "eager" else chooser_runtime(model, a.backend, a.spike_mode)
    # Compilation and graph capture finish before accepting any simulation requests.
    t0 = time.perf_counter()
    with torch.inference_mode():
        for _ in range(5):
            infer(model, prepare, device, "catch the red ball", np.zeros((224, 224, 3), np.uint8),
                  np.zeros((224, 224, 3), np.uint8), np.zeros(8, np.float32), runtime)
    torch.cuda.synchronize(device)
    stop = threading.Event()
    print(f"post-warmup SIGINT handler: {signal.getsignal(signal.SIGINT)}", flush=True)
    install_stop_handlers(stop)
    print(f"chooser ready on {ADDRESS[0]}:{a.port}; startup {time.perf_counter() - t0:.1f} s; "
          f"backend={a.backend} spike_mode={a.spike_mode} mosaic_zoom={model.input_mosaic_zoom}", flush=True)
    try:
        with Listener((ADDRESS[0], a.port), authkey=AUTHKEY) as listener:
            accept_until_stopped(listener, lambda conn: serve(conn, model, prepare, device, runtime), stop)
    except KeyboardInterrupt:
        pass
    finally:
        with LOCK:
            if TIMES:
                print(f"{len(TIMES)} choices, median compute {np.median(TIMES):.1f} ms, "
                      f"p95 {np.percentile(TIMES, 95):.1f} ms", flush=True)
            if runtime is not None:
                print(json.dumps(runtime.metadata()), flush=True)
                runtime.close()
        print('chooser stopped cleanly', flush=True)
