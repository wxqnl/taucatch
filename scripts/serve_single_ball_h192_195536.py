"""Serve the explicitly trained 12x8ms native SpikeVLA action head on one user-authorized GPU: 0, 5 or 6."""
import argparse
import json
import os
import signal
import threading
import time
from dataclasses import asdict
from multiprocessing.connection import Listener
import numpy as np
import torch

from taucatch.action_h192_195536 import SPEC, build_h192_policy, require_checkpoint
from taucatch.action_link import ADDRESS,AUTHKEY,unpack_action_input
from taucatch.action_policy import action_runtime,pack_state
from taucatch.chooser import to_pixels
from taucatch.compute import selected_gpu
from chooser_server import accept_until_stopped,install_stop_handlers,canonical_device,cache_counts,cache_delta


def infer_action(model,prepare,device,text,sample,runtime=None):
    if runtime is not None and runtime.gpu_preprocess:
        pixels=torch.from_numpy(np.stack((sample['over'],sample['mosaic'])))[None].contiguous().to(device)
    else:pixels=to_pixels(sample['over'][None],sample['mosaic'][None],prepare,device)
    robot=torch.from_numpy(np.asarray(sample['state'],np.float32)[None]).to(device)
    events=torch.from_numpy(np.asarray(sample['events'],np.float32)[None]).to(device)
    out=(model([text],pixels,robot,events) if runtime is None else
         runtime([text],{'pixel_values':pixels},pack_state(robot,events)))
    return out[0]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',required=True)
    p.add_argument('--port',type=int,default=ADDRESS[1])
    p.add_argument('--backend',choices=('eager','cached','cuda_graph','compile','compile_graph'),default='cached')
    p.add_argument('--spike-mode',choices=('none','mixed_fp16','mixed_fp16_ext'),default='none')
    p.add_argument('--runtime-options',help='JSON containing independently verified runtime options')
    a=p.parse_args()
    selected_gpu()
    if not torch.cuda.is_available():p.error('CUDA is unavailable')
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(10*2**30/torch.cuda.get_device_properties(0).total_memory)
    signal.signal(signal.SIGINT,signal.default_int_handler)
    device=canonical_device('cuda');model,prepare=build_h192_policy()
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    require_checkpoint(ck)
    model.load_state_dict(ck['state'],strict=True);model.to(device).eval()
    options=dict(backend=a.backend,spike_mode=a.spike_mode)
    if a.runtime_options:
        with open(a.runtime_options) as stream:options.update(json.load(stream))
    backend=options.pop('backend');spike_mode=options.pop('spike_mode')
    runtime=None if backend=='eager' else action_runtime(model,backend,spike_mode,**options)
    lock=threading.Lock();stop=threading.Event();times=[]
    sample=dict(over=np.zeros((224,224,3),np.uint8),mosaic=np.zeros((224,224,3),np.uint8),
                state=np.zeros(8,np.float32),events=np.zeros((2,6,60,80),np.float16))
    start=time.perf_counter()
    with torch.inference_mode():
        for _ in range(5):infer_action(model,prepare,device,'catch the red ball',sample,runtime)
    torch.cuda.synchronize(device);install_stop_handlers(stop)
    if runtime is not None:runtime._text.clear()  # Startup text is not a prepared episode instruction.
    print(json.dumps(dict(ready=True,port=a.port,checkpoint=a.checkpoint,step=ck['step'],
        action_spec=ck['action_spec'],backend=backend,spike_mode=spike_mode,runtime_options=options,
        physical_gpu=selected_gpu(),startup_s=time.perf_counter()-start)),flush=True)
    def handle(conn):
        with conn,torch.inference_mode():
            while not stop.is_set():
                try:payload=conn.recv_bytes()
                except (EOFError,ConnectionResetError):return
                try:
                    if payload.startswith(b'{'):
                        msg=json.loads(payload)
                        with lock:
                            if runtime is None:raise ValueError('Text operations require a cached runtime')
                            if msg['op']=='clear_text_cache':
                                n=len(runtime._text);runtime._text.clear();out=dict(cleared_entries=n)
                            elif msg['op']=='prepare':
                                before=cache_counts(runtime);torch.cuda.synchronize(device);t0=time.perf_counter()
                                runtime._features([msg['text']],device);torch.cuda.synchronize(device)
                                out=dict(prepared_ms=1000*(time.perf_counter()-t0),text_cache=cache_delta(before,runtime))
                            else:raise ValueError('Unknown control request')
                    else:
                        text,s=unpack_action_input(payload);waiting=time.perf_counter()
                        with lock:
                            lock_ms=1000*(time.perf_counter()-waiting);before=cache_counts(runtime)
                            torch.cuda.synchronize(device);t0=time.perf_counter()
                            actions=infer_action(model,prepare,device,text,s,runtime).float().cpu().numpy()
                            ms=1000*(time.perf_counter()-t0);times.append(ms)
                            if actions.shape!=(12,7) or not np.isfinite(actions).all():raise ValueError('Nonfinite native output')
                            out=dict(actions=actions.tolist(),ms=ms,lock_wait_ms=lock_ms,
                                     text_cache=cache_delta(before,runtime),action_spec=asdict(model.spec))
                    conn.send_bytes(json.dumps(out).encode())
                except Exception as exc:
                    conn.send_bytes(json.dumps(dict(error=f'{type(exc).__name__}: {exc}')).encode())
    try:
        with Listener((ADDRESS[0],a.port),authkey=AUTHKEY) as listener:
            accept_until_stopped(listener,handle,stop)
    finally:
        with lock:
            if times:print(json.dumps(dict(requests=len(times),median_ms=float(np.median(times)),p95_ms=float(np.percentile(times,95)))),flush=True)
            if runtime is not None:print(json.dumps(runtime.metadata()),flush=True);runtime.close()
        print('native action server stopped cleanly',flush=True)


if __name__=='__main__':main()
