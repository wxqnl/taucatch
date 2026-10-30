"""Finite adapted-native plus XYZ mean/zero RPC; no sampling or optimizer."""
import argparse
import json
import os
from pathlib import Path
import threading
import time
from multiprocessing.connection import Listener
import numpy as np
import torch

from chooser_server import accept_until_stopped, install_stop_handlers
from taucatch.action_horizon_data import sha256
from taucatch.action_link import AUTHKEY
from taucatch.xyz_coarse_contract_v308 import XYZ_CONTRACT
from taucatch.xyz_context_softsign_v273 import CONTEXT_CONTRACT
from taucatch.xyz_fusion_rpc_v602 import unpack, no_delay, BEHAVIOR, NAMESPACE
from taucatch.xyz_fusion_runtime_v601 import FusionPrethrowRuntime as RawPrethrowRuntime, KIND, OBSERVATION
from taucatch.xyz_prethrow_v198 import admit_clocks



def main():
    p=argparse.ArgumentParser();p.add_argument('--configuration',type=Path,required=True);a=p.parse_args()
    c=json.loads(a.configuration.read_text());assert os.environ['CUDA_VISIBLE_DEVICES']=='5'
    gate=json.loads(Path(c['numeric_gate']).read_text());assert gate['accepted']
    assert gate['new_SFT_artifact_sha256']==sha256(c['artifact'])
    assert c['kind']==KIND and c['observation_contract']==OBSERVATION
    tape=Path(c['root'])/'tape';tape.mkdir(exist_ok=False)
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(10*2**30/torch.cuda.get_device_properties(0).total_memory)
    artifact=torch.load(c['artifact'],map_location='cpu',weights_only=False)
    assert artifact['selected_joint_gradient_steps']==16 and artifact['raw_RL_steps']==1 and artifact['supervised_recovery_steps']>0
    feature_gate=json.loads(Path(c['feature_gate']).read_text())
    options=json.loads(Path(feature_gate['runtime_options']).read_text())
    engine=RawPrethrowRuntime(c['artifact'],options)
    with np.load(c['preflight_inputs'][0],allow_pickle=False)as z:
        warm={k:z[k].copy()for k in z.files}
    with torch.inference_mode():
        for _ in range(3):full=engine.anchor('catch the ball',warm)
        assert all(torch.equal(x,y)for x,y in zip(full,engine.build_fast_graph(warm)))
    engine.reset_episode()
    identity=dict(observation_contract=OBSERVATION,kind=KIND,behavior_contract=BEHAVIOR,behavior_namespace=NAMESPACE,
        base_checkpoint_sha256=sha256(c['artifact']),joint_policy_sha256=sha256(c['artifact']),artifact_sha256=sha256(c['artifact']),
        base_initialization_sha256=artifact['base_initialization_sha256'],
        actor_initialization_sha256=artifact['actor_initialization_sha256'],
        xyz_contract=XYZ_CONTRACT,context_contract=CONTEXT_CONTRACT,policy_version=artifact['policy_version'],raw_RL_steps=artifact['raw_RL_steps'],parent_raw_joint_sha256=artifact['parent_raw_joint_sha256'],
        optimization_steps=artifact['supervised_recovery_steps'],supervised_recovery_steps=artifact['supervised_recovery_steps'],
        selected_joint_gradient_steps=artifact['selected_joint_gradient_steps'],
        attempted_joint_gradient_steps=artifact['attempted_joint_gradient_steps'],
        source_coarse_joint_steps=artifact['source_coarse_joint_steps'],
        sampling_allowed=False,old_behavior_probability_reuse=False,fusion_spec=artifact['fusion_spec'])
    (tape/'startup_gate.json').write_text(json.dumps(dict(accepted=True,anchor_fast_exact=True,**identity),indent=2)+'\n')
    stop=threading.Event();install_stop_handlers(stop);lock=threading.Lock()
    counter=conditions=fast_count=0;current=previous_t=release=text=None
    print(json.dumps(dict(ready=True,port=c['port'],**identity)),flush=True)

    def handle(conn):
        nonlocal counter,conditions,fast_count,current,previous_t,release,text
        no_delay(conn)
        with conn,torch.inference_mode():
            while not stop.is_set():
                try:payload=conn.recv_bytes()
                except(EOFError,ConnectionResetError):return
                try:
                    with lock:
                        if payload.startswith(b'{'):
                            msg=json.loads(payload)
                            if msg['op']=='status':out=dict(ready=True,requests=counter)
                            elif msg['op']=='prepare':
                                text=msg['text'];assert text=='catch the ball'
                                engine.engine.runtime._text.clear()
                                engine.engine.runtime._features([text],torch.device('cuda:0'))
                                torch.cuda.synchronize();engine.reset_episode()
                                current=previous_t=release=None;out=dict(prepared=True)
                            else:raise ValueError('Unknown control operation')
                        else:
                            if counter>=c['maximum_total_requests']:raise ValueError('Finite request cap')
                            started=time.perf_counter_ns();s,mode,op,episode=unpack(payload)
                            if text is None:raise ValueError('Prepare instruction explicitly')
                            if op=='condition':
                                if current is not None or conditions>=c['maximum_condition_preparations']:
                                    raise ValueError('Condition already present or cap')
                                if s['observed_ns']>time.perf_counter_ns():raise ValueError('Future image')
                                condition,padding=engine.prepare_condition(text,s)
                                encoded=time.perf_counter_ns()
                                tensors=dict(condition=condition.cpu(),padding=padding.cpu())
                                current=dict(episode=episode,sample=s,encoded_ns=encoded,id=counter)
                                out=dict(encoded_ns=encoded,capture_ns=s['capture_ns'],observed_ns=s['observed_ns'],
                                         condition_only=True,absolute_output_executed=False)
                                conditions+=1
                            else:
                                if current is None or episode!=current['episode'] or fast_count>=c['maximum_fast_requests']:
                                    raise ValueError('Missing episode condition or cap')
                                rgb=current['sample'];release_ns=s['release_ns']
                                admit_clocks(rgb['capture_ns'],rgb['observed_ns'],current['encoded_ns'],
                                             s['preparation_complete_ns'],release_ns,time.perf_counter_ns())
                                if release is None:release=release_ns
                                if release!=release_ns or(previous_t is not None and s['observed_t']<=previous_t):
                                    raise ValueError('Stale flight or changed release epoch')
                                engine.release_condition((rgb['capture_ns']-release)/1e9,(rgb['observed_ns']-release)/1e9)
                                f,ctx,mean,value=engine.fast(s);previous_t=s['observed_t']
                                action=mean.clone() if mode=='mean' else torch.zeros_like(mean)
                                tensors=dict(features=f.cpu(),context=ctx.cpu(),mean=mean.cpu(),value=value.cpu())
                                out=dict(observed_t=s['observed_t'],anchor_observed_t=engine.anchor_observed_t,
                                    anchor_rgb_t=engine.anchor_rgb_t,condition_available_t=(s['preparation_complete_ns']-release)/1e9,
                                    condition_record_id=current['id'],actions=action[0].cpu().tolist(),
                                    mean=mean[0].cpu().tolist(),value=float(value[0]),
                                    latent=None,log_prob=None,behavior_index=None,
                                    absolute_output_executed=False)
                                fast_count+=1
                            i=counter;counter+=1
                            inp,feat=tape/f'{i:05d}.npz',tape/f'{i:05d}.pt'
                            inp.write_bytes(payload);torch.save(tensors,feat)
                            record=dict(out,id=i,op=op,episode=episode,mode=mode,
                                input_record=str(inp),input_record_sha256=sha256(inp),
                                feature_record=str(feat),feature_record_sha256=sha256(feat),
                                server_ms=(time.perf_counter_ns()-started)/1e6,**identity)
                            output=tape/f'{i:05d}.json';output.write_text(json.dumps(record,indent=2)+'\n')
                            out=dict(record,output_record=str(output),output_record_sha256=sha256(output))
                        conn.send_bytes(json.dumps(dict(out,**identity)).encode())
                except Exception as exc:
                    conn.send_bytes(json.dumps(dict(error=f'{type(exc).__name__}: {exc}')).encode())
    try:
        with Listener(('127.0.0.1',c['port']),authkey=AUTHKEY)as listener:
            accept_until_stopped(listener,handle,stop)
    finally:engine.close()


if __name__=='__main__':main()
