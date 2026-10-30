"""Independently rebuild every saved preparation and original full flight call."""
import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
import torch

from taucatch.action_horizon_data import sha256
from taucatch.xyz_fusion_runtime_v601 import FusionPrethrowRuntime as RawPrethrowRuntime, KIND, OBSERVATION
from taucatch.xyz_fusion_rpc_v602 import BEHAVIOR
from taucatch.xyz_fusion_rpc_v602 import unpack
from taucatch.xyz_prethrow_v198 import admit_clocks



def main():
    p=argparse.ArgumentParser();p.add_argument('--configuration',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    assert os.environ['CUDA_VISIBLE_DEVICES']=='5'and not a.out.exists()
    c=json.loads(a.configuration.read_text())
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(10*2**30/torch.cuda.get_device_properties(0).total_memory)
    gate=json.loads(Path(c['feature_gate']).read_text())
    e=RawPrethrowRuntime(c['artifact'],json.loads(Path(gate['runtime_options']).read_text()))
    current=None;rows=[]
    try:
        with torch.inference_mode():
            for path in sorted((Path(c['root'])/'tape').glob('[0-9]*.json')):
                r=json.loads(path.read_text());assert r['id']==len(rows);assert r['fusion_spec']==__import__('taucatch.xyz_fusion_policy_v596',fromlist=['FUSION_SPEC']).FUSION_SPEC
                assert r['kind']==KIND and r['behavior_contract']==BEHAVIOR
                assert r['observation_contract']==OBSERVATION and r['joint_policy_sha256']==sha256(c['artifact'])
                assert r['base_initialization_sha256']==sha256(c['base_checkpoint'])
                assert r['selected_joint_gradient_steps']==16 and r['raw_RL_steps']>0 and not r['sampling_allowed']
                for key in ['input_record','feature_record']:assert sha256(r[key])==r[key+'_sha256']
                s,mode,op,episode=unpack(Path(r['input_record']).read_bytes())
                saved=torch.load(r['feature_record'],map_location='cpu',weights_only=False)
                if op=='condition':
                    result=e.prepare_condition('catch the ball',s)
                    assert torch.equal(result[0].cpu(),saved['condition'])and torch.equal(result[1].cpu(),saved['padding'])
                    assert r['condition_only']and not any(k in r for k in ['mean','actions','latent','log_prob'])
                    current=dict(sample=s,episode=episode,record=r)
                else:
                    assert current is not None and current['episode']==episode
                    rgb=current['sample'];pr=current['record'];assert r['condition_record_id']==pr['id']
                    admit_clocks(rgb['capture_ns'],rgb['observed_ns'],pr['encoded_ns'],s['preparation_complete_ns'],s['release_ns'],time.perf_counter_ns())
                    result=e.engine.encode('catch the ball',dict(s,over=rgb['over'],mosaic=rgb['mosaic']),cached=False)
                    for key,value in zip(['features','context','mean','value'],result):assert torch.equal(saved[key],value.cpu()),key
                    expected=saved['mean'][0].numpy() if mode=='mean' else np.zeros((12,3),np.float32)
                    assert np.array_equal(expected,np.asarray(r['actions'],np.float32))
                    assert r['latent'] is None and r['log_prob'] is None and r['behavior_index'] is None
                rows.append(dict(id=r['id'],op=op,episode=episode,exact=True))
    finally:e.close()
    assert c['minimum_audit_requests']<=len(rows)<=c['maximum_total_requests']
    result=dict(accepted=True,n=len(rows),rows=rows,
        actual_prerelease_condition_and_full_native_flight_exact=True,
        sampling_allowed=False,old_behavior_probability_reuse=False,joint_policy_sha256=sha256(c['artifact']),
        condition_only_has_no_actions_or_flight_inputs=True,
        physical_replays=0,optimizer_steps=0,VLA_gain_proven=False)
    a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items()if k!='rows'}))


if __name__=='__main__':main()
