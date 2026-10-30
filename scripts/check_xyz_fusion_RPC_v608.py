"""One actual condition-only RPC then three distinct saved flight inputs."""
import argparse
import json
from pathlib import Path
import time
import numpy as np

from taucatch.action_horizon_data import sha256
from taucatch.xyz_fusion_rpc_v602 import PrethrowClient


def main():
    p=argparse.ArgumentParser();p.add_argument('--configuration',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    assert not a.out.exists();c=json.loads(a.configuration.read_text())
    with np.load(c['preflight_condition_reference'],allow_pickle=False)as z:rgb={k:z[k].copy()for k in z.files}
    client=PrethrowClient(('127.0.0.1',c['port']),sha256(c['artifact']),sha256(c['artifact']),c['preflight_mode'])
    rows=[]
    try:
        text=client.control('prepare','catch the ball')
        prep=client.predict(rgb,'condition','new_fusion_SFT_transport_v602')
        release=time.perf_counter_ns()
        for i in [1,2,3]:
            with np.load(c['preflight_inputs'][i],allow_pickle=False)as z:
                s={k:z[k].copy()for k in ['state','events','context','observed_t']}
            s.update(release_ns=release,preparation_complete_ns=prep['client_complete_ns'])
            r=client.predict(s,'fast','new_fusion_SFT_transport_v602');rows.append(r)
            a.out.with_name(a.out.stem+'_journal.json').write_text(json.dumps(dict(preparation=prep,rows=rows),indent=2)+'\n')
    finally:client.close()
    fast=[r['roundtrip_ms']for r in rows]
    result=dict(accepted=True,actual_RPC_calls=4,joint_policy_sha256=sha256(c['artifact']),condition_RPC_ms=prep['roundtrip_ms'],text_RPC_ms=text['roundtrip_ms'],
        fast_median_RPC_ms=float(np.median(fast)),fast_p95_RPC_ms=float(np.percentile(fast,95)),fast_max_RPC_ms=float(max(fast)),latency_sample_size=3,
        provisional_latency_accepted=bool(prep['roundtrip_ms']<=c['condition_RPC_budget_ms'] and max(fast)<=c['fast_RPC_p95_budget_ms']),
        physical_trials=0,optimizer_steps=0,rows=rows,preparation=prep,
        scope='Genuine held-world RGB plus saved TRAIN DVS/state/context for transport only; no physical release or live-context quality proof')
    a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items()if k not in ['rows','preparation']}))


if __name__=='__main__':main()
