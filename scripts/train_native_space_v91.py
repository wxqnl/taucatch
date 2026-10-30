"""Fixed120-step spatial SFT on the audited actual-state pilot, no successor."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from train_action_h96 import atomic_save,evaluate,loss_fn,prediction
from train_single_ball_h192_195536 import frozen_digest
from taucatch.action_h192_195536 import SPEC,build_h192_policy,require_checkpoint
from taucatch.action_horizon_data import sha256
from taucatch.action_recovery_training_v1 import set_scope,training_mode
from taucatch.native_servo_space_v90 import KIND,TIME_CONTRACT,SERVO_CONTRACT,require_space_checkpoint
from taucatch.servo_native_data_v88 import load_servo_native_data


class Sampler:
    def __init__(self,data,manifest):
        physical={r['seed']:json.loads(Path(r['physical']['path']).read_text()) for r in manifest['records']}
        self.groups={k:{} for k in ['early','flight','hold']}
        for i,(_,path,original) in enumerate(data.rows):
            seed=int(Path(path).parent.name)
            proof=physical[seed]['sample_proof'][original]
            phase='hold' if proof['timer_closed'] else 'early' if proof['t']<.08 else 'flight'
            self.groups[phase].setdefault(seed,[]).append(i)
        assert all(self.groups.values())
    def draw(self,rng):
        indices=[]
        for phase in ['early']*2+['flight']*4+['hold']*2:
            seeds=sorted(self.groups[phase]);seed=seeds[int(rng.integers(len(seeds)))]
            values=self.groups[phase][seed];indices.append(values[int(rng.integers(len(values)))])
        return np.array(indices,dtype=np.int64)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);a=p.parse_args()
    c=json.loads(a.config.read_text());logs=Path(c['logs']);out=Path(c['models'])
    assert os.environ['CUDA_VISIBLE_DEVICES']=='5' and not Path('outputs/logs/selective_monitor/STOP').exists()
    assert c['steps']==120 and c['batch']==8 and not (out/'candidate.pt').exists()
    assert sha256(c['initialization'])==c['initialization_sha256']
    torch.set_num_threads(4);torch.manual_seed(c['seed']);torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(10*2**30/torch.cuda.get_device_properties(0).total_memory)
    data=load_servo_native_data(c['data_root']);summaries={s:d.summary() for s,d in data.items()}
    assert summaries==c['dataset']
    manifest=json.loads((Path(c['data_root'])/'manifest.json').read_text());sampler=Sampler(data['train'],manifest)
    ck=torch.load(c['initialization'],map_location='cpu',weights_only=False);require_checkpoint(ck)
    model,prepare=build_h192_policy();model.load_state_dict(ck['state'],strict=True);del ck
    model.cuda();scope=set_scope(model);before=frozen_digest(model);device=torch.device('cuda:0')
    params=[p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(params,lr=3e-5,weight_decay=0.);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=120,eta_min=1e-6)
    rng=np.random.default_rng(c['seed']);batches=[];digest=hashlib.sha256();history=[]
    initial={s:evaluate(model,prepare,d,device,8) for s,d in data.items()}
    (logs/'initial_metrics.json').write_text(json.dumps(initial,indent=2)+'\n')
    print(json.dumps(dict(stage='initial',metrics=initial,scope=scope,sampling={k:{s:len(v) for s,v in g.items()} for k,g in sampler.groups.items()})),flush=True)
    def payload(step):
        return dict(kind=KIND,time_contract=TIME_CONTRACT,action_spec=asdict(SPEC),servo_contract=SERVO_CONTRACT,
            step=step,budget_steps=120,state=model.state_dict(),dataset=summaries,
            dataset_manifest_sha256=c['dataset_manifest_sha256'],initialization_sha256=c['initialization_sha256'],
            frozen_upstream_sha256=before,trained_scope_prefixes=['event_encoder.','vla.action_head.'],scope=scope,
            optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),numpy_rng=rng.bit_generator.state,
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),batch_stream_sha256=digest.hexdigest(),history=history,
            training_config_sha256=sha256(a.config),architecture='native_spikevla_rgb_language_dvs_proprio_action_v1')
    started=time.monotonic()
    for step in range(1,121):
        training_mode(model);indices=sampler.draw(rng);batches.append(indices.copy());digest.update(indices.tobytes())
        x,y,mask=data['train'].batch(indices);assert set(x)=={'over','mosaic','events','state','text'} and mask.any()
        pred=prediction(model,prepare,x,device);loss=loss_fn(pred,y.to(device),mask.to(device))
        assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
        if step==1:
            grad={name:sum(p.grad.abs().sum().item() for p in module.parameters() if p.grad is not None) for name,module in [('DVS',model.event_encoder),('native_head',model.vla.action_head)]}
            assert min(grad.values())>0
            (logs/'first_gradient.json').write_text(json.dumps(dict(finite_nonzero=True,grad=grad,actor_whitelist=True),indent=2)+'\n')
        opt.step();scheduler.step();history.append(dict(step=step,loss=float(loss)))
        if step==1 or step%20==0:print(json.dumps(dict(step=step,loss=float(loss),seconds=time.monotonic()-started,peak_gib=torch.cuda.max_memory_allocated()/2**30)),flush=True)
        if step%40==0:atomic_save(out/'last.pt',payload(step))
    assert frozen_digest(model)==before
    final={s:evaluate(model,prepare,d,device,8) for s,d in data.items()}
    candidate=payload(120);candidate['metrics']=dict(initial=initial,final=final);require_space_checkpoint(candidate)
    atomic_save(out/'candidate.pt',candidate)
    loaded=torch.load(out/'candidate.pt',map_location='cpu',weights_only=False);require_space_checkpoint(loaded)
    assert all(torch.equal(v.detach().cpu(),loaded['state'][k]) for k,v in model.state_dict().items())
    assert loaded['numpy_rng']==rng.bit_generator.state and torch.equal(loaded['torch_rng'],torch.get_rng_state())
    np.save(logs/'batch_indices.npy',np.stack(batches),allow_pickle=False)
    gate=dict(position_relative=final['val']['position_cm']/initial['val']['position_cm'],
        closure_not_degraded=final['val']['closure_mae']<=initial['val']['closure_mae']+.05,
        normal_not_degraded=final['val']['normal_degrees']<=initial['val']['normal_degrees']+10.)
    gate['accepted']=gate['position_relative']<=.8 and gate['closure_not_degraded'] and gate['normal_not_degraded']
    result=dict(complete=True,steps=120,checkpoint=str(out/'candidate.pt'),checkpoint_sha256=sha256(out/'candidate.pt'),dataset=summaries,
        initial=initial,final=final,quality_gate=gate,frozen_upstream_exact=True,full_save_load_state_exact=True,
        batch_stream_sha256=digest.hexdigest(),new_policy_updates=120,vla_gain_proven=False,automatic_successor=False,
        scope='Fixed final120step pilot SFT, five successful TRAIN scenes, two VAL scenes; RGB/text/condition projectors frozen. Teacher-conditional errors, not autonomous catch or generalization. No validation checkpoint search or RL.')
    (logs/'training.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
