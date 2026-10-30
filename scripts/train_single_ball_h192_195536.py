"""Finite explicitly labelled192ms single-ball BC. Sensor inputs only; native DVS encoder and action head trained."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from train_action_h96 import ValidationView, atomic_save, evaluate, loss_fn, prediction
from taucatch.action_h192_195536 import CONTRACT, SPEC, build_h192_policy, require_checkpoint
from taucatch.action_h96 import require_checkpoint as require_initial_checkpoint
from taucatch.action_horizon_data import sha256
from taucatch.action_h192_data_195536 import Data192
from taucatch.action_recovery_training_v1 import set_scope, training_mode
from taucatch.action_sampling import ActionSampler
from taucatch.compute import selected_gpu


def frozen_digest(model):
    h=hashlib.sha256()
    for name,value in model.state_dict().items():
        if not name.startswith(('event_encoder.','vla.action_head.')):
            h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    a=p.parse_args();c=json.loads(a.config.read_text());out=Path(c['models'])
    assert not (out/'last.pt').exists() and not (out/'best.pt').exists()
    selected_gpu();torch.set_num_threads(4);torch.manual_seed(c['seed'])
    torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(10*2**30/torch.cuda.get_device_properties(0).total_memory)
    data={s:Data192(c['source'],c['overlay'],s) for s in ('train','val')}
    summaries={s:d.summary() for s,d in data.items()}
    for s,d in data.items():
        assert d.rollouts >= c['minimum_successful'][s]
    assert sha256(c['initialization'])==c['initialization_sha256']
    ck=torch.load(c['initialization'],map_location='cpu',weights_only=False);require_initial_checkpoint(ck)
    model,prepare=build_h192_policy();model.load_state_dict(ck['state'],strict=True);del ck
    model.cuda();scope=set_scope(model);frozen_before=frozen_digest(model)
    params=[p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(params,lr=c['lr'],weight_decay=0.)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=c['steps'],eta_min=1e-6)
    sampler=ActionSampler(data['train'],'balanced');rng=np.random.default_rng(c['seed'])
    val=ValidationView(data['val'],4);best=float('inf');history=[];device=torch.device('cuda:0')
    def payload(step):
        return dict(state=model.state_dict(),step=step,best=best,history=history,
            action_spec=asdict(SPEC),time_contract=CONTRACT,dataset=summaries,args=c,
            architecture='native_spikevla_rgb_language_dvs_proprio_action_v1',
            initialization_sha256=c['initialization_sha256'],scope=scope,
            optimizer=opt.state_dict(),scheduler=sched.state_dict(),rng=rng.bit_generator.state,
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state())
    print(json.dumps(dict(dataset=summaries,scope=scope,sampling=sampler.summary(data['train']))),flush=True)
    initial=evaluate(model,prepare,val,device,c['batch'])
    Path(c['logs'],'initialization_probe192.json').write_text(json.dumps(dict(metrics=initial,scope='Untrained192ms initialization on expert inputs only; not served or retimed for control'),indent=2)+'\n')
    start=time.monotonic()
    for step in range(1,c['steps']+1):
        training_mode(model)
        x,y,valid=data['train'].batch(sampler.draw(rng,c['batch']))
        pred=prediction(model,prepare,x,device)
        loss=loss_fn(pred,y.to(device),valid.to(device))
        opt.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
        if step==1:
            for module in (model.event_encoder,model.vla.action_head):
                assert sum(p.grad.abs().sum().item() for p in module.parameters() if p.grad is not None)>0
        opt.step();sched.step()
        if step==1 or step%20==0:
            print(json.dumps(dict(step=step,loss=loss.item(),seconds=time.monotonic()-start,
                                 peak_gib=torch.cuda.max_memory_allocated()/2**30)),flush=True)
        if step%c['eval_every']==0 or step==c['steps']:
            metrics=evaluate(model,prepare,val,device,c['batch']);history.append(dict(step=step,val=metrics))
            print(json.dumps(history[-1]),flush=True)
            if metrics['loss']<best:
                best=metrics['loss'];atomic_save(out/'best.pt',payload(step))
        if step%100==0 or step==c['steps']:
            atomic_save(out/'last.pt',payload(step))
    frozen_after=frozen_digest(model);assert frozen_before==frozen_after
    ck=torch.load(out/'best.pt',map_location='cpu',weights_only=False);require_checkpoint(ck)
    model.load_state_dict(ck['state'],strict=True)
    report=dict(complete=True,steps=c['steps'],selected_step=ck['step'],dataset=summaries,
        frozen_upstream_unchanged=True,frozen_sha256=frozen_after,
        best_sha256=sha256(out/'best.pt'),last_sha256=sha256(out/'last.pt'),
        validation=evaluate(model,prepare,data['val'],device,c['batch']),
        limitation='Successful expert-conditional imitation; autonomous real-RPC catches evaluated separately')
    Path(c['logs'],'training.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
