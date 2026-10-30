"""Fine-tune the native SpikeVLA 12x7 decoder, directly conditioned on DVS and robot state.

Independent 96ms-label experiment uses successful expert demonstrations for supervised imitation. Reported
validation errors are conditional on that filter; full student catches require separate rollouts.
"""
import argparse
import hashlib
import json
import os
import signal
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from taucatch.action_chunk import ActionSpec
from taucatch.action_horizon_data import load_horizon_data
from taucatch.action_h96 import SPEC, CONTRACT, build_h96_policy, require_checkpoint
from taucatch.action_sampling import ActionSampler
from taucatch.chooser import to_pixels
from taucatch.compute import selected_gpu


def prediction(model,prepare,x,device,mask=None):
    pixels=to_pixels(x['over'],x['mosaic'],prepare,device)
    events=x['events'].to(device,dtype=torch.float32)
    robot=x['state'].to(device,dtype=torch.float32)
    text=x['text']
    if mask=='events':events=torch.zeros_like(events)
    if mask=='language':text=['catch a ball']*len(text)
    return model(text,pixels,robot,events)


def loss_fn(pred,target,valid):
    per=F.smooth_l1_loss(pred,target,reduction='none',beta=.1).mean(-1)
    return (per*valid).sum()/valid.sum().clamp_min(1)


@torch.no_grad()
def evaluate(model,prepare,data,device,batch=16,mask=None):
    training=model.training;model.eval();totals=np.zeros(5);count=0
    scale=torch.tensor(SPEC.scale,device=device)
    for start in range(0,len(data),batch):
        x,target,valid=data.batch(range(start,min(start+batch,len(data))))
        target=target.to(device);valid=valid.to(device)
        pred=prediction(model,prepare,x,device,mask)
        err=F.smooth_l1_loss(pred,target,reduction='none',beta=.1).mean(-1)
        pos=((pred[...,:3]-target[...,:3])*scale).norm(dim=-1)*100
        normal=F.cosine_similarity(pred[...,3:6],target[...,3:6],dim=-1).clamp(-1,1).acos()*180/np.pi
        closure=(pred[...,6]-target[...,6]).abs()/2
        right=(pred[...,6]>=0)==(target[...,6]>=0)
        totals+=np.array([v[valid].sum().item() for v in (err,pos,normal,closure,right)])
        count+=int(valid.sum())
    model.train(training)
    return dict(zip(('loss','position_cm','normal_degrees','closure_mae','closure_accuracy'),
                    (totals/max(1,count)).tolist()),waypoints=count,observations=len(data),
                condition='Successful expert demonstrations only')


def atomic_save(path,payload):
    tmp=path.with_suffix('.tmp');torch.save(payload,tmp);tmp.replace(path)


class ValidationView:
    """A fixed, evenly spaced per-rollout probe; full validation is retained at final evaluation."""
    def __init__(self,data,per_rollout):
        from collections import defaultdict
        self.data=data;groups=defaultdict(list)
        for i,(_,path,_) in enumerate(data.rows):groups[path].append(i)
        self.indices=np.array([v[j] for v in groups.values() for j in
            np.unique(np.linspace(0,len(v)-1,min(per_rollout,len(v)),dtype=int))])
    def __len__(self):return len(self.indices)
    def batch(self,indices):return self.data.batch(self.indices[list(indices)])


def groups(model,args):
    vision,head,rest=[],[],[]
    for name,p in model.named_parameters():
        if not p.requires_grad:continue
        (vision if 'vision_encoder' in name else head if name.startswith('event_encoder') or 'action_head' in name else rest).append(p)
    return [dict(params=rest,lr=args.lr),dict(params=vision,lr=args.vision_lr),dict(params=head,lr=args.head_lr)]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',default='data/action_v3');p.add_argument('--overlay',required=True);
    p.add_argument('--run-steps',type=int,default=0,help='Finite interruption budget; zero runs to total steps');p.add_argument('--out',type=Path,required=True)
    p.add_argument('--steps',type=int,default=1000);p.add_argument('--batch',type=int,default=16)
    p.add_argument('--eval-every',type=int,default=200);p.add_argument('--save-every',type=int,default=100)
    p.add_argument('--lr',type=float,default=5e-5);p.add_argument('--vision-lr',type=float,default=1e-5)
    p.add_argument('--head-lr',type=float,default=1e-4);p.add_argument('--gpu-memory-gb',type=float,default=10.)
    p.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    initial=p.add_mutually_exclusive_group();initial.add_argument('--resume',type=Path)
    initial.add_argument('--initialize',type=Path,help='Weights only, new optimizer and explicit new cosine schedule')
    p.add_argument('--sampling',choices=('uniform','balanced'),default='uniform')
    p.add_argument('--val-per-rollout',type=int,default=0,help='Fixed checkpoint-selection probe; 0 uses full val')
    p.add_argument('--milestone-every',type=int,default=0)
    a=p.parse_args();device=torch.device(a.device);torch.set_num_threads(4);torch.manual_seed(0)
    if a.steps<=0 or a.batch<=0 or a.eval_every<=0 or a.save_every<=0 or a.val_per_rollout<0 or a.run_steps<0:p.error('Invalid training budget')
    if a.device=='cuda':
        selected_gpu()
        if not torch.cuda.is_available():p.error('CUDA is unavailable')
        torch.cuda.set_per_process_memory_fraction(a.gpu_memory_gb*2**30/torch.cuda.get_device_properties(0).total_memory)
    torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    last=a.out.with_name(a.out.stem+'_last.pt')
    if not a.resume and (a.out.exists() or last.exists()):p.error('Refuse overwriting an existing experiment')
    a.out.parent.mkdir(parents=True,exist_ok=True)
    data=load_horizon_data(a.data,a.overlay);summary={s:d.summary() for s,d in data.items()}
    sampler=ActionSampler(data['train'],a.sampling)
    validation=ValidationView(data['val'],a.val_per_rollout) if a.val_per_rollout else data['val']
    selection_scope=dict(observations=len(validation),full_observations=len(data['val']),
                         method='fixed evenly spaced per rollout' if a.val_per_rollout else 'full validation')
    print(json.dumps(dict(dataset=summary,args=vars(a)),default=str),flush=True)
    print(json.dumps(dict(sampling=sampler.summary(data['train']),selection_scope=selection_scope)),flush=True)
    model,prepare=build_h96_policy();model.to(device).train()
    opt=torch.optim.AdamW(groups(model,a),weight_decay=0.)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=a.steps,eta_min=1e-6)
    rng=np.random.default_rng(0);best=float('inf');start=0;history=[];stop=[];initialization=None
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:stop.append(True))
    if a.initialize:
        digest=hashlib.sha256(a.initialize.read_bytes()).hexdigest()
        ck=torch.load(a.initialize,map_location='cpu',weights_only=False)
        if ck['action_spec']!=asdict(ActionSpec()):raise ValueError('Expected original 48ms warm-start weights')
        model.load_state_dict(ck['state'])
        initialization=dict(path=str(a.initialize),sha256=digest,source_step=ck['step'],
                            optimizer='reset',schedule='new cosine cycle',source_dataset=ck['dataset'],
                            source_action_spec=ck['action_spec'],target_action_spec=asdict(SPEC),
                            scope='Weights only; retrain on 96ms labels, never serve these initial weights at 8ms')
        print(json.dumps(dict(initialization=initialization)),flush=True)
        del ck
    if a.resume:
        ck=torch.load(a.resume,map_location='cpu',weights_only=False)
        require_checkpoint(ck)
        for key in ('data','overlay','steps','batch','lr','vision_lr','head_lr'):
            if ck['args'][key]!=getattr(a,key):raise ValueError(f'Resume setting differs: {key}')
        for key,default in (('sampling','uniform'),('val_per_rollout',0),('milestone_every',0)):
            if ck['args'].get(key,default)!=getattr(a,key):raise ValueError(f'Resume setting differs: {key}')
        if ck['dataset']!=summary:raise ValueError('Resume demonstration identity differs')
        model.load_state_dict(ck['state']);opt.load_state_dict(ck['optimizer']);sched.load_state_dict(ck['scheduler'])
        rng.bit_generator.state=ck['rng'];torch.set_rng_state(ck['torch_rng'])
        if device.type=='cuda':torch.cuda.set_rng_state(ck['cuda_rng'])
        start=ck['step'];best=ck['best'];history=ck['history']
        initialization=ck.get('initialization')
    def payload(step,full=False):
        ck=dict(state=model.state_dict(),step=step,best=best,history=history,action_spec=asdict(model.spec),
                dataset=summary,args=vars(a),initialization=initialization,selection_scope=selection_scope,
                architecture='native_spikevla_rgb_language_dvs_proprio_action_v1',time_contract=CONTRACT)
        if full:ck.update(optimizer=opt.state_dict(),scheduler=sched.state_dict(),rng=rng.bit_generator.state,
                          torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state() if device.type=='cuda' else None)
        return ck
    begin=time.time()
    for step in range(start,a.steps):
        x,y,valid=data['train'].batch(sampler.draw(rng,a.batch))
        out=prediction(model,prepare,x,device);loss=loss_fn(out,y.to(device),valid.to(device))
        opt.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        if step==start:
            head=sum(p.grad.abs().sum().item() for p in model.vla.action_head.decoder.action_projection.parameters() if p.grad is not None)
            dvs=sum(p.grad.abs().sum().item() for p in model.event_encoder.parameters() if p.grad is not None)
            if head<=0 or dvs<=0:raise RuntimeError('Missing native action-head or DVS gradient')
            print(json.dumps(dict(native_head_gradient=head,dvs_gradient=dvs)),flush=True)
        opt.step();sched.step()
        if step%20==0:
            print(json.dumps(dict(step=step+1,loss=loss.item(),seconds=time.time()-begin,
                peak_gib=torch.cuda.max_memory_allocated()/2**30 if device.type=='cuda' else None)),flush=True)
        n=step+1
        if n%a.eval_every==0 or n==a.steps:
            val=evaluate(model,prepare,validation,device,a.batch);history.append(dict(step=n,val=val,selection_scope=selection_scope))
            print(json.dumps(dict(step=n,val=val)),flush=True)
            if val['loss']<best:best=val['loss'];atomic_save(a.out,payload(n))
        if a.run_steps and n-start>=a.run_steps and n<a.steps:stop.append(True)
        if n%a.save_every==0 or n==a.steps or stop:
            atomic_save(last,payload(n,True))
        if a.milestone_every and n%a.milestone_every==0:
            atomic_save(a.out.with_name(a.out.stem+f'_step{n}.pt'),payload(n))
        if stop:
            print(f'Stopped cleanly after saved step {n}',flush=True);return
    model.load_state_dict(torch.load(a.out,map_location=device,weights_only=False)['state'])
    report=dict(dataset=summary,history=history,initialization=initialization,selection_scope=selection_scope,
                sampling=sampler.summary(data['train']),validation={mask or 'full':evaluate(model,prepare,data['val'],device,a.batch,mask)
                   for mask in (None,'events','language')},limitation='Imitation errors, not closed-loop catch rates')
    a.out.with_suffix('.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
