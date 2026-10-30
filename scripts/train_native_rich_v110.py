"""Fixed360 rich48 spatial SFT; frozen condition cache, live DVS/native decoder."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from train_action_h96 import atomic_save,loss_fn,prediction
from train_native_space_v91 import Sampler
from train_single_ball_h192_195536 import frozen_digest
from taucatch.action_h192_195536 import SPEC,build_h192_policy,require_checkpoint
from taucatch.action_horizon_data import sha256
from taucatch.action_recovery_training_v1 import set_scope,training_mode
from taucatch.native_frozen_condition_v108 import cached_native_prediction
from taucatch.native_scene_condition_v115 import assemble_condition_cache
from taucatch.native_servo_rich_v109 import KIND,TIME_CONTRACT,SERVO_CONTRACT,require_rich_checkpoint
from taucatch.servo_native_rich_data_v107 import load_rich_servo_native_data


def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n')


def tensor_tree_equal(a,b):
    if torch.is_tensor(a):return torch.is_tensor(b) and torch.equal(a.detach().cpu(),b.detach().cpu())
    if isinstance(a,dict):return isinstance(b,dict) and set(a)==set(b) and all(tensor_tree_equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(tensor_tree_equal(x,y) for x,y in zip(a,b))
    return a==b


@torch.no_grad()
def evaluate_cached(model,data,cache,device,phase_groups,prediction_path,sensor_mode='full'):
    model.eval();totals=np.zeros(5);count=0;predictions=[];scale=torch.tensor(SPEC.scale,device=device)
    if sensor_mode not in ['full','zero_DVS','home_state']:raise ValueError('Unknown diagnostic sensor mode')
    first={}
    for i,(_,path,_) in enumerate(data.rows):first.setdefault(path,i)
    for start in range(0,len(data),8):
        ix=list(range(start,min(start+8,len(data))));x,target,mask=data.batch(ix);target=target.to(device);mask=mask.to(device);condition,padding=cache.batch(ix,device)
        robot=x['state'].to(device);events=x['events'].to(device)
        if sensor_mode=='zero_DVS':events=torch.zeros_like(events)
        if sensor_mode=='home_state':robot=torch.from_numpy(data.parts['state'][[first[data.rows[i][1]] for i in ix]]).to(device)
        pred=cached_native_prediction(model,condition,padding,robot,events);predictions.append(pred.float().cpu().numpy())
        loss=F.smooth_l1_loss(pred,target,reduction='none',beta=.1).mean(-1);pos=((pred[...,:3]-target[...,:3])*scale).norm(dim=-1)*100
        normal=F.cosine_similarity(pred[...,3:6],target[...,3:6],dim=-1).clamp(-1,1).acos()*180/np.pi;closure=(pred[...,6]-target[...,6]).abs()/2;right=(pred[...,6]>=0)==(target[...,6]>=0)
        totals+=np.array([v[mask].sum().item() for v in [loss,pos,normal,closure,right]]);count+=int(mask.sum())
    predictions=np.concatenate(predictions);np.save(prediction_path,predictions,allow_pickle=False)
    err=np.linalg.norm((predictions[...,:3]-data.parts['actions'][...,:3])*np.asarray(SPEC.scale),axis=-1)*100;phases={}
    for name,seed_groups in phase_groups.items():
        ix=np.array([i for indices in seed_groups.values() for i in indices],dtype=int);mask=data.parts['valid'][ix]
        phases[name]=dict(observations=len(ix),valid_points=int(mask.sum()),position_cm=float(err[ix][mask].mean()))
    before=np.array([i for name in ['early','flight'] for indices in phase_groups[name].values() for i in indices],dtype=int)
    return dict(zip(['loss','position_cm','normal_degrees','closure_mae','closure_accuracy'],(totals/count).tolist()),waypoints=count,observations=len(data),by_phase=phases,before_close_position_cm=float(err[before][data.parts['valid'][before]].mean()),sensor_mode=sensor_mode,condition='Successful teacher inputs; sensor interventions are diagnostics, not autonomous catching or VLA benefit')


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);a=p.parse_args();c=json.loads(a.config.read_text());logs=Path(c['logs']);out=Path(c['models'])
    assert os.environ['CUDA_VISIBLE_DEVICES']=='5' and not Path('outputs/logs/selective_monitor/STOP').exists()
    assert c['steps']==360 and c['batch']==8 and not (out/'candidate.pt').exists() and not (out/'last.pt').exists()
    assert Path(c['collection_done']).exists() and sha256(c['dataset_manifest'])==c['dataset_manifest_sha256']
    gate=json.loads(Path(c['cache_gate']).read_text());assert gate['accepted'] and set(gate['models'])=={'initial','pilot_candidate'} and sha256(c['cache_gate'])==c['cache_gate_sha256']
    assert sha256(c['initialization'])==c['initialization_sha256']
    torch.set_num_threads(4);torch.manual_seed(c['seed']);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(10*2**30/torch.cuda.get_device_properties(0).total_memory)
    data=load_rich_servo_native_data(c['data_root']);summaries={s:d.summary() for s,d in data.items()};assert summaries==c['dataset']
    manifest=json.loads(Path(c['dataset_manifest']).read_text());samplers={s:Sampler(d,manifest) for s,d in data.items()}
    ck=torch.load(c['initialization'],map_location='cpu',weights_only=False);require_checkpoint(ck);model,prepare=build_h192_policy();model.load_state_dict(ck['state'],strict=True);del ck
    model.cuda();scope=set_scope(model);before=frozen_digest(model);device=torch.device('cuda:0');model.eval();caches={};cache_records={}
    for split,d in data.items():
        start=time.monotonic();caches[split],reuse=assemble_condition_cache(model,prepare,d,device,c['initialization_sha256'],before,c.get('precomputed_condition_index'));path=logs/f'frozen_condition_{split}.pt'
        atomic_save(path,caches[split].payload(d.fingerprint,before));cache_records[split]=dict(path=str(path),sha256=sha256(path),encoding_seconds=time.monotonic()-start,**reuse)
        print(json.dumps(dict(stage='frozen_condition_cache',split=split,**cache_records[split])),flush=True)
    write(logs/'condition_cache.json',cache_records)
    assembled_gate={}
    with torch.no_grad():
        for split,d in data.items():
            ix=np.linspace(0,len(d)-1,8,dtype=int);x,_,_=d.batch(ix);cond,pad=caches[split].batch(ix,device)
            cached=cached_native_prediction(model,cond,pad,x['state'].to(device),x['events'].to(device));direct=prediction(model,prepare,x,device);error=float((cached-direct).abs().max())
            assembled_gate[split]=dict(accepted=error<=1e-6,max_action_error=error,indices=ix.tolist());assert error<=1e-6,'Assembled scene cache changes full native prediction'
    write(logs/'assembled_cache_gate.json',dict(accepted=True,splits=assembled_gate,policy_updates=0))
    initial={s:evaluate_cached(model,d,caches[s],device,samplers[s].groups,logs/f'initial_{s}_predictions.npy') for s,d in data.items()};write(logs/'initial_metrics.json',initial)
    sensor_diagnostics={'initial':{mode:evaluate_cached(model,data['val'],caches['val'],device,samplers['val'].groups,logs/f'initial_val_{mode}_predictions.npy',mode) for mode in ['zero_DVS','home_state']}}
    params=[p for p in model.parameters() if p.requires_grad];opt=torch.optim.AdamW(params,lr=3e-5,weight_decay=0.);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=360,eta_min=1e-6)
    rng=np.random.default_rng(c['seed']);batches=[];digest=hashlib.sha256();history=[]
    def payload(step):
        return dict(kind=KIND,time_contract=TIME_CONTRACT,action_spec=asdict(SPEC),servo_contract=SERVO_CONTRACT,step=step,budget_steps=360,state=model.state_dict(),dataset=summaries,
            dataset_manifest_sha256=c['dataset_manifest_sha256'],initialization_sha256=c['initialization_sha256'],frozen_upstream_sha256=before,trained_scope_prefixes=['event_encoder.','vla.action_head.'],scope=scope,
            optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),batch_stream_sha256=digest.hexdigest(),sampled_batch_indices=np.stack(batches),history=history,
            training_config_sha256=sha256(a.config),cache_gate_sha256=c['cache_gate_sha256'],condition_cache=cache_records,architecture='native_spikevla_rgb_language_dvs_proprio_action_v1')
    started=time.monotonic();print(json.dumps(dict(stage='initial',metrics=initial,trainable_parameters=scope['trainable_parameters'])),flush=True)
    for step in range(1,361):
        training_mode(model);indices=samplers['train'].draw(rng);batches.append(indices.copy());digest.update(indices.tobytes());x,y,valid=data['train'].batch(indices)
        assert set(x)=={'over','mosaic','events','state','text'} and valid.any();condition,padding=caches['train'].batch(indices,device)
        pred=cached_native_prediction(model,condition,padding,x['state'].to(device),x['events'].to(device));loss=loss_fn(pred,y.to(device),valid.to(device));assert torch.isfinite(loss)
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
        if step==1:
            grad={name:sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None) for name,module in [('DVS',model.event_encoder),('native_head',model.vla.action_head)]};assert min(grad.values())>0
            write(logs/'first_gradient.json',dict(finite_nonzero=True,grad=grad,actor_whitelist=True,upstream_frozen=True))
        opt.step();scheduler.step();history.append(dict(step=step,loss=float(loss.detach())))
        if step==1 or step%20==0:print(json.dumps(dict(step=step,loss=float(loss.detach()),seconds=time.monotonic()-started,peak_gib=torch.cuda.max_memory_allocated()/2**30)),flush=True)
        if step%60==0:atomic_save(out/'last.pt',payload(step))
    assert frozen_digest(model)==before
    final={s:evaluate_cached(model,d,caches[s],device,samplers[s].groups,logs/f'final_{s}_predictions.npy') for s,d in data.items()}
    sensor_diagnostics['final']={mode:evaluate_cached(model,data['val'],caches['val'],device,samplers['val'].groups,logs/f'final_val_{mode}_predictions.npy',mode) for mode in ['zero_DVS','home_state']};write(logs/'sensor_diagnostics.json',sensor_diagnostics)
    candidate=payload(360);candidate['metrics']=dict(initial=initial,final=final);require_rich_checkpoint(candidate);atomic_save(out/'candidate.pt',candidate)
    loaded=torch.load(out/'candidate.pt',map_location='cpu',weights_only=False);require_rich_checkpoint(loaded)
    assert tensor_tree_equal(model.state_dict(),loaded['state']) and tensor_tree_equal(opt.state_dict(),loaded['optimizer']) and tensor_tree_equal(scheduler.state_dict(),loaded['scheduler'])
    assert loaded['numpy_rng']==rng.bit_generator.state and torch.equal(loaded['torch_rng'],torch.get_rng_state()) and torch.equal(loaded['cuda_rng'],torch.cuda.get_rng_state()) and np.array_equal(loaded['sampled_batch_indices'],np.stack(batches))
    gate=dict(position_relative=final['val']['position_cm']/initial['val']['position_cm'],before_close_not_degraded=final['val']['before_close_position_cm']<=initial['val']['before_close_position_cm'],early_not_degraded=final['val']['by_phase']['early']['position_cm']<=initial['val']['by_phase']['early']['position_cm'],closure_not_degraded=final['val']['closure_mae']<=initial['val']['closure_mae']+.05,normal_not_degraded=final['val']['normal_degrees']<=initial['val']['normal_degrees']+10.)
    gate['accepted']=gate['position_relative']<=.8 and all(gate[k] for k in ['before_close_not_degraded','early_not_degraded','closure_not_degraded','normal_not_degraded'])
    result=dict(complete=True,steps=360,checkpoint=str(out/'candidate.pt'),checkpoint_sha256=sha256(out/'candidate.pt'),dataset=summaries,initial=initial,final=final,quality_gate=gate,sensor_diagnostics=sensor_diagnostics,frozen_upstream_exact=True,full_save_load_model_optimizer_scheduler_rng_exact=True,condition_cache=cache_records,training_seconds=time.monotonic()-started,new_policy_updates=360,vla_gain_proven=False,automatic_successor=False,scope='Fixed final360, rich36TRAIN/12VAL, original initialization, no validation checkpoint search. Successful-teacher conditional errors; not autonomous catching. Frozen RGB/text cache only; live DVS and action head. No interrupted continuation has been tested.')
    write(logs/'training.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
