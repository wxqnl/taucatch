"""Finite raw-native fusion SFT, three feasible failure directions and33 successes."""
import fcntl,json,os,subprocess
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from taucatch.action_horizon_data import sha256
from taucatch.native_servo_rich_v109 import build_rich_policy
from taucatch.xyz_fusion_policy_v596 import FusionXYZActor,FUSION_SPEC,KIND
from taucatch.xyz_raw_gradient_v404 import load_saved_input,set_gradient_scope,differentiable_xyz
from taucatch.xyz_teacher_objective_v325 import trajectory_loss
E=Path('outputs/logs/xyz_neighbor_215536');OUT=Path('outputs/logs/xyz_fusion_hardnegative_v617');FAIL=[1100115,1100040,1100035]

def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']=='5'and not OUT.exists();c=json.loads((E/'fit_hardnegative_configuration.json').read_text())
    for key in ['source_sha256','protected_models']:
        for f,h in c[key].items():assert sha256(f)==h,f
    for name in ['outcome_data_CPU_audit.json','fusion_initializer_CPU_gate.json']:assert json.loads((E/name).read_text())['accepted']
    assert json.loads(Path('outputs/logs/xyz_neighbor35_teacher_v591/completion.json').read_text())['controlled_rescue']
    with Path('outputs/logs/action_gpu5.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);policy=json.loads(Path('outputs/logs/selective_gpu_policy.json').read_text());assert policy['allowed_compute_gpus']==[0,5]and policy['disabled_compute_gpus']==[6]
        assert int(subprocess.check_output(['nvidia-smi','-i','5','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))>=12288
        torch.set_num_threads(4);torch.manual_seed(202610272157);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.cuda.set_per_process_memory_fraction(10*2**30/torch.cuda.get_device_properties(0).total_memory)
        parent=torch.load(c['parent'],map_location='cpu',weights_only=False);assert parent['kind']=='native_xyz_raw_RL_adapter_v464'and parent['policy_version']==2
        model,_,ck=build_rich_policy(parent['base_initialization']);params=dict(model.named_parameters())
        with torch.no_grad():
            for n,v in parent['native_trainable_state'].items():params[n].copy_(v)
        actor=FusionXYZActor(ck);actor.load_base_state(parent['actor_state']);model.cuda();actor.cuda();scope=set_gradient_scope(model,actor);actor.freeze_base()
        scope.update(XYZ_base_frozen=True,fusion_parameters=sum(p.numel()for p in actor.fusion.parameters()),XYZ_trainable_parameters=sum(p.numel()for p in actor.parameters()if p.requires_grad),fusion_spec=FUSION_SPEC)
        frozen={n:p._version for n,p in model.state_dict(keep_vars=True).items()if not p.requires_grad};base_before={n:p.detach().cpu().clone()for n,p in actor.state_dict().items()if not n.startswith('fusion.')};fusion_initial={n:p.detach().cpu().clone()for n,p in actor.fusion.state_dict().items()}
        m=json.loads(Path(c['manifest']).read_text());all_data=torch.load(m['targets'],map_location='cpu',weights_only=False);reserve=torch.isin(all_data['seed'],torch.tensor(m['reserved_hard_holdout_geometries']));assert int(reserve.sum())==8
        global_ids=(~reserve).nonzero().flatten();d={k:(v[:,global_ids]if k=='features'else v[global_ids])for k,v in all_data.items()};records=[m['records'][i]for i in global_ids.tolist()];raw=[load_saved_input(r)for r in records]
        assert len(raw)==168 and not torch.isin(d['seed'],torch.tensor([1100067,1100058,1100088])).any()
        target=d['target'].cuda();mask=d['mask'].cuda();seed=d['seed'];original=seed==1100121;newhold=torch.isin(seed,torch.tensor(m['selection_holdout_geometries']));train=~(original|newhold);failed={s:seed==s for s in FAIL}
        assert int(train.sum())==144 and int(original.sum())==8 and int(newhold.sum())==16
        groups={int(s):((seed==s)&train&d['mask'].any(1)).nonzero().flatten().tolist()for s in torch.unique(seed[train])};assert len(groups)==36 and all(len(groups[s])==4 for s in FAIL);other=[s for s in groups if s not in FAIL];assert len(other)==33
        assert not target[(train&~torch.isin(seed,torch.tensor(FAIL))).to('cuda')].any()
        hr=json.loads(Path('outputs/logs/xyz_confirmation_v449/1100088_zero/zero.json').read_text());hp=json.loads(Path(hr['physical']).read_text());hard=[load_saved_input(dict(feature_record=q['rpc']['feature_record'],feature_record_sha256=q['rpc']['feature_record_sha256']))for q in hp['requests']]
        def predict(items,embedding=False):
            values=[];rep=[]
            for x in items:
                y,f=differentiable_xyz(model,actor,x,'cuda');values.append(y)
                if embedding:rep.append(torch.cat((f.mean((0,2)),x['context'].to('cuda')),1))
            return (torch.cat(values),torch.cat(rep))if embedding else torch.cat(values)
        with torch.no_grad():
            initial,embedding=predict(raw,True);initial_hard=predict(hard);oldhold=float(trajectory_loss(initial[original],target[original],mask[original]));hardenergy=float(initial_hard.square().mean())
            fresh=[i for i,g in enumerate(global_ids.tolist())if g>=152];assert len(fresh)==16
            saved=torch.tensor([json.loads(Path(records[i]['feature_record']).with_suffix('.json').read_text())['mean']for i in fresh],device='cuda');numeric=float((initial[fresh]-saved).abs().max());assert numeric<=5e-6
        def metrics(pred):
            with torch.no_grad():
                h=predict(hard);v=dict(train_MSE=float(trajectory_loss(pred[train],target[train],mask[train])),original_hold_zero_MSE=float(trajectory_loss(pred[original],target[original],mask[original])),selection_hold_zero_MSE=float(trajectory_loss(pred[newhold],target[newhold],mask[newhold])),hard88_output_energy=float(h.square().mean()),hard88_max_mean_drift=float((h-initial_hard).abs().max()))
                for s,ids in failed.items():v['failed'+str(s)+'_MSE']=float(trajectory_loss(pred[ids],target[ids],mask[ids]))
                positive=pred[original|newhold];v['positive_max_norm_cm']=float(positive.norm(dim=-1).max()*6.);v['positive_RMS_norm_cm']=float(positive.square().sum(-1).mean().sqrt()*6.);v['hard88_max_norm_cm']=float(h.norm(dim=-1).max()*6.);v['hard88_RMS_norm_cm']=float(h.square().sum(-1).mean().sqrt()*6.)
                v['guard_accepted']=bool(np.isfinite(list(v.values())).all()and v['positive_max_norm_cm']<=3. and v['positive_RMS_norm_cm']<=1.5 and v['hard88_max_norm_cm']<=3. and v['hard88_RMS_norm_cm']<=1.5);return v
        def snapshot():return dict(native={n:p.detach().cpu().clone()for n,p in model.named_parameters()if p.requires_grad},actor={n:p.detach().cpu().clone()for n,p in actor.state_dict().items()})
        first=metrics(initial);best=snapshot();beststep=0;bestvalue=1.;history=[dict(step=0,**first)];batches=[];gradient={};OUT.mkdir()
        with torch.no_grad():
            center=embedding[train].mean(0);spread=embedding[train].std(0).clamp_min(.001);scaled=(embedding-center)/spread;centroids={sid:scaled[ids].mean(0)for sid,ids in groups.items()};neighbors={sid:sorted(other,key=lambda q:float((centroids[sid]-centroids[q]).square().mean()))[:5]for sid in FAIL}
        opt=torch.optim.AdamW([dict(params=[p for p in model.parameters()if p.requires_grad],lr=1e-7),dict(params=list(actor.fusion.parameters()),lr=5e-4)],weight_decay=0.);rng=np.random.default_rng(20261027221536)
        for step in range(1,257):
            assert not Path('outputs/logs/selective_monitor/STOP').exists()
            if step%3==0:positive=[int(s)for s in rng.choice(other,3,replace=False)]
            else:
                positive=[]
                for sid in FAIL:
                    pool=[q for q in neighbors[sid]if q not in positive]or[q for q in other if q not in positive];positive.append(int(rng.choice(pool)))
            ss=FAIL+positive;ids=[int(rng.choice(groups[s]))for s in ss];assert all(bool(train[i])for i in ids);batches.append([int(global_ids[i])for i in ids]);opt.zero_grad(set_to_none=True)
            for i in ids:
                pred,_=differentiable_xyz(model,actor,raw[i],'cuda');strength=1. if int(seed[i])in FAIL else 8.;loss=strength*trajectory_loss(pred,target[i:i+1],mask[i:i+1])+.1*(pred-initial[i:i+1]).square().mean();(loss/6).backward()
            if step==1:
                for label,pp in [('DVS',model.event_encoder.parameters()),('native_decoder',(p for n,p in model.named_parameters()if n.startswith('vla.action_head.')and p.requires_grad)),('fusion',actor.fusion.parameters())]:
                    gs=[p.grad for p in pp if p.grad is not None];assert gs and all(torch.isfinite(g).all()for g in gs);gradient[label]=sum(float(g.abs().sum())for g in gs);assert gradient[label]>0
                assert all(p.grad is None for n,p in actor.named_parameters()if not n.startswith('fusion.'))
            norm=torch.nn.utils.clip_grad_norm_([p for p in list(model.parameters())+list(actor.parameters())if p.requires_grad],1.,error_if_nonfinite=True);opt.step()
            if step%8==0:
                with torch.no_grad():current=metrics(predict(raw))
                history.append(dict(step=step,gradient_norm=float(norm),**current));failgood=all(current['failed'+str(s)+'_MSE']<=.8*first['failed'+str(s)+'_MSE']for s in FAIL)
                score=sum(current['failed'+str(s)+'_MSE']/first['failed'+str(s)+'_MSE']for s in FAIL)/3;current['mean_failure_MSE_ratio']=score
                if current['guard_accepted']and failgood and score<bestvalue-1e-6:best=snapshot();beststep=step;bestvalue=score
                (OUT/'progress.json').write_text(json.dumps(dict(step=step,selected_steps=beststep,history=history),indent=2)+'\n')
        assert frozen=={n:p._version for n,p in model.state_dict(keep_vars=True).items()if not p.requires_grad}
        for n,x in actor.state_dict().items():
            if not n.startswith('fusion.'):assert torch.equal(x.cpu(),base_before[n]),n
        with torch.no_grad():
            for n,p in model.named_parameters():
                if p.requires_grad:p.copy_(best['native'][n].to(p.device))
            actor.load_state_dict(best['actor'],strict=True);final=metrics(predict(raw))
        artifact=dict(parent,kind=KIND,fusion_spec=FUSION_SPEC,fusion_initial_state=fusion_initial,hardnegative_neighbors=neighbors,hardnegative_embedding_source='initial_FA_native_pooled256_context22_TRAIN_only',offline_screen_only=True,native_trainable_state=best['native'],actor_state=best['actor'],recovery_parent=c['parent'],recovery_parent_sha256=sha256(c['parent']),policy_version=2+int(beststep>0),supervised_recovery_steps=beststep,attempted_supervised_recovery_steps=256,supervised_manifest=c['manifest'],supervised_manifest_sha256=sha256(c['manifest']),supervised_configuration_sha256=sha256(E/'fit_hardnegative_configuration.json'),supervised_history=history,supervised_sampled_indices=batches,reserved_confirmation_inputs_never_GPU_loaded=[1100067,1100058],old_behavior_probability_reuse=False,optimizer_resume_supported=False)
        path=OUT/'candidate.pt';torch.save(artifact,path)
        result=dict(accepted=True,updated=beststep>0,selected_steps=beststep,attempted_steps=256,initial=first,selected=final,history=history,scope=scope,gradient_L1=gradient,neutral_current_raw_mean_max_delta=numeric,base_XYZ_RGB_text_absolute_projection_statistics_frozen=True,holdouts_never_trained=[1100121,1100088,*m['new_holdout_geometries']],reserved67_58_not_GPU_loaded_or_selected=True,offline_screen_only=True,hardnegative_neighbors=neighbors,training_development_observations=168,training_observations=144,training_valid_observations=sum(len(x)for x in groups.values()),sampled_indices=batches,candidate=str(path),candidate_sha256=sha256(path),new_physical_trials=0,VLA_gain_proven=False,automatic_successor=False,peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30)
        (OUT/'decision.json').write_text(json.dumps(result,indent=2)+'\n');(OUT/'done').write_text(datetime.now(timezone.utc).isoformat()+'\n');print(json.dumps({k:result[k]for k in ['updated','selected_steps','initial','selected','peak_allocated_GiB']}))
if __name__=='__main__':main()
