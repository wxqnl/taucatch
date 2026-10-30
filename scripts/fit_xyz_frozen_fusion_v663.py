"""Finite CPU-only fusion update, frozen native/base, new zero-success coverage and own-state retention."""
import argparse,copy,json,os
from pathlib import Path
import numpy as np
import torch
from torch import nn
from taucatch.action_horizon_data import sha256
from taucatch.xyz_teacher_objective_v325 import trajectory_loss


def main():
    p=argparse.ArgumentParser();p.add_argument('--configuration',type=Path,required=True);a=p.parse_args();c=json.loads(a.configuration.read_text());root=Path(c['root']);assert os.environ['CUDA_VISIBLE_DEVICES']==''and(root/'feature_bank.done').exists()and not(root/'candidate.pt').exists()
    torch.set_num_threads(1);torch.manual_seed(c['seed']);parent=torch.load(c['parent'],map_location='cpu',weights_only=False);b=torch.load(root/'feature_bank.pt',map_location='cpu',weights_only=False)
    assert b['parent_sha256']==sha256(c['parent'])and b['native_and_base_exactly_frozen']
    fusion=nn.Sequential(nn.LayerNorm(278),nn.Linear(278,128),nn.SiLU(),nn.Linear(128,3));fusion.load_state_dict({k[len('fusion.'):]:v for k,v in parent['actor_state'].items()if k.startswith('fusion.')})
    n=b['development_n'];rn=b['retention_n'];sid=b['seed'];target=b['target'];mask=b['mask'];x=b['inputs'];base=b['base_logits']
    hold=(sid==1100121)|torch.isin(sid,torch.tensor(b['selection_holdout_geometries']));train=~hold
    failures=c['failure_seeds'];groups={int(s):((sid==s)&train&mask.any(1)).nonzero().flatten().tolist()for s in torch.unique(sid[train])}
    positives=[s for s in groups if s not in failures];fresh=[s for s in positives if s in b['new_TRAIN_geometries']];old=[s for s in positives if s not in fresh];assert len(fresh)==11
    assert not target[train&~torch.isin(sid,torch.tensor(failures))].any()
    def predict(ids=None):return torch.tanh(base+(fusion(x)))if ids is None else torch.tanh(base[ids]+fusion(x[ids]))
    with torch.no_grad():initial=predict();assert float((initial-b['parent_mean']).abs().max())<=5e-6
    newpos=train&torch.isin(sid,torch.tensor(fresh))
    def metrics(y):
        with torch.no_grad():
            held=y[:n][hold];hard=y[n+rn:];ret=y[n:n+rn];drift=ret-b['retention_target'];r=dict(new_success_zero_MSE=float(trajectory_loss(y[:n][newpos],target[newpos],mask[newpos])),
                positive_max_norm_cm=float(held.norm(dim=-1).max()*6),positive_RMS_cm=float(held.square().sum(-1).mean().sqrt()*6),
                hard88_max_norm_cm=float(hard.norm(dim=-1).max()*6),hard88_RMS_cm=float(hard.square().sum(-1).mean().sqrt()*6),
                retention_max_drift_cm=float(drift.norm(dim=-1).max()*6),retention_RMS_drift_cm=float(drift.square().sum(-1).mean().sqrt()*6))
            for s in failures:
                ids=sid==s;r['failed'+str(s)+'_MSE']=float(trajectory_loss(y[:n][ids],target[ids],mask[ids]))
            r['guard_accepted']=bool(np.isfinite(list(r.values())).all()and r['positive_max_norm_cm']<=3 and r['positive_RMS_cm']<=1.5 and r['hard88_max_norm_cm']<=3 and r['hard88_RMS_cm']<=1.5 and r['retention_max_drift_cm']<=1 and r['retention_RMS_drift_cm']<=.5);return r
    first=metrics(initial);best=copy.deepcopy(fusion.state_dict());selected=0;score0=1.;history=[dict(step=0,**first)];batches=[];rng=np.random.default_rng(c['seed']);opt=torch.optim.AdamW(fusion.parameters(),lr=c['fusion_lr'],weight_decay=0.)
    for step in range(1,c['maximum_steps']+1):
        assert not Path('outputs/logs/selective_monitor/STOP').exists()
        chosen=failures+[int(s)for s in rng.choice(fresh,2,replace=False)]+[int(s)for s in rng.choice(old,2,replace=False)];ids=[int(rng.choice(groups[s]))for s in chosen];ri=int(rng.integers(rn))+n
        opt.zero_grad(set_to_none=True);y=predict(ids);loss=torch.zeros(())
        for j,i in enumerate(ids):loss+=(1. if int(sid[i])in failures else 8.)*trajectory_loss(y[j:j+1],target[i:i+1],mask[i:i+1])+.1*(y[j:j+1]-initial[i:i+1]).square().mean()
        ret=predict([ri]);loss+=4.*trajectory_loss(ret,b['retention_target'][ri-n:ri-n+1],b['retention_mask'][ri-n:ri-n+1]);(loss/(len(ids)+1)).backward();torch.nn.utils.clip_grad_norm_(fusion.parameters(),1.,error_if_nonfinite=True);opt.step();batches.append(dict(development=ids,retention=ri-n))
        if step%8==0:
            with torch.no_grad():m=metrics(predict())
            fail_ratios=[m['failed'+str(s)+'_MSE']/first['failed'+str(s)+'_MSE']for s in failures];positive_ratio=m['new_success_zero_MSE']/first['new_success_zero_MSE'];score=.5*sum(fail_ratios)/len(fail_ratios)+.5*positive_ratio
            good=all(m['failed'+str(s)+'_MSE']<=1.1*first['failed'+str(s)+'_MSE']for s in [1100115,1100040,1100035])and positive_ratio<=.9
            if 1100072 in failures:good=good and m['failed1100072_MSE']<=.8*first['failed1100072_MSE']
            if good and m['guard_accepted']and score<score0-1e-6:best=copy.deepcopy(fusion.state_dict());selected=step;score0=score
            history.append(dict(step=step,score=score,**m));(root/'progress.json').write_text(json.dumps(dict(step=step,maximum_steps=c['maximum_steps'],selected_steps=selected,history=history),indent=2)+'\n')
    fusion.load_state_dict(best)
    with torch.no_grad():final=metrics(predict())
    actor={k:v.clone()for k,v in parent['actor_state'].items()}
    for k,v in fusion.state_dict().items():actor['fusion.'+k]=v.detach().clone()
    result=dict(parent,actor_state=actor,recovery_parent=c['parent'],recovery_parent_sha256=sha256(c['parent']),policy_version=3+int(selected>0),
        supervised_recovery_steps=parent['supervised_recovery_steps']+selected,incremental_fusion_steps=selected,attempted_incremental_fusion_steps=c['maximum_steps'],
        refinement_type='frozen_native_base_CPU_fusion_only_v663',refinement_manifest=c['manifest'],refinement_manifest_sha256=sha256(c['manifest']),
        frozen_native_parent_sha256=sha256(c['parent']),refinement_feature_bank_sha256=sha256(root/'feature_bank.pt'),refinement_history=history,refinement_batches=batches,
        old_behavior_probability_reuse=False,optimizer_resume_supported=False,native_refinement_steps=0)
    torch.save(result,root/'candidate.pt');r=dict(accepted=True,updated=selected>0,selected_steps=selected,attempted_steps=c['maximum_steps'],initial=first,selected=final,history=history,
        native_and_base_frozen=True,only_fusion_parameters_trained=True,TRAIN_geometries=len(groups),training_valid_observations=sum(len(v)for v in groups.values()),
        old_evaluation_not_used=True,reserved_not_GPU_loaded=b['never_GPU_loaded_reserved'],parent_sha256=sha256(c['parent']),candidate_sha256=sha256(root/'candidate.pt'),
        new_physical_trials=0,VLA_gain_proven=False,canonical_promotion=False,automatic_successor=False)
    (root/'decision.json').write_text(json.dumps(r,indent=2)+'\n');(root/'fit.done').write_text('finite CPUfusion complete\n');print(json.dumps({k:r[k]for k in ['updated','selected_steps','initial','selected']}))


if __name__=='__main__':main()
