"""Independent CPU provenance/scope/split/selection audit of nonlinear fusion."""
import json,os
from pathlib import Path
import torch
from taucatch.action_horizon_data import sha256
from taucatch.xyz_fusion_policy_v596 import FUSION_SPEC,KIND
E=Path('outputs/logs/xyz_neighbor_215536');ROOT=Path('outputs/logs/xyz_fusion_hardnegative_v617');FAIL={1100035,1100040,1100115}
def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']==''and(ROOT/'done').exists();out=E/'fusion_hardnegative_CPU_audit.json';assert not out.exists();d=json.loads((ROOT/'decision.json').read_text());c=json.loads((E/'fit_hardnegative_configuration.json').read_text())
    a=torch.load(d['candidate'],map_location='cpu',weights_only=False);b=torch.load(c['parent'],map_location='cpu',weights_only=False);assert a['kind']==KIND and a['fusion_spec']==FUSION_SPEC and a['raw_RL_steps']==b['raw_RL_steps']==1 and a['attempted_supervised_recovery_steps']==256
    assert sha256(d['candidate'])==d['candidate_sha256']and a['recovery_parent_sha256']==sha256(c['parent'])and a['supervised_configuration_sha256']==sha256(E/'fit_hardnegative_configuration.json')and a['supervised_manifest_sha256']==sha256(c['manifest'])
    for key in ['base_initialization','base_initialization_sha256','actor_initialization','actor_initialization_sha256','selected_joint_gradient_steps','source_coarse_joint_steps','raw_RL_behavior_counter_next','source_private_RNG_next']:assert a.get(key)==b.get(key),key
    native=0
    for n,x in a['native_trainable_state'].items():
        assert n.startswith(('event_encoder.','vla.action_head.'))and not n.startswith('vla.action_head.decoder.action_projection.')and torch.isfinite(x).all();native+=not torch.equal(x,b['native_trainable_state'][n])
    for n,x in b['actor_state'].items():assert torch.equal(a['actor_state'][n],x),n
    fusion=0
    for n,x in a['actor_state'].items():
        assert torch.isfinite(x).all()
        if n.startswith('fusion.'):fusion+=not torch.equal(x,a['fusion_initial_state'][n[len('fusion.'):]])
    m=json.loads(Path(c['manifest']).read_text());data=torch.load(m['targets'],map_location='cpu',weights_only=False);held={1100121,1100088,*m['new_holdout_geometries']}
    for sid,neighbors in a['hardnegative_neighbors'].items():assert int(sid)in FAIL and len(neighbors)==5 and not set(neighbors)&(held|FAIL)
    for batch in d['sampled_indices']:
        seeds=[int(data['seed'][i])for i in batch];assert len(seeds)==len(set(seeds))==6 and FAIL.issubset(seeds)and not set(seeds)&held and all(bool(data['mask'][i].any())for i in batch)
    assert len(d['sampled_indices'])==256 and d['reserved67_58_not_GPU_loaded_or_selected']and a['reserved_confirmation_inputs_never_GPU_loaded']==[1100067,1100058]
    if d['updated']:
        assert native>0 and fusion>0 and a['policy_version']==3 and d['selected']['guard_accepted'];assert all(d['selected']['failed'+str(s)+'_MSE']<=.8*d['initial']['failed'+str(s)+'_MSE']for s in FAIL)
        assert d['selected']['positive_max_norm_cm']<=3. and d['selected']['positive_RMS_norm_cm']<=1.5 and d['selected']['hard88_max_norm_cm']<=3. and d['selected']['hard88_RMS_norm_cm']<=1.5
        assert d['offline_screen_only']and a['offline_screen_only']
    else:assert native==fusion==0 and a['policy_version']==2
    r=dict(accepted=True,updated=d['updated'],native_tensors_changed=native,fusion_tensors_changed=fusion,base_XYZ_tensors_changed=0,new_fusion_kind_required=True,selected_steps=d['selected_steps'],attempted_steps=256,raw_RL_steps_preserved=1,RNG_lineage_preserved=True,frozen_RGB_text_absolute_projection_and_base_XYZ_exact=True,all1536_sampling_indices_TRAIN_only=True,three_failed_plus_three_success_geometries_per_batch=True,reserved67_58_not_GPU_loaded_or_selected=True,holdouts_never_trained=sorted(held),candidate_sha256=d['candidate_sha256'],offline_screen_only=True,old600_rejection_not_reclassified=True,physical_trials=0,VLA_gain_proven=False)
    out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
if __name__=='__main__':main()
