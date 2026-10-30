"""Hierarchical training-only sampling; validation retains its natural distribution."""
from collections import defaultdict
import numpy as np

PHASE_MASS={'pre_release':.1,'early':.3,'flight':.4,'hold':.2}


class ActionSampler:
    def __init__(self,data,mode='uniform'):
        if mode not in ('uniform','balanced'):raise ValueError(mode)
        self.mode=mode;self.size=len(data);self.probabilities=None
        if mode=='balanced':
            groups=defaultdict(lambda:defaultdict(lambda:defaultdict(lambda:defaultdict(list))))
            strata=getattr(data,'sampling_strata',[('launcher','legacy')]*len(data))
            for i,(kind,seed,phase,rollout) in enumerate(data.sampling_metadata):
                if data.parts['valid'][i].any():groups[(kind,strata[i])][seed][phase][rollout].append(i)
            if not groups:raise ValueError('No valid action waypoints for sampling')
            kinds=defaultdict(int)
            for kind,_ in groups:kinds[kind]+=1
            p=np.zeros(len(data),np.float64)
            for (kind,_),seeds in groups.items():
                for phases in seeds.values():
                    phase_total=sum(PHASE_MASS[k] for k in phases)
                    for phase,rollouts in phases.items():
                        mass=PHASE_MASS[phase]/phase_total/len(kinds)/kinds[kind]/len(seeds)/len(rollouts)
                        for indices in rollouts.values():p[indices]=mass/len(indices)
            self.probabilities=p/p.sum()

    def draw(self,rng,batch):
        if self.mode=='uniform':return rng.integers(self.size,size=batch)
        return rng.choice(self.size,size=batch,p=self.probabilities)

    def summary(self,data):
        p=self.probabilities if self.probabilities is not None else np.full(len(data),1/len(data))
        report={key:defaultdict(float) for key in ('kind','phase','thrower_speed')}
        strata=getattr(data,'sampling_strata',[('launcher','legacy')]*len(data))
        for mass,(kind,_,phase,_),stratum in zip(p,data.sampling_metadata,strata):
            report['kind'][kind]+=float(mass);report['phase'][phase]+=float(mass)
            report['thrower_speed']['/'.join(stratum)]+=float(mass)
        return dict(mode=self.mode,expected_probability={k:dict(v) for k,v in report.items()},
                    scope='Training labels determine sampling only; no metadata enters the policy')
