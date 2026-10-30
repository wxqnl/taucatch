"""Native policy action contract and latency-aware absolute waypoint execution.

Task actions are normalized world palm XYZ, palm normal XYZ, and hand closure.
Twelve 4 ms waypoints are executed by a 1 ms joint servo, without target IDs or ball paths.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ActionSpec:
    horizon: int = 12
    dt: float = .004
    action_dim: int = 7
    name: str = 'palm_xyz_normal_closure_v1'
    center: tuple = (.6, 0., 1.2)
    scale: tuple = (.65, .85, .85)

    def encode(self, position, normal, closure):
        p=(np.asarray(position)-self.center)/self.scale
        n=np.asarray(normal,dtype=float);n=n/max(np.linalg.norm(n),1e-9)
        a=np.r_[p,n,2*float(closure)-1]
        if not np.isfinite(a).all() or np.max(np.abs(a))>1.00001:
            raise ValueError(f'Expert action outside explicit task range: {a}')
        return a.astype(np.float32)

    def decode(self, action, fallback_normal=(1.,0.,0.)):
        a=np.asarray(action,dtype=float)
        if a.shape!=(7,) or not np.isfinite(a).all():raise ValueError('Nonfinite or malformed action')
        a=np.clip(a,-1,1);n=a[3:6]
        if np.linalg.norm(n)<1e-6:n=np.asarray(fallback_normal,dtype=float)
        n=n/max(np.linalg.norm(n),1e-9)
        return np.asarray(self.center)+np.asarray(self.scale)*a[:3],n,float((a[6]+1)/2)


class ChunkBuffer:
    def __init__(self,spec=ActionSpec()):
        self.spec=spec;self.reset()

    def reset(self):
        self.actions=None;self.anchor=-np.inf;self.available=-np.inf
        self.accepted=0;self.stale=0

    def submit(self,actions,observed_at,available_at,origin):
        a=np.asarray(actions,dtype=np.float32);origin=np.asarray(origin,dtype=np.float32)
        if a.shape!=(self.spec.horizon,7) or origin.shape!=(7,) or not np.isfinite(a).all() or not np.isfinite(origin).all():
            raise ValueError('Invalid action chunk')
        if not np.isfinite([observed_at,available_at]).all() or available_at<observed_at:
            raise ValueError('Invalid action timestamps')
        if observed_at<=self.anchor or available_at>=observed_at+self.spec.horizon*self.spec.dt:
            self.stale+=1;return False
        self.actions=np.vstack((origin,np.clip(a,-1,1)))
        self.anchor=float(observed_at);self.available=float(available_at);self.accepted+=1
        return True

    def sample(self,t):
        if self.actions is None or t<self.available-1e-9:return None
        x=np.clip((t-self.anchor)/self.spec.dt,0,self.spec.horizon)
        lo=min(int(np.floor(x)),self.spec.horizon);hi=min(lo+1,self.spec.horizon)
        return self.actions[lo]+(x-lo)*(self.actions[hi]-self.actions[lo])


def robot_state(sim):
    ids,m,d=sim.ids,sim.model,sim.data
    bounds=m.jnt_range[[m.joint(f'fr3_joint{i}').id for i in range(1,8)]]
    q=d.qpos[ids.arm_qpos]
    normalized=2*(q-bounds[:,0])/(bounds[:,1]-bounds[:,0])-1
    from .catcher import HAND_CUP,HAND_CLOSED
    span=HAND_CLOSED-HAND_CUP;valid=np.abs(span)>1e-6
    hand=d.qpos[ids.hand_qpos]
    closure=float(np.mean((hand[valid]-HAND_CUP[valid])/span[valid])) if valid.any() else 0.
    return np.r_[np.clip(normalized,-1,1),np.clip(closure,0,1)].astype(np.float32)


def reference_action(sim,spec=ActionSpec()):
    cat=sim.catcher
    p,r,_=cat.ik._fk(cat.ref[0])
    return spec.encode(p,-r[:,1],float(cat.closing))


def current_action(sim,spec=ActionSpec()):
    p=sim.data.site_xpos[sim.ids.palm_site]
    r=sim.data.site_xmat[sim.ids.palm_site].reshape(3,3)
    return spec.encode(p,-r[:,1],robot_state(sim)[-1])
