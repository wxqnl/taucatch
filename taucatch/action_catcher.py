"""Low-level execution of VLA palm waypoints. No target selection or intercept planner."""
import mujoco
import numpy as np
from .action_chunk import ActionSpec,ChunkBuffer
from .catcher import Catcher
from .arm import ARM_QDD_MAX,ARM_QDDD_MAX
from .scene import ARM_PROFILES


class ActionCatcher(Catcher):
    def __init__(self,*args,spec=ActionSpec(),**kwargs):
        super().__init__(*args,**kwargs)
        self.spec=spec;self.buffer=ChunkBuffer(spec)
        profile=ARM_PROFILES[kwargs.get('arm','agile')]
        self.acc_max=profile['acc']*ARM_QDD_MAX
        self.jerk_max=profile['acc']*ARM_QDDD_MAX

    def reset(self,data):
        super().reset(data);self.buffer.reset();self.ik_rejected=0
        self.normal=-data.site_xmat[self.ids.palm_site].reshape(3,3)[:,1].copy()

    def plan(self,*args,**kwargs):raise RuntimeError('Action execution cannot use the analytic ball planner')
    def plan_path(self,*args,**kwargs):raise RuntimeError('Action execution cannot use a selected ball path')

    def step(self,data,t,unused_ball_state=None):
        ids,m,cfg=self.ids,self.m,self.cfg
        action=self.buffer.sample(t+self.dt)
        closure=float(self.closing)
        if action is not None:
            p,normal,closure=self.spec.decode(action,self.normal)
            target,perr,rerr,_=self.ik.solve_normal(self.ref[0],p,normal,iters=20)
            # A cross-product residual is also zero for the opposite normal.
            solved_normal=-self.ik.d.site_xmat[ids.palm_site].reshape(3,3)[:,1]
            aligned=float(solved_normal@normal)>np.cos(.2)
            if perr<.01 and rerr<.2 and aligned:
                self.normal=normal
                if self.first_cmd_t is None:self.first_cmd_t=t
            else:
                self.ik_rejected+=1;target=self.ref[0]
        else:target=self.ref[0]
        q0,v0,a0=self.ref
        # Position servo with a velocity-bounded reference. The physical torque limit remains active.
        # Re-integrating an acceleration-limited trajectory here caused avoidable lag in dense waypoint replay.
        qref=q0+np.clip(target-q0,-self.planner.qd_max*self.dt,self.planner.qd_max*self.dt)
        qref=np.clip(qref,self.ik.jr[:,0]+1e-3,self.ik.jr[:,1]-1e-3)
        vref=(qref-q0)/self.dt
        acceleration=np.clip((vref-v0)/self.dt,-self.acc_max,self.acc_max)
        self.ref=qref,vref,acceleration
        mujoco.mj_fullM(m,data,self.mfull)
        mass=self.mfull[np.ix_(ids.arm_dof,ids.arm_dof)]
        q,qd=data.qpos[ids.arm_qpos],data.qvel[ids.arm_dof]
        tau=mass@(acceleration+cfg.kp*(qref-q)+cfg.kd*(vref-qd))+data.qfrc_bias[ids.arm_dof]
        lo,hi=m.actuator_ctrlrange[ids.arm_act].T
        self.saturated=bool(np.any((tau<lo)|(tau>hi)))
        data.ctrl[ids.arm_act]=np.clip(tau,lo,hi)
        data.ctrl[ids.hand_act]=self.cup+closure*(self.closed-self.cup)
        self.closing=closure>=.5
