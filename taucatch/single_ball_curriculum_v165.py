"""TRAIN-only speed curriculum retaining a seed's launch/aim/spin geometry.

Environment generation metadata never enters the policy packet. No old seed is
reinterpreted: controlled_speed_mps and this new protocol must be recorded.
"""
import numpy as np
from .single_ball_v1 import PROTOCOL
from .volley import sample_volley,VolleySim

NAME='single_ball_fixed_geometry_speed_curriculum_v165'


def sample_controlled(seed,speed=None):
    rng=np.random.default_rng(seed)
    nominal=float(rng.uniform(*PROTOCOL['speed_mps']))
    spin=float(rng.uniform(*PROTOCOL['spin_rps']))
    chosen=nominal if speed is None else float(speed)
    if not np.isfinite(chosen) or not PROTOCOL['speed_mps'][0]<=chosen<=PROTOCOL['speed_mps'][1]:
        raise ValueError('Curriculum speed must stay in authorized12–18m/s task range')
    volley=sample_volley(rng,ball='tennis',n=1,speed=chosen,spin_rps=spin,
        max_offset=0.,window_half=PROTOCOL['window_half_m'])
    volley.colors=['yellow']
    assert volley.target==0 and len(volley.throws)==1 and volley.release[0]==0
    return volley


def setup_controlled(seed,speed):
    volley=sample_controlled(seed,speed)
    return VolleySim('tennis',n_balls=1,arm='agile',thrower='launcher'),volley
