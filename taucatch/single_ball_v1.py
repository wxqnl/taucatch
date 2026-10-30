"""Single tennis-ball task. Scene facts are labels/scoring, never actor inputs."""
import numpy as np

from .volley import VolleySim, sample_volley

INSTRUCTION = 'catch the ball'
TRAIN_SEEDS = tuple(range(1100000, 1100128))
VAL_SEEDS = tuple(range(1110000, 1110032))
CONFIRMATION_SEEDS = tuple(range(1120000, 1120128))
PROTOCOL = dict(ball='tennis', balls=1, thrower='launcher', speed_mps=[12., 18.],
                spin_rps=[0., 5.], window_half_m=.2, hold_seconds=.5,
                instruction=INSTRUCTION, label='single_ball_native_h96_v1')


def setup(seed):
    rng = np.random.default_rng(seed)
    speed = float(rng.uniform(*PROTOCOL['speed_mps']))
    spin = float(rng.uniform(*PROTOCOL['spin_rps']))
    volley = sample_volley(rng, ball='tennis', n=1, speed=speed, spin_rps=spin,
                           max_offset=0., window_half=PROTOCOL['window_half_m'])
    volley.colors = ['yellow']
    assert volley.target == 0 and len(volley.throws) == 1 and volley.release[0] == 0
    return VolleySim('tennis', n_balls=1, arm='agile', thrower='launcher'), volley
