"""One toss: hold, release, control at 500 Hz, policy queries every 20 ms, catch check at the end."""
import numpy as np

from .sim import CatchSim

PHYSICS_DT = 0.001
CONTROL_EVERY = 2          # 500 Hz control
STEP = 0.020               # policy step and action-chunk spacing
AFTER = 0.45               # time after the planned arrival before the catch is checked


def run_expert(sim, expert, toss, on_tick=None):
    """Expert rollout. `on_tick(t, released, target, rot, close)` sees every control tick."""
    sim.reset_arm()
    sim.paint(toss)
    expert.reset()
    end = toss.hold + toss.flight + AFTER
    n = int(round(end / PHYSICS_DT))
    target, rot, close = sim.ready_grasp, sim.ready_rot, False
    release = int(round(toss.hold / PHYSICS_DT))
    for k in range(n):
        t = k * PHYSICS_DT
        released = k >= release
        if k < release:
            sim.place_ball(toss.launch)
        elif k == release:
            sim.place_ball(toss.launch, toss.velocity)
        if toss.others:
            sim.move_distractors(toss, t)
        if k % CONTROL_EVERY == 0:
            target, rot, close = expert.act(released, t)
            sim.track(target, rot, CONTROL_EVERY * PHYSICS_DT)
            sim.set_finger(close)
            if on_tick is not None:
                on_tick(t, released, target, rot, close)
        sim.step_physics()
    return sim.caught()
