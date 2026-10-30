"""Actual wall-clock prerelease image preparation; no future action or DVS."""
import time
import numpy as np

from .choice import views

OBSERVATION = 'prerelease_RGB_text_then_fresh_DVS_state_context_v198'


def capture_condition(sim, camera):
    """Render a held, stationary prerelease world over a real exposure/readout."""
    start = time.perf_counter_ns()
    images = []
    for i in range(3):
        camera.renderer.update_scene(sim.data, camera=camera.cfg.name)
        images.append(camera.renderer.render().astype(np.float32))
        if i < 2:
            time.sleep(.001)
    captured = time.perf_counter_ns()
    time.sleep(camera.cfg.latency)
    over, mosaic = views(np.mean(images, 0).astype(np.uint8), [])
    return dict(over=over, mosaic=mosaic, capture_ns=captured,
                observed_ns=time.perf_counter_ns(), exposure_start_ns=start,
                readout_ns=round(camera.cfg.latency*1e9))


def admit_clocks(capture_ns, observed_ns, encoded_ns, complete_ns, release_ns, now_ns):
    values = [capture_ns, observed_ns, encoded_ns, complete_ns, release_ns, now_ns]
    if any(not isinstance(x, (int, np.integer)) or x <= 0 for x in values):
        raise ValueError('Actual positive monotonic nanoseconds required')
    if not capture_ns < observed_ns <= encoded_ns <= complete_ns <= release_ns <= now_ns:
        raise ValueError('Condition not completed before release invocation')


def bill_slot(actual_seconds, budget_seconds):
    if not np.isfinite(actual_seconds) or actual_seconds < 0 or budget_seconds <= 0:
        raise ValueError('Invalid measured compute/budget')
    return max(actual_seconds, budget_seconds), bool(actual_seconds > budget_seconds)
