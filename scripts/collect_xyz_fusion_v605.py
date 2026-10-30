"""Actual prerelease RGB/text preparation followed only by fresh flight inputs."""
import argparse
from collections import deque
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from collect_single_ball_v1 import save
from taucatch.action_chunk import current_action
from taucatch.action_h192_195536 import SPEC
from taucatch.action_horizon_data import sha256
from taucatch.differential_servo_v72 import DifferentialServoMixin, SERVO_SPEC
from taucatch.multiscale_catcher_v1 import MultiscaleCatcher
from taucatch.fused_observation_v232 import FusedObservation as GraphObservation
from taucatch.xyz_prethrow_v198 import capture_condition,bill_slot
from taucatch.multiscale_plane_controller_v13 import PlaneBodyController, PlaneConfig
from taucatch.native_close_timing_execution_v44 import FinalTimingBuffer
from taucatch.native_close_timing_v44 import DeadlineGripTimer
from taucatch.single_ball_v1 import INSTRUCTION, PROTOCOL
from taucatch.single_ball_curriculum_v165 import setup_controlled,NAME
from taucatch.xyz_context_softsign_v273 import context, CONTEXT_CONTRACT
from taucatch.xyz_fusion_rpc_v602 import PrethrowClient as TimescaleClient
from taucatch.xyz_raw_runtime_v414 import OBSERVATION as OBSERVATION_CONTRACT
from taucatch.xyz_fast_observation_v176 import fast_sample
from taucatch.xyz_coarse_execution_v309 import CoarseXYZFinalBuffer as XYZFinalBuffer, XYZ_CONTRACT


class XYZCatcher(DifferentialServoMixin, MultiscaleCatcher):
    def __init__(self, *args, timer, **kwargs):
        super().__init__(*args, mode='fast', **kwargs)
        self.buffer = XYZFinalBuffer(FinalTimingBuffer(self.buffer, timer, self.dt), self.dt)

    def reset(self, data):
        super().reset(data)
        normal = -data.site_xmat[self.ids.palm_site].reshape(3, 3)[:, 1]
        self.buffer.base.base.last_action = self.spec.encode(data.site_xpos[self.ids.palm_site], normal, 0.)

    def step(self, data, t, unused_ball_state=None):
        super().step(data, t, unused_ball_state)
        if self.servo_evidence and abs(self.servo_evidence[-1]['t'] - t) < 1e-12:
            self.servo_evidence[-1].update(delivered_xyz_lane_id=self.buffer.rows[-1]['delivered_id'],
                                          actual_palm=data.site_xpos[self.ids.palm_site].tolist())


def forecast_record(value):
    if value is None:
        return None
    result = asdict(value)
    for key in ['positions', 'covariance', 'offsets']:
        result[key] = result[key].tolist()
    return result


def run(mode, root, client, seed, controlled_speed):
    sim, volley = setup_controlled(seed,controlled_speed)
    cfg = PlaneConfig(period=.004)
    timer = DeadlineGripTimer(cfg)
    sim.catcher = XYZCatcher(sim.model, sim.ids, sim.ball, sim.home_q, arm='agile', spec=SPEC, timer=timer)
    sim.reset(volley)
    obs = GraphObservation(sim, seed, cfg.period)
    obs.numeric_root = root / 'finder_raw_checks'
    obs.reset()
    controller = PlaneBodyController(SPEC, cfg, 'fast')
    commands = deque()
    last_timer, busy, next_fast = object(), -np.inf, 0.
    last_pose, last_pose_t, prior_pose, prior_t = None, None, None, None
    feedback, updates, requests, truth = [], [], [], []
    next_model, anchor_ready, anchor_rgb_t = .020, None, None
    episode = str(root)

    def control(s, t):
        nonlocal last_timer, busy, next_fast, next_model, anchor_ready, anchor_rgb_t, last_pose, last_pose_t, prior_pose, prior_t
        while commands and commands[0][0] <= t + 1e-9:
            ready, action = commands.popleft()
            s.catcher.buffer.set_command(action, ready, ready + .02)
        estimate = obs.queue.deliver(t)
        if estimate is not last_timer and t + 1e-9 >= timer.busy_until:
            started = time.perf_counter()
            proposal = timer.propose(estimate, t)
            prior = timer.busy_until
            timer.submit(proposal, t, 0.)
            elapsed = time.perf_counter() - started
            charged, overrun = bill_slot(elapsed, .00025)
            ready = max(t, prior) + charged
            timer.pending[-1] = (ready, proposal)
            timer.busy_until = ready
            updates.append(dict(t=t, available_t=ready, compute_ms=1000 * charged, actual_compute_ms=1000*elapsed, padding_ms=1000*(charged-elapsed), overrun=overrun, proposal=proposal))
            last_timer = estimate
        if t + 1e-9 >= next_fast:
            measured = current_action(s, SPEC)
            started = time.perf_counter()
            action, detail = controller.update(t, estimate, s.catcher.buffer.native, measured)
            elapsed = time.perf_counter() - started
            charged, overrun = bill_slot(elapsed, .002)
            ready = max(t, busy) + charged
            busy = ready
            commands.append((ready, action))
            next_fast = t + cfg.period
            feedback.append(dict(id=len(feedback), t=t, available_t=ready, compute_ms=1000 * charged, actual_compute_ms=1000*elapsed, padding_ms=1000*(charged-elapsed), overrun=overrun,
                                 measured_action=measured.tolist(), forecast=forecast_record(estimate),
                                 command=None if action is None else action.tolist(), detail=detail))
            prior_pose, prior_t = last_pose, last_pose_t
            last_pose, last_pose_t = SPEC.decode(measured)[0], t
        if len(requests) < 4 and next_model - 1e-9 <= t <= .180 + 1e-9 and estimate is not None:
            op = 'fast'
            started = time.perf_counter()
            sample = fast_sample(obs, t)
            sample_ms = 1000 * (time.perf_counter() - started)
            if sample is not None:
                started = time.perf_counter()
                inner = s.catcher.buffer.base
                delivered = inner.rows[-1]['delivered_id'] if inner.rows else None
                base = None if delivered is None else inner.rows[delivered]['generated']
                base_ready = None if delivered is None else inner.rows[delivered]['available_t']
                previous, previous_t = (last_pose, last_pose_t) if last_pose_t < t else (prior_pose, prior_t)
                palm = s.data.site_xpos[s.ids.palm_site].copy()
                qvel = s.data.qvel[s.ids.arm_dof].copy()
                dynamic = context(t, base, base_ready, qvel, palm, previous, previous_t,
                                  estimate.measured_t, estimate.available_t)
                context_ms = 1000 * (time.perf_counter() - started)
                sample.update(release_ns=release_ns, preparation_complete_ns=preparation_complete_ns)
                reply = client.predict(dict(sample, context=dynamic), op, episode)
                assert reply['xyz_contract'] == XYZ_CONTRACT and reply['mode'] == mode
                assert reply['observation_contract'] == OBSERVATION_CONTRACT
                started = time.perf_counter()
                action = np.asarray(reply['actions'], np.float32).copy()
                for key in ['input_record', 'output_record', 'feature_record']:
                    assert sha256(reply[key]) == reply[key+'_sha256']
                transform_ms = 1000 * (time.perf_counter() - started)
                path = root / ('sensor_%02d.npz' % len(requests))
                # Evidence serialization is offline instrumentation, not producer inference.
                perception_ready = max(t, obs.queue.busy_until)
                ready = perception_ready + (sample_ms + context_ms + reply['roundtrip_ms'] + transform_ms) / 1000
                accepted, available, enqueue_ms = s.catcher.buffer.enqueue_billed_xyz(action, t, ready)
                proof = dict(t=t, fast_action=base, fast_available_t=base_ready,
                             fast_source_row=delivered, joint_velocity=qvel.tolist(), palm=palm.tolist(),
                             previous_palm=previous.tolist(), previous_t=previous_t,
                             measured_t=estimate.measured_t, forecast_available_t=estimate.available_t,
                             robot_source='live_proprioception', scoring_sidecar_read=False, ball_truth_read=False)
                prior_anchor_ready = anchor_ready
                requests.append(dict(op=op, episode=episode, condition_available_t=prior_anchor_ready,
                                     anchor_observed_t=reply['anchor_observed_t'],
                                     anchor_rgb_t=anchor_rgb_t, event_window_end_t=t, state_observed_t=t,
                                     observed_t=t, available_t=available, accepted=accepted,
                                     perception_ready_t=perception_ready, sample_ms=sample_ms,
                                     context_ms=context_ms, transform_ms=transform_ms, roundtrip_ms=reply['roundtrip_ms'],
                                     enqueue_ms=enqueue_ms, artificial_delay_ms=0., rpc=reply,
                                     action=action.tolist(), context=dynamic.tolist(), context_proof=proof,
                                     sensor_path=str(path), producer=OBSERVATION_CONTRACT))
                np.savez_compressed(path, **sample, context=dynamic, text=INSTRUCTION)
                next_model = max(t + .016, available + .002)
        return None

    def frame(s, t):
        obs.step(t)
        p, v, _ = s.ball_state(0)
        truth.append(dict(t=t, p=p.tolist(), v=v.tolist(), contact_t=s.catcher.contact_t))

    attempt = root / 'physical_attempt.json'
    prep_started_ns = time.perf_counter_ns()
    frontend_warmup = obs.prepare_frontend()
    text_reply = client.control('prepare', INSTRUCTION)
    qpos, qvel = sim.data.qpos.copy(), sim.data.qvel.copy()
    condition_sample = capture_condition(sim, obs.rgb)
    np.savez_compressed(root/'condition_sensor.npz', **condition_sample)
    condition_reply = client.predict(condition_sample, 'condition', episode)
    preparation_complete_ns = condition_reply['client_complete_ns']
    release_ns = time.perf_counter_ns()
    anchor_ready = (preparation_complete_ns-release_ns)/1e9
    anchor_rgb_t = (condition_sample['capture_ns']-release_ns)/1e9
    preparation = dict(frontend_warmup=frontend_warmup, started_ns=prep_started_ns, text_roundtrip_ms=text_reply['roundtrip_ms'],
        condition_rpc=condition_reply, sensor_path=str(root/'condition_sensor.npz'),
        sensor_sha256=sha256(root/'condition_sensor.npz'), completed_ns=preparation_complete_ns,
        release_invoke_ns=release_ns, total_preparation_ms=(preparation_complete_ns-prep_started_ns)/1e6,
        actor_DVS_or_state_read=False, actor_ball_truth_read=False)
    original_reset, original_release = sim.reset, sim._release_ball
    def checked_reset(v):
        original_reset(v)
        assert np.array_equal(qpos,sim.data.qpos) and np.array_equal(qvel,sim.data.qvel)
        preparation['held_world_equals_release_reset'] = True
    def stamped_release(i):
        preparation['actual_release_ns'] = time.perf_counter_ns()
        original_release(i)
    sim.reset, sim._release_ball = checked_reset, stamped_release
    try:
        outcome = sim.run(volley, 0., hold=.5, state_fn=control, frame_cb=frame)
        physical = dict(complete=True, preparation=preparation, perception_estimates=obs.estimates, perception_timings=obs.timings, early_budgets=dict(finder_slot_ms=8.,body_ms=2.,timer_ms=.25), seed=seed, mode=mode, outcome=outcome, fast_config=asdict(cfg),
                        protocol=dict(PROTOCOL,controlled_protocol=NAME,controlled_speed_mps=controlled_speed), controlled_environment=dict(name=NAME,seed=seed,speed_mps=controlled_speed,p0=volley.throws[0].p0.tolist(),v0=volley.throws[0].v0.tolist(),w0=volley.throws[0].w0.tolist(),target=volley.throws[0].target.tolist()), feedback=feedback, timer_updates=updates, requests=requests,
                        inner_timing_lane=sim.catcher.buffer.base.rows, xyz_lane=sim.catcher.buffer.rows,
                        servo=sim.catcher.servo_evidence, installations=sim.catcher.buffer.residual.accepted,
                        truth_scoring_only=truth, timer_closed_t=timer.closed_t,
                        servo_velocity_limit=sim.catcher.planner.qd_max.tolist(),
                        arm_ctrlrange=sim.model.actuator_ctrlrange[sim.ids.arm_act].tolist(),
                        xyz_contract=XYZ_CONTRACT, context_contract=CONTEXT_CONTRACT,
                        servo_contract=SERVO_SPEC.contract(), model_calls=len(requests), optimizer_steps=0,
                        actor_truth_used=False, gain_proven=False)
        save(attempt, physical)
    except BaseException as exc:
        save(root/'partial_attempt.json',dict(complete=False,error=repr(exc),preparation=preparation,
            feedback=feedback,timer_updates=updates,requests=requests,truth_scoring_only=truth,
            perception_estimates=obs.estimates,xyz_lane=sim.catcher.buffer.rows,
            inner_timing_lane=sim.catcher.buffer.base.rows,servo=sim.catcher.servo_evidence,
            state_resume_supported=False,automatic_retry=False))
        raise
    finally:
        import sys
        primary_active = sys.exc_info()[0] is not None
        try:obs.close()
        except BaseException as exc:
            save(root/'close_error.json',dict(error=repr(exc),primary_error_preserved=primary_active))
            if not primary_active:raise
    assert 1 <= len(requests) <= 4
    return dict(complete=True, seed=seed, mode=mode, outcome=outcome, physical=str(attempt),
                physical_sha256=sha256(attempt), finder_numeric=obs.numeric_report,
                sensor_sha256={r['sensor_path']: sha256(r['sensor_path']) for r in requests},
                observation_contract=OBSERVATION_CONTRACT, model_calls=len(requests), optimizer_steps=0,
                artificial_delay_ms=0., VLA_gain_proven=False,
                causal_delivery_revision='fused_strict_slot8_v232', scope='Actual prerelease RGB/text preparation then fresh flight DVS/state/context only; shared8ms finder slots with guarded fused GPU peaks, original CPU fallback, CPUgrid prewarm/2ms body/.25ms timer budgets never undercharge measured work; initializer only, no new training/gain claim.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['mean','zero'], required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--configuration', type=Path, required=True)
    parser.add_argument('--episode-index',type=int,required=True)
    args = parser.parse_args()
    assert not args.out.exists()
    torch.set_num_threads(1)
    folder = args.out.with_suffix('')
    folder.mkdir(exist_ok=False)
    config = json.loads(args.configuration.read_text())
    gate = json.loads(Path(config['feature_gate']).read_text())
    client = TimescaleClient(('127.0.0.1', config['port']), sha256(config['artifact']), sha256(config['artifact']), args.mode)
    scene=config['episode_schedule'][args.episode_index]
    assert scene['index']==args.episode_index
    try:result = run(args.mode, folder, client, int(scene['seed']), float(scene['speed_mps']))
    finally:client.close()
    result.update(configuration_path=str(args.configuration),episode_index=args.episode_index,motion_group=scene['motion_group'],draw_id=args.out.stem, configuration_sha256=sha256(args.configuration), execution_xyz_contract=XYZ_CONTRACT, fixed_control_bridge=False, training_ready=False, fresh_on_policy=args.mode=='sample', policy_version=config['policy_version'], base_checkpoint_sha256=sha256(config['artifact']), joint_policy_sha256=sha256(config['artifact']), base_initialization_sha256=sha256(config['base_checkpoint']), selected_joint_gradient_steps=16, raw_RL_steps=config['raw_RL_steps'],supervised_recovery_steps=config['supervised_recovery_steps'], artifact_sha256=sha256(config['artifact']), runtime_options_sha256=gate['runtime_options_sha256'], feature_gate=config['feature_gate'], learned_xyz_output_executed=args.mode=='mean', stochastic_xyz_output_executed=args.mode=='sample')
    save(args.out, result)
    print(json.dumps(result), flush=True)
