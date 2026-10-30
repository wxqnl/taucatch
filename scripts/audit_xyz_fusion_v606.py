"""Audit one real-RPC XYZ trial once: all measured fees, causal execution and local IK."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from taucatch.action_h192_195536 import SPEC
from taucatch.action_horizon_data import sha256
from taucatch.arm import ArmIK
from taucatch.differential_servo_v72 import local_step, SERVO_SPEC
from taucatch.multiscale_plane_controller_v13 import PlaneBodyController, PlaneConfig
from taucatch.multiscale_v1 import Forecast
from taucatch.native_close_timing_v44 import DeadlineGripTimer
from taucatch.native_residual_v26 import ResidualChunks, ResidualMixer, RESIDUAL_SPEC
from taucatch.single_ball_v1 import setup
from taucatch.xyz_context_softsign_v273 import context, CONTEXT_CONTRACT
from taucatch.xyz_coarse_contract_v308 import XYZ_CONTRACT
from taucatch.xyz_coarse_execution_v309 import CoarseXYZMixer as ResidualMixer
from taucatch.xyz_fusion_rpc_v602 import unpack
from taucatch.xyz_raw_runtime_v414 import OBSERVATION as OBSERVATION_CONTRACT
from taucatch.xyz_prethrow_v198 import admit_clocks,bill_slot
from taucatch.single_ball_curriculum_v165 import sample_controlled, NAME
from taucatch.xyz_fusion_rpc_v602 import KIND, BEHAVIOR


def equal(a, b):
    assert (a is None) == (b is None)
    if a is not None:
        assert np.array_equal(np.asarray(a, np.float32), np.asarray(b, np.float32))


def delivered(rows, i):
    row = rows[i]
    due = [r for r in rows[:i] if r['available_t'] <= row['t'] + 1e-12]
    live = due[-1] if due and due[-1]['expiry_t'] >= row['t'] - 1e-12 else None
    assert row['delivered_id'] == (None if live is None else live['id'])
    equal(row['returned'], None if live is None else live['generated'])
    assert abs(row['available_t'] - max(row['t'], rows[i-1]['available_t'] if i else -np.inf) - row['compute_ms']/1000) < 1e-10


def audit_executed_peaks(report):
    from taucatch.finder import BallFinder,decode_peaks
    raw_ref=report['finder_numeric']['raw_inputs_outputs']
    peak_ref=report['finder_numeric']['executed_peak_evidence']
    assert sha256(raw_ref['path'])==raw_ref['sha256']
    assert sha256(peak_ref['path'])==peak_ref['sha256']
    raw=np.load(raw_ref['path']);executed=np.load(peak_ref['path'])
    n=len(raw['grids']);assert 16<=n<=32 and n==peak_ref['batches']
    assert executed['peaks'].shape==(n,2,4,3) and executed['counts'].shape==(n,2)
    assert executed['CPU_fallback'].shape==(n,2)
    model=BallFinder().eval()
    model.load_state_dict(torch.load('outputs/models/finder_v2.pt',map_location='cpu',weights_only=True))
    max_raw=max_xy=max_score=max_CPU_xy=0.
    with torch.inference_mode():
        for i,grid in enumerate(raw['grids']):
            cpu=model(torch.from_numpy(grid))
            gpu=(torch.from_numpy(raw['gpu_heat'][i]),torch.from_numpy(raw['gpu_offset'][i]))
            for a,b in zip(cpu,gpu):max_raw=max(max_raw,float((a-b).abs().max()))
            reference=decode_peaks(*gpu,4,.3);cpu_peaks=decode_peaks(*cpu,4,.3)
            for camera,(expected,ref_cpu) in enumerate(zip(reference,cpu_peaks)):
                count=int(executed['counts'][i,camera]);assert count==len(expected)==len(ref_cpu)
                actual=executed['peaks'][i,camera,:count]
                assert np.isnan(executed['peaks'][i,camera,count:]).all()
                if count:
                    assert np.isfinite(actual).all()
                    max_xy=max(max_xy,float(np.abs(expected[:,:2]-actual[:,:2]).max()))
                    max_score=max(max_score,float(np.abs(expected[:,2]-actual[:,2]).max()))
                    max_CPU_xy=max(max_CPU_xy,float(np.abs(ref_cpu[:,:2]-actual[:,:2]).max()))
    assert max_raw<=5e-5 and max_xy<=.01 and max_score<=5e-6 and max_CPU_xy<=.01
    return dict(accepted=True,batches=n,actual_peak_file_sha256=peak_ref['sha256'],
        CPU_CNN_max_raw_delta=max_raw,max_ordered_actual_peak_pixel_delta=max_xy,
        max_ordered_actual_peak_score_delta=max_score,CPU_CNN_max_ordered_peak_pixel_delta=max_CPU_xy,
        CPU_fallback_cameras=int(executed['CPU_fallback'].sum()),no_ball_truth_read=True)


def audit(report_path):
    report = json.loads(report_path.read_text())
    assert report['causal_delivery_revision']=='fused_strict_slot8_v232'
    assert report['complete'] and sha256(report['physical']) == report['physical_sha256']
    assert report['finder_numeric']['accepted'] and 1 <= report['model_calls'] <= 4
    peak_audit=audit_executed_peaks(report)
    assert not report['fresh_on_policy'] and not report['stochastic_xyz_output_executed']
    assert report['mode'] in ['mean','zero']
    campaign=json.loads(Path(report['configuration_path']).read_text())
    initialization=torch.load(campaign['artifact'],map_location='cpu',weights_only=False)
    assert initialization['kind']==KIND and initialization['observation_contract']==OBSERVATION_CONTRACT
    assert sha256(campaign['artifact'])==report['joint_policy_sha256']==report['base_checkpoint_sha256']
    assert initialization['base_initialization_sha256']==sha256(campaign['base_checkpoint'])==report['base_initialization_sha256']
    assert initialization['selected_joint_gradient_steps']==report['selected_joint_gradient_steps']==16
    assert initialization['raw_RL_steps']==report['raw_RL_steps']==1
    assert initialization['supervised_recovery_steps']==report['supervised_recovery_steps']>0
    assert sha256(report['configuration_path'])==report['configuration_sha256']
    scene=campaign['episode_schedule'][report['episode_index']]
    assert scene['index']==report['episode_index'] and scene['seed']==report['seed']
    assert scene['motion_group']==report['motion_group'] and scene['speed_mps']==report['outcome']['speed']
    from taucatch.single_ball_v1 import TRAIN_SEEDS
    assert scene['seed']in TRAIN_SEEDS
    p = json.loads(Path(report['physical']).read_text())
    assert p['xyz_contract'] == XYZ_CONTRACT and p['context_contract'] == CONTEXT_CONTRACT
    assert p['servo_contract'] == SERVO_SPEC.contract() and not p['actor_truth_used']
    cfg = PlaneConfig(**p['fast_config'])
    controller = PlaneBodyController(SPEC, cfg, 'fast')
    for f in p['feedback']:
        fr = f['forecast']
        if fr is not None:
            fr = dict(fr)
            for k in ['positions', 'covariance', 'offsets']:
                fr[k] = np.asarray(fr[k])
            fr = Forecast(**fr)
            assert fr.available_t <= f['t'] + 1e-9
        action, detail = controller.update(f['t'], fr, None, f['measured_action'])
        equal(action, f['command'])
        assert detail == f['detail']

    timer = DeadlineGripTimer(cfg)
    fi, ui, command, expiry = 0, 0, None, -np.inf
    last = np.asarray(p['feedback'][0]['measured_action'], np.float32)
    inner, outer = p['inner_timing_lane'], p['xyz_lane']
    assert len(inner) == len(outer)
    for i, row in enumerate(inner):
        t = row['t']
        while fi < len(p['feedback']) and p['feedback'][fi]['available_t'] <= t + 1e-9:
            f = p['feedback'][fi]
            command, expiry = f['command'], f['available_t'] + .02
            fi += 1
        while ui < len(p['timer_updates']) and p['timer_updates'][ui]['t'] <= t + 1e-9:
            u = p['timer_updates'][ui]
            assert abs(u['available_t'] - max(u['t'], timer.busy_until) - u['compute_ms']/1000) < 1e-10
            timer.pending.append((u['available_t'], u['proposal']))
            timer.busy_until = u['available_t']
            ui += 1
        closed = timer.advance(row['query_t'], timing_sample_t=t)
        assert row['close_timing'] == timer.last_detail
        body = command if command is not None and row['query_t'] <= expiry + 2e-9 else None
        if body is not None:
            last = np.asarray(body, np.float32).copy()
        baseline = None if body is None and not closed else last.copy()
        if baseline is not None:
            baseline[6] = 1. if closed else -1.
        equal(baseline, row['baseline'])
        equal(row['generated'], baseline)
        assert row['target'] is None and not np.any(row['offset'])
        delivered(inner, i)
    assert timer.closed_t == p['timer_closed_t']

    prep = p['preparation']
    warm=prep['frontend_warmup']
    assert warm['tracker_state_unchanged'] and warm['synthetic_inputs'] and not warm['future_flight_inputs'] and not warm['model_actions']
    cr = prep['condition_rpc']
    assert cr['condition_only'] and not any(k in cr for k in ['actions','mean','latent','log_prob'])
    for key in ['input_record','output_record','feature_record']:
        assert sha256(cr[key]) == cr[key+'_sha256']
    rgb, prep_mode, prep_op, prep_episode = unpack(Path(cr['input_record']).read_bytes())
    assert prep_mode == p['mode'] and prep_op == 'condition'
    assert sha256(prep['sensor_path']) == prep['sensor_sha256']
    with np.load(prep['sensor_path'],allow_pickle=False) as z:
        assert all(np.array_equal(z[k],v) for k,v in rgb.items())
    assert warm['completed_ns'] <= rgb['capture_ns']
    assert prep['held_world_equals_release_reset'] and not prep['actor_DVS_or_state_read'] and not prep['actor_ball_truth_read']
    admit_clocks(rgb['capture_ns'],rgb['observed_ns'],cr['encoded_ns'],prep['completed_ns'],
                 prep['release_invoke_ns'],prep['actual_release_ns'])
    assert cr['client_complete_ns'] == prep['completed_ns']
    assert prep['total_preparation_ms'] == (prep['completed_ns']-prep['started_ns'])/1e6
    assert p['early_budgets'] == dict(finder_slot_ms=8.,body_ms=2.,timer_ms=.25)
    from taucatch.xyz_prethrow_strict_v214 import StrictDeliveryQueue as DeliveryQueue
    q = DeliveryQueue()
    for row in p['perception_estimates']:
        assert round(row['t']/.001)%8 == 0
        charge,overrun = bill_slot(row['actual_compute_ms']/1000,.008)
        assert abs(1000*charge-row['compute_ms'])<1e-10 and overrun==row['overrun']
        assert abs(row['padding_ms']-(row['compute_ms']-row['actual_compute_ms']))<1e-10
        fr=row['forecast']
        if fr is not None:
            fr=dict(fr)
            for k in ['positions','covariance','offsets']:fr[k]=np.asarray(fr[k])
            fr=Forecast(**fr)
        assert abs(q.submit(fr,row['t'],charge)-row['available_t'])<1e-10
    for f in p['feedback']:
        fr=q.deliver(f['t'])
        expected=None if fr is None else {k:v.tolist() if isinstance(v,np.ndarray)else v for k,v in vars(fr).items()}
        assert expected == f['forecast']
    for rows,budget in [(p['feedback'],.002),(p['timer_updates'],.00025)]:
        for row in rows:
            charge,overrun=bill_slot(row['actual_compute_ms']/1000,budget)
            assert abs(1000*charge-row['compute_ms'])<1e-10 and overrun==row['overrun']
    requests = p['requests']
    assert 1 <= len(requests) <= 4 and report['model_calls'] == len(requests)
    env = p['controlled_environment']
    assert env['name'] == NAME and p['protocol']['controlled_protocol'] == NAME
    assert env['seed'] == p['seed'] and env['speed_mps'] == p['outcome']['speed']
    expected_env = sample_controlled(p['seed'], env['speed_mps']).throws[0]
    for key in ['p0', 'v0', 'w0', 'target']:
        assert np.array_equal(getattr(expected_env, key), np.asarray(env[key]))
    sensor_inputs = []
    for index, request in enumerate(requests):
        assert request['producer'] == OBSERVATION_CONTRACT and request['observed_t'] >= .02
        assert request['artificial_delay_ms'] == 0.
        rpc = request['rpc']
        assert rpc['observation_contract'] == OBSERVATION_CONTRACT
        assert rpc['kind'] == KIND and rpc['behavior_contract'] == BEHAVIOR
        assert rpc['policy_version'] == report['policy_version']
        assert rpc['raw_RL_steps']==report['raw_RL_steps']
        for provenance_key in ['selected_joint_gradient_steps','attempted_joint_gradient_steps','source_coarse_joint_steps']:
            assert rpc[provenance_key]==initialization[provenance_key]
        assert not rpc['sampling_allowed'] and not rpc['old_behavior_probability_reuse']
        assert rpc['joint_policy_sha256']==report['joint_policy_sha256']
        assert rpc['base_checkpoint_sha256'] == report['base_checkpoint_sha256']
        assert rpc['artifact_sha256'] == report['artifact_sha256']
        assert rpc['xyz_contract'] == XYZ_CONTRACT and rpc['context_contract'] == CONTEXT_CONTRACT
        assert rpc['roundtrip_ms'] >= rpc['server_ms'] and rpc['roundtrip_ms'] == request['roundtrip_ms']
        for key in ['input_record', 'output_record', 'feature_record']:
            assert sha256(rpc[key]) == rpc[key+'_sha256']
        wire_sample, wire_mode, wire_op, episode = unpack(Path(rpc['input_record']).read_bytes())
        assert wire_mode == p['mode'] and wire_op == request['op'] == 'fast'
        assert episode == request['episode'] and not rpc['absolute_output_executed']
        assert request['event_window_end_t'] == request['state_observed_t'] == request['observed_t']
        assert rpc['observed_t'] == request['observed_t']
        assert episode == prep_episode
        assert wire_sample['release_ns'] == prep['release_invoke_ns']
        assert wire_sample['preparation_complete_ns'] == prep['completed_ns']
        assert request['condition_available_t'] == rpc['condition_available_t'] == (prep['completed_ns']-prep['release_invoke_ns'])/1e9
        assert request['anchor_observed_t'] == rpc['anchor_observed_t'] == (rgb['observed_ns']-prep['release_invoke_ns'])/1e9
        assert request['anchor_rgb_t'] == rpc['anchor_rgb_t'] == (rgb['capture_ns']-prep['release_invoke_ns'])/1e9
        assert rpc['condition_record_id'] == cr['id']
        if index:
            assert request['observed_t'] >= requests[index-1]['available_t'] + .002 - 1e-9
        ready = request['perception_ready_t'] + sum(request[k] for k in ['sample_ms','context_ms','roundtrip_ms','transform_ms','enqueue_ms'])/1000
        assert abs(ready - request['available_t']) < 1e-10
        assert request['perception_ready_t'] >= request['observed_t']
        assert request['accepted'] == (ready < request['observed_t'] + .192)
        proof = request['context_proof']
        assert not proof['scoring_sidecar_read'] and not proof['ball_truth_read']
        dynamic = context(proof['t'], proof['fast_action'], proof['fast_available_t'],
                          proof['joint_velocity'], proof['palm'], proof['previous_palm'], proof['previous_t'],
                          proof['measured_t'], proof['forecast_available_t'])
        equal(dynamic, request['context'])
        prior_rows = [r for r in inner if r['t'] < request['observed_t'] - 1e-9]
        assert prior_rows[-1]['delivered_id'] == proof['fast_source_row']
        source = None if proof['fast_source_row'] is None else inner[proof['fast_source_row']]
        equal(proof['fast_action'], None if source is None else source['generated'])
        assert sha256(request['sensor_path']) == report['sensor_sha256'][request['sensor_path']]
        with np.load(request['sensor_path'], allow_pickle=False) as z:
            assert float(z['observed_t']) == request['observed_t']
            assert 'over' not in z and 'mosaic' not in z and 'rgb_t' not in z
            equal(z['context'], dynamic)
            assert str(z['text']) == 'catch the ball'
            assert all(np.array_equal(z[k], wire_sample[k]) for k in wire_sample)
            sensor_inputs.append({k:z[k].copy() for k in ['events','state','context']})
        actions = np.asarray(request['action'], np.float32)
        if p['mode'] == 'sample':
            latent = np.asarray(rpc['latent'],np.float32)
            assert latent.shape == (12,3) and np.isfinite(latent).all() and np.isfinite(rpc['log_prob'])
            expected = latent.clip(-1,1)
        else:
            expected = np.asarray(rpc['mean'],np.float32) if p['mode']=='mean' else np.zeros((12,3),np.float32)
            assert rpc['latent'] is None and rpc['log_prob'] is None
        equal(actions, expected)
        equal(actions, rpc['actions'])
    assert report['execution_xyz_contract'] == XYZ_CONTRACT and not report['fixed_control_bridge'] and not report['training_ready']

    chunks, mixer, installed = ResidualChunks(), ResidualMixer(), 0
    for i, row in enumerate(outer):
        while installed < len(requests) and requests[installed]['observed_t'] <= row['t'] + 1e-9:
            request = requests[installed]
            if request['accepted']:
                actions = np.asarray(request['action'], np.float32)
                assert chunks.enqueue(np.pad(actions, ((0, 0), (0, 4))), request['observed_t'], request['available_t'], RESIDUAL_SPEC.contract())
            installed += 1
        target = chunks.sample(row['t'])
        equal(row['baseline'], inner[i]['returned'])
        assert row['base_timing_delivered_id'] == inner[i]['delivered_id']
        generated, detail = mixer.mix(row['t'], row['baseline'], target, fast_closed=row['fast_closed'])
        equal(row['target'], target)
        equal(row['generated'], generated)
        assert all(row[k] == v for k, v in detail.items())
        assert np.linalg.norm(np.asarray(row['offset'])[:3]) <= XYZ_CONTRACT['translation_m'] + 1e-10 and not np.any(row['offset'][3:])
        if generated is not None:
            equal(generated[3:], np.asarray(row['baseline'])[3:])
            if p['mode'] == 'zero':
                equal(generated, row['baseline'])
        delivered(outer, i)

    contacts = [r['contact_t'] for r in p['truth_scoring_only'] if r['contact_t'] is not None]
    crossings = [r['t'] for r in p['truth_scoring_only'] if r['p'][0] < cfg.plane_x]
    cutoff = min(contacts + [t + .012 for t in crossings[:1]] + [float('inf')])
    sim, _ = setup(p['seed'])
    ik = ArmIK(sim.model, sim.ids, sim.home_q)
    limit = np.asarray(p['servo_velocity_limit']) * .001
    bounds = np.asarray(p['arm_ctrlrange'])
    commands_cm, conditional_reference_cm, actual = [], [], []
    by_tick = {round(r['t'], 9): r for r in outer}
    for row in p['servo']:
        lane = by_tick[round(row['t'], 9)]
        assert lane['delivered_id'] == row['delivered_xyz_lane_id']
        source = outer[lane['delivered_id']]
        point, normal, _ = SPEC.decode(source['generated'])
        equal(point, row['command'])
        equal(normal, row['normal'])
        qref, _ = local_step(ik, row['q0'], row['command'], row['normal'], limit)
        assert np.array_equal(qref, np.asarray(row['qref']))
        assert row['hand_mapping_error'] == 0 and row['max_velocity_fraction'] <= 1 + 1e-10
        assert np.all(np.asarray(row['arm_ctrl']) >= bounds[:, 0]) and np.all(np.asarray(row['arm_ctrl']) <= bounds[:, 1])
        if row['t'] <= cutoff + 1e-9:
            baseline_point, baseline_normal, _ = SPEC.decode(source['baseline'])
            change = float(100 * np.linalg.norm(point - baseline_point))
            if change > 1e-5:
                request = next(r for r in requests if r['observed_t'] == source['residual_anchor'])
                assert request['available_t'] <= source['t'] + 1e-9
                zero_qref, _ = local_step(ik, row['q0'], baseline_point, baseline_normal, limit)
                delta = float(100 * np.linalg.norm(ik._fk(qref)[0] - ik._fk(zero_qref)[0]))
                commands_cm.append(change)
                conditional_reference_cm.append(delta)
                actual.append(dict(t=row['t'], delivered_xyz_lane_id=source['id'],
                                   source_observed_t=request['observed_t'], source_op=request['op'],
                                   command_delta_cm=change, conditional_reference_delta_cm=delta))
    if p['mode'] == 'zero':
        assert not actual
    fast_delivered = sorted({outer[row['delivered_xyz_lane_id']]['residual_anchor'] for row in p['servo']
                             if row['t'] <= cutoff + 1e-9 and row['delivered_xyz_lane_id'] is not None
                             and outer[row['delivered_xyz_lane_id']]['residual_anchor'] in [r['observed_t'] for r in requests]})
    import hashlib
    diversity = {k:len({hashlib.sha256(s[k].tobytes()).hexdigest() for s in sensor_inputs}) for k in ['events','state','context']}
    maximum = max(commands_cm, default=0.)
    reference = max(conditional_reference_cm, default=0.)
    return dict(accepted=True, executed_peak_independent_audit=peak_audit, report=str(report_path), report_sha256=sha256(report_path), mode=p['mode'],
                servo_steps_exact=len(p['servo']), fast_commands_exact=len(p['feedback']),
                actual_xyz_steps_before_contact=len(actual), max_command_delta_cm=maximum,
                max_conditional_IK_reference_delta_cm=reference,
                meaningful_authority=len(actual) >= 10 and maximum >= .5 and reference >= .01,
                source_credit=actual, cutoff_t=cutoff, fast_normal_closure_bit_exact=True,
                context_causal_reconstructed=True, synthetic_delay_ms=0., model_calls=len(requests),
                observation_contract=OBSERVATION_CONTRACT, controlled_environment_exact=True, prerelease_clock_and_RGB_proof=True, common_budget_costs_reconstructed=True,
                perception_overruns=sum(x['overrun']for x in p['perception_estimates']),
                body_overruns=sum(x['overrun']for x in p['feedback']),
                early_perception_overruns=sum(x['overrun']for x in p['perception_estimates']if x['t']<requests[0]['available_t']),
                input_diversity=diversity, fast_packets_delivered_before_contact=fast_delivered,
                fast_actual_nonzero_steps_before_contact=sum(r['source_op']=='fast' for r in actual),
                optimizer_steps=0, VLA_gain_proven=False,
                scope='New slow-anchor/fresh-input ABI with actual measured delivery and independent context/multi-chunk/servo/IK replay; adapted native/XYZ candidate selected16 joint steps; no optimizer here and no VLA gain claim.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists()
    result = audit(args.report)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'source_credit'}), flush=True)
