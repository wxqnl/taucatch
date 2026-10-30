import json
from dataclasses import asdict

import numpy as np
import unittest
import tempfile
from pathlib import Path

from taucatch.action_chunk import ActionSpec,ChunkBuffer


def make_demo(folder,seed,split,target=0):
    n=2;over=np.zeros((n,224,224,3),np.uint8)
    np.savez(folder/f'{seed}_0.npz',over=over,mosaic=over,state=np.zeros((n,8),np.float32),
             events=np.zeros((n,2,6,60,80),np.float16),actions=np.zeros((n,12,7),np.float32),
             valid=np.ones((n,12),bool),observed_t=[.008,.016],rgb_t=[.005,.013],
             text='catch red',seed=seed,split=split,scoring_target=target,
             outcome=json.dumps(dict(caught=True)),action_spec=json.dumps(asdict(ActionSpec())))


class ActionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.tmp_path=Path(self.temp.name)

    def test_codec_has_metric_units_normal_and_fractional_grip(self):
        spec=ActionSpec();p=np.array([.72,-.2,1.3]);n=np.array([2.,1.,-1.])
        a=spec.encode(p,n,.3);actual,normal,closure=spec.decode(a)
        np.testing.assert_allclose(actual,p,atol=1e-7)
        np.testing.assert_allclose(normal,n/np.linalg.norm(n),atol=1e-7)
        self.assertAlmostEqual(closure,.3,places=6)
        with self.assertRaises(ValueError):spec.encode([99,0,0],n,0)
        with self.assertRaises(ValueError):spec.decode(np.full(7,np.nan))


    def test_late_chunk_uses_original_observation_timeline(self):
        b=ChunkBuffer();actions=np.zeros((12,7),np.float32)
        actions[:,0]=np.arange(1,13)/12
        assert b.submit(actions,1.,1.010,np.zeros(7))
        assert b.sample(1.009) is None
        # At 10 ms, waypoint index is 2.5, rather than re-starting the first waypoint.
        self.assertAlmostEqual(float(b.sample(1.010)[0]),2.5/12,places=6)
        self.assertAlmostEqual(float(b.sample(1.012)[0]),3/12,places=6)
        self.assertAlmostEqual(float(b.sample(1.2)[0]),1.,places=6)


    def test_expired_or_out_of_order_reply_does_not_replace_current_chunk(self):
        b=ChunkBuffer();a=np.zeros((12,7));origin=np.zeros(7)
        assert b.submit(a,1.,1.001,origin)
        assert not b.submit(a+.5,.99,1.002,origin)
        assert not b.submit(a+.5,1.01,1.1,origin)
        np.testing.assert_array_equal(b.sample(1.02),np.zeros(7));assert b.stale==2
        with self.assertRaises(ValueError):b.submit(a,2,1,origin)
        with self.assertRaises(ValueError):b.submit(a[:11],2,2,origin)


    def test_data_whitelists_inputs_and_requires_complete_manifest(self):
        tmp_path=self.tmp_path
        from taucatch.action_data import ActionData
        (tmp_path/'manifest.json').write_text(json.dumps(dict(complete=False)))
        with self.assertRaisesRegex(ValueError,'not complete'):ActionData(tmp_path,'train')
        (tmp_path/'manifest.json').write_text(json.dumps(dict(complete=True)))
        make_demo(tmp_path,900000,'train',target=127)
        x,y,valid=ActionData(tmp_path,'train').batch([0])
        assert set(x)=={'over','mosaic','state','events','text'}
        assert y.shape==(1,12,7);assert valid.all()
        # Privileged scoring target is deliberately not usable by the policy.
        assert 'scoring_target' not in x


    def test_whole_seed_split_leak_is_rejected(self):
        tmp_path=self.tmp_path
        from taucatch.action_data import load_action_data
        (tmp_path/'manifest.json').write_text(json.dumps(dict(complete=True)))
        make_demo(tmp_path,900000,'train')
        make_demo(tmp_path,900001,'val')
        path=tmp_path/'900001_0.npz'
        with np.load(path,allow_pickle=False) as z:rows={k:z[k].copy() for k in z.files}
        rows['seed']=900000;np.savez(path,**rows)
        with self.assertRaisesRegex(ValueError,'seed leakage'):load_action_data(tmp_path)


    def test_action_servo_does_not_accept_analytic_plans(self):
        from taucatch.action_catcher import ActionCatcher
        # Calls must fail even before construction; no planner should be consulted.
        with self.assertRaises(RuntimeError):ActionCatcher.plan(None,np.zeros(3))
        with self.assertRaises(RuntimeError):ActionCatcher.plan_path(None,np.zeros((10,3)))

    def test_action_rpc_carries_sensor_inputs_without_scoring_metadata(self):
        from taucatch.action_link import pack_action_input,unpack_action_input
        sample=dict(over=np.zeros((224,224,3),np.uint8),mosaic=np.zeros((224,224,3),np.uint8),
                    state=np.arange(8,dtype=np.float32),events=np.ones((2,6,60,80),np.float16),
                    scoring_target=14,true_ball_state=np.ones(9))
        text,actual=unpack_action_input(pack_action_input('catch red',sample))
        self.assertEqual(text,'catch red')
        self.assertEqual(set(actual),{'over','mosaic','state','events'})
        np.testing.assert_array_equal(actual['events'],sample['events'])

    def test_masked_padding_cannot_change_action_loss(self):
        import torch
        from scripts.train_action_policy import loss_fn
        prediction=torch.zeros(2,12,7);target=torch.zeros_like(prediction)
        valid=torch.ones(2,12,dtype=torch.bool);valid[:,-2:]=False
        first=loss_fn(prediction,target,valid)
        target[:,-2:]=100
        self.assertEqual(float(first),float(loss_fn(prediction,target,valid)))

    def test_raw_dvs_position_survives_within_one_pooled_patch(self):
        import torch
        from taucatch.action_policy import EventTokens
        torch.manual_seed(2);encoder=EventTokens(16)
        with torch.no_grad():encoder.project.weight.zero_();encoder.project.bias.zero_()
        a=torch.zeros(1,2,6,60,80);b=a.clone()
        # Same mass and coarse patch, distinct locations: convolution branch is disabled.
        a[0,0,0,3,2]=1;b[0,0,0,3,6]=1
        with torch.no_grad():
            one=encoder(a);two=encoder(b);empty=encoder(torch.zeros_like(a))
        self.assertGreater(float((one-two).abs().max()),1e-4)
        self.assertTrue(torch.isfinite(empty).all())

    def test_opposite_palm_normal_is_not_accepted_as_zero_rotation_error(self):
        from taucatch.volley import VolleySim,sample_volley
        from taucatch.action_catcher import ActionCatcher
        sim=VolleySim(n_balls=2,arm='agile');volley=sample_volley(np.random.default_rng(700000))
        sim.catcher=ActionCatcher(sim.model,sim.ids,sim.ball,sim.home_q,arm='agile');sim.reset(volley)
        before=sim.catcher.normal.copy();p=sim.data.site_xpos[sim.ids.palm_site].copy()
        # The shared IK uses a cross-product error, which is zero at antipodal normals.
        action=ActionSpec().encode(p,-before,0.)
        sim.catcher.buffer.submit(np.tile(action,(12,1)),0,0,action)
        sim.catcher.step(sim.data,0,None)
        self.assertEqual(sim.catcher.ik_rejected,1)
        np.testing.assert_allclose(sim.catcher.normal,before)

if __name__=="__main__":unittest.main()
