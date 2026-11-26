import contextlib
import io
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'rsl_rl'),str(ROOT/'scripts')]
import torch
from rsl_rl.modules import HIMActorCritic, HIMEstimator
from rsl_rl.algorithms import HIMPPO
from rsl_rl.storage import HIMRolloutStorage
from project.metrics import EpisodeMetrics
from project.experiments import apply_variant, load_experiments
from summarize import summarize
from run import validate_checkpoint

torch.set_num_threads(1)

def algorithm(augmentation=True, regularizer=True):
    with contextlib.redirect_stdout(io.StringIO()):
        actor=HIMActorCritic(456,79,76,79,6,1,12,actor_hidden_dims=[32],critic_hidden_dims=[32])
    return HIMPPO(actor,use_flip=augmentation,use_symmetry_loss=regularizer,
                  num_learning_epochs=1,num_mini_batches=2,device='cpu',symmetry_scale=.1)

class CoreTests(unittest.TestCase):
    def test_checkpoint_metadata_prevents_seed_mislabeling(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'model.pt';path.write_bytes(b'test-fixture-not-a-model')
            (Path(tmp)/'run.json').write_text(json.dumps({'plan':{'mode':'train','variant':'full','seed':2}}))
            self.assertEqual(validate_checkpoint(path,'full',2),path.resolve())
            with self.assertRaises(ValueError):validate_checkpoint(path,'full',0)
            with self.assertRaises(ValueError):validate_checkpoint(path,'no_symmetry',2)

    def test_mirror_is_involution_for_entire_history(self):
        a=algorithm()
        for size,flip in [(456,a.flip_g1_actor_obs),(79,a.flip_g1_critic_obs),(12,a.flip_g1_actions)]:
            x=torch.randn(8,size)
            torch.testing.assert_close(flip(flip(x)),x)
        x=torch.zeros(1,456);x[0,1]=1.;x[0,77]=2.
        y=a.flip_g1_actor_obs(x)
        self.assertEqual(y[0,1],-1.);self.assertEqual(y[0,77],-2.)

    def test_mirrored_transition_uses_executed_action(self):
        a=algorithm();a.init_storage(4,2,[456],[79],[12])
        with torch.no_grad():
            action=a.act(torch.randn(4,456),torch.randn(4,79))
            torch.testing.assert_close(a.transition_sym.actions,a.flip_g1_actions(action))
            a.process_env_step(torch.ones(4),torch.zeros(4,dtype=torch.bool),{},torch.randn(4,79))
        self.assertEqual(a.storage.step,2)

    def test_gae_bootstraps_branches_separately(self):
        s=HIMRolloutStorage(1,2,[1],[1],[1],mirrored=True)
        s.step=4;s.rewards[:,0,0]=torch.tensor([1.,10.,2.,20.])
        s.compute_returns(torch.tensor([[[3.]],[[30.]]]),1.,1.)
        torch.testing.assert_close(s.returns[:,0,0],torch.tensor([6.,60.,5.,50.]))

    def test_single_branch_terminal_gae(self):
        s=HIMRolloutStorage(1,2,[1],[1],[1],mirrored=False)
        s.step=2;s.rewards[:,0,0]=torch.tensor([1.,2.]);s.dones[-1]=1
        s.compute_returns(torch.tensor([[[99.]]]),1.,1.)
        torch.testing.assert_close(s.returns[:,0,0],torch.tensor([3.,2.]))

    def test_four_symmetry_settings_update_with_finite_losses(self):
        for aug in [False,True]:
            for reg in [False,True]:
                with self.subTest(augmentation=aug,regularizer=reg):
                    a=algorithm(aug,reg);a.init_storage(8,3,[456],[79],[12])
                    before=next(a.actor_critic.actor.parameters()).detach().clone()
                    with torch.no_grad():
                        for _ in range(3):
                            a.act(torch.randn(8,456),torch.randn(8,79))
                            a.process_env_step(torch.randn(8),torch.zeros(8,dtype=torch.bool),{},torch.randn(8,79))
                        self.assertEqual(a.storage.step,6 if aug else 3)
                        a.compute_returns(torch.randn(8,79))
                    losses=a.update()
                    self.assertTrue(all(math.isfinite(x) for x in losses))
                    self.assertFalse(torch.equal(before,next(a.actor_critic.actor.parameters())))
                    if not reg:self.assertEqual(losses[-2:],(0,0))

    def test_successor_encoder_excludes_privileged_velocity(self):
        estimator=HIMEstimator(6,76,0)
        captured=[]
        hook=estimator.target.register_forward_pre_hook(lambda _,args:captured.append(args[0].clone()))
        nxt=torch.randn(8,79);nxt[:,76:]=100.
        estimator.update(torch.randn(8,456),nxt)
        hook.remove()
        torch.testing.assert_close(captured[0],nxt[:,:76])

    def test_metrics_ignore_post_failure_resets(self):
        m=EpisodeMetrics(2,'cpu',.02)
        m.update({'error':torch.tensor([1.,2.])},torch.tensor([True,False]))
        m.update({'error':torch.tensor([999.,4.])},torch.tensor([False,False]))
        d=m.summary()
        self.assertAlmostEqual(d['error'],2.)
        self.assertAlmostEqual(d['living_time_s'],.03)
        self.assertAlmostEqual(d['survival_rate'],.5)

    def test_variant_evaluation_has_equal_full_difficulty(self):
        for name in load_experiments()['variants']:
            env=NS(env=NS(),domain_rand=NS(),rewards=NS(scales=NS()))
            train=NS(algorithm=NS(),runner=NS())
            apply_variant(env,train,name,evaluation=True)
            self.assertFalse(env.env.action_curriculum)
            self.assertEqual(env.env.upper_sampling,'full_support')
            self.assertEqual(env.domain_rand.init_upper_ratio,1.)
        apply_variant(env,train,'loss_only')
        self.assertFalse(train.algorithm.use_flip)
        self.assertTrue(train.algorithm.use_symmetry_loss)

    def test_summary_seed_std_and_duplicate_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths=[]
            for seed in [0,1,2]:
                p=Path(tmp)/f'{seed}.json'
                p.write_text(json.dumps(dict(status='measured',variant='test',training_seed=seed,
                     evaluation_seed=1,num_envs=2,horizon_s=20,upper_ratio=1,protocol='test',metrics={'x':float(seed)})))
                paths.append(p)
            r=summarize(paths)[0]
            self.assertEqual(r['mean'],1.)
            self.assertEqual(r['std'],1.)
            with self.assertRaises(ValueError):summarize(paths+[paths[0]])
            d=json.loads(paths[0].read_text());d['horizon_s']=10;paths[0].write_text(json.dumps(d))
            with self.assertRaises(ValueError):summarize(paths)

if __name__=='__main__':unittest.main()
