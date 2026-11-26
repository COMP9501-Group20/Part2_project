"""Train, evaluate or view a G1 policy in simulation."""
import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from project.experiments import load_experiments, apply_variant

def parse_args(argv=None):
    specs = load_experiments()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['train', 'evaluate', 'play'])
    p.add_argument('--variant', choices=list(specs['variants']), default='full')
    p.add_argument('--seed', type=int, default=0, help='Training seed (also used to identify evaluation results)')
    p.add_argument('--eval-seed', type=int, default=12345)
    p.add_argument('--num-envs', type=int)
    p.add_argument('--iterations', type=int, default=specs['train']['iterations'])
    p.add_argument('--horizon', type=float, default=20.)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--headless', action='store_true')
    p.add_argument('--dry-run', action='store_true', help='Print the plan without importing Isaac Gym')
    a = p.parse_args(argv)
    if a.num_envs is None:
        a.num_envs = {'train':4096, 'evaluate':1000, 'play':16}[a.mode]
    if a.num_envs < 1 or a.iterations < 1 or a.horizon <= 0:
        p.error('num-envs, iterations and horizon must be positive')
    if a.mode != 'train' and a.checkpoint is None and not a.dry_run:
        p.error('evaluate/play requires an explicit --checkpoint')
    return a

def git_state():
    def git(*args):
        try:
            return subprocess.check_output(['git','-C',str(ROOT),*args], stderr=subprocess.DEVNULL, text=True).strip()
        except (FileNotFoundError, subprocess.CalledProcessError):
            return None
    return {'commit':git('rev-parse','HEAD'),'dirty':bool(git('status','--porcelain'))}

def validate_checkpoint(path, variant, seed):
    checkpoint=path.resolve(strict=True)
    manifest_path=checkpoint.parent/'run.json'
    if not manifest_path.exists():
        raise ValueError('Checkpoint needs its training run.json for variant/seed verification')
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    plan=manifest['plan']
    if plan['mode']!='train' or plan['variant']!=variant or plan['seed']!=seed:
        raise ValueError('Checkpoint training variant/seed does not match the requested experiment')
    return checkpoint

def main(argv=None):
    a = parse_args(argv)
    plan = dict(vars(a), design=load_experiments()['variants'][a.variant])
    if a.dry_run:
        print(json.dumps(plan, indent=2, default=str))
        return
    checkpoint=validate_checkpoint(a.checkpoint,a.variant,a.seed) if a.checkpoint else None
    # Isaac Gym must be imported before torch.
    try:
        import isaacgym
        from isaacgym import gymapi
    except ImportError as e:
        raise SystemExit('Isaac Gym Preview 4 is required. Use Ubuntu + Python 3.8; see docs/SETUP.md. --dry-run works without it.') from e
    import torch
    from legged_gym.envs.g1.g1_29dof_config import G1RoughCfg, G1RoughCfgPPO
    from legged_gym.utils.helpers import class_to_dict, set_seed, parse_sim_params
    from rsl_rl.runners import HIMOnPolicyRunner
    from project.environment import CourseG1
    from project.metrics import EpisodeMetrics

    env_cfg, train_cfg = G1RoughCfg(), G1RoughCfgPPO()
    apply_variant(env_cfg, train_cfg, a.variant, evaluation=a.mode != 'train')
    env_cfg.env.num_envs = a.num_envs
    env_cfg.env.episode_length_s = a.horizon if a.mode != 'train' else 20.
    train_cfg.seed = a.seed
    set_seed(a.seed if a.mode == 'train' else a.eval_seed)
    sim_args = SimpleNamespace(physics_engine=gymapi.SIM_PHYSX, use_gpu=a.device.startswith('cuda'),
                              use_gpu_pipeline=a.device.startswith('cuda'),subscenes=0,num_threads=0)
    sim_params = parse_sim_params(sim_args, {'sim':class_to_dict(env_cfg.sim)})
    sim_params.use_gpu_pipeline = a.device.startswith('cuda')
    env = CourseG1(env_cfg, sim_params, gymapi.SIM_PHYSX, a.device, a.headless or a.mode=='evaluate')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    run_dir = a.output or ROOT/'runs'/f'{a.variant}_seed{a.seed}_{stamp}'
    run_dir.mkdir(parents=True, exist_ok=False)
    metadata = {'schema_version':1, 'created_at':stamp, 'status':'started', 'plan':plan,
                'code':git_state(), 'python':sys.version, 'platform':platform.platform(),
                'torch':torch.__version__, 'env_config':class_to_dict(env_cfg), 'train_config':class_to_dict(train_cfg)}
    if a.checkpoint:
        metadata['checkpoint_sha256'] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    (run_dir/'run.json').write_text(json.dumps(metadata,indent=2,default=str)+'\n',encoding='utf-8')
    try:
        runner = HIMOnPolicyRunner(env,class_to_dict(train_cfg),str(run_dir) if a.mode=='train' else None,a.device)
        if a.checkpoint:
            runner.load(str(checkpoint),load_optimizer=a.mode=='train')
        if a.mode=='train':
            runner.learn(a.iterations,init_at_random_ep_len=True)
        else:
            policy=runner.get_inference_policy(device=env.device)
            # Reset without the extra simulation step in BaseTask.reset().
            env.reset_idx(torch.arange(env.num_envs,device=env.device))
            env.obs_buf.zero_()
            env.privileged_obs_buf.zero_()
            env.compute_observations()
            obs=env.get_observations()
            meter=EpisodeMetrics(env.num_envs,env.device,env.dt)
            steps=round(a.horizon/env.dt)
            if steps < 1: raise ValueError('horizon is shorter than one policy step')
            with torch.no_grad():
                for _ in range(steps):
                    actions=policy(obs)
                    mirror_actions=policy(runner.alg.flip_g1_actor_obs(obs))
                    symmetry=(mirror_actions-runner.alg.flip_g1_actions(actions)).square().sum(-1)
                    obs,_,_,_,_,_,_=env.step(actions)
                    values=dict(env.metric_snapshot,policy_symmetry_mse=symmetry)
                    meter.update(values,env.metric_failed)
            result={'schema_version':1,'variant':a.variant,'training_seed':a.seed,'evaluation_seed':a.eval_seed,
                    'checkpoint':str(checkpoint),'checkpoint_sha256':metadata['checkpoint_sha256'],
                    'num_envs':env.num_envs,'horizon_s':steps*env.dt,'upper_ratio':1.0,
                    'metrics':meter.summary(),'status':'measured','protocol':'group20-first-episode-v1',
                    'code':metadata['code']}
            (run_dir/'metrics.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
            print(json.dumps(result,indent=2))
        metadata['status']='completed'
    except BaseException as e:
        metadata['status']='failed'
        metadata['error']=str(e)
        raise
    finally:
        (run_dir/'run.json').write_text(json.dumps(metadata,indent=2,default=str)+'\n',encoding='utf-8')
        if env.viewer:env.gym.destroy_viewer(env.viewer)
        env.gym.destroy_sim(env.sim)

if __name__=='__main__': main()
