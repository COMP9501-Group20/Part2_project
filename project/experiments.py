"""Dependency-free experiment definitions and configuration overrides."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def load_experiments():
    return json.loads((ROOT / 'configs/experiments.json').read_text(encoding='utf-8'))

def apply_variant(env_cfg, train_cfg, name, evaluation=False):
    spec = load_experiments()
    v = spec['variants'][name]
    env_cfg.env.action_curriculum = v['curriculum'] and not evaluation
    env_cfg.env.upper_sampling = 'full_support' if evaluation else v['upper_sampling']
    env_cfg.env.capture_metrics = evaluation
    env_cfg.domain_rand.init_upper_ratio = 1.0 if evaluation or not v['curriculum'] else 0.0
    env_cfg.rewards.scales.tracking_base_height = v['height_weight']
    env_cfg.rewards.scales.deviation_knee_joint = v['knee_weight']
    train_cfg.algorithm.use_flip = v['augmentation']
    train_cfg.algorithm.use_symmetry_loss = v['symmetry_loss']
    train_cfg.runner.logger = 'tensorboard'
    train_cfg.runner.experiment_name = 'group20_g1'
    return v
