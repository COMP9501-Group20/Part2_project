"""Print a seed/ablation training plan, or run it sequentially with --execute."""
import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from project.experiments import load_experiments

def main():
    cfg=load_experiments()
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variants',nargs='+',choices=list(cfg['variants']),default=list(cfg['variants']))
    p.add_argument('--seeds',nargs='+',type=int,default=cfg['seeds'])
    p.add_argument('--iterations',type=int,default=cfg['train']['iterations'])
    p.add_argument('--num-envs',type=int,default=4096)
    p.add_argument('--execute',action='store_true')
    a=p.parse_args()
    for variant in a.variants:
        for seed in a.seeds:
            command=[sys.executable,str(ROOT/'scripts/run.py'),'train','--variant',variant,'--seed',str(seed),
                     '--iterations',str(a.iterations),'--num-envs',str(a.num_envs),'--headless']
            print(shlex.join(command),flush=True)
            if a.execute:subprocess.run(command,check=True,cwd=ROOT)
if __name__=='__main__':main()
