"""Aggregate actual evaluation files across independent training seeds."""
import argparse
import csv
import json
import statistics
import math
from pathlib import Path

def summarize(paths):
    groups={}
    protocol=None
    seen=set()
    for path in paths:
        d=json.loads(Path(path).read_text(encoding='utf-8'))
        if d.get('status')!='measured':raise ValueError('Not a measured result: '+str(path))
        signature=tuple(d[k] for k in ('protocol','evaluation_seed','num_envs','horizon_s','upper_ratio'))
        if protocol is not None and signature!=protocol:raise ValueError('Evaluation protocols differ')
        protocol=signature
        key=(d['variant'],d['training_seed'])
        if key in seen:raise ValueError('Duplicate variant/training seed: '+str(key))
        seen.add(key)
        groups.setdefault(d['variant'],[]).append(d['metrics'])
    if not groups:raise ValueError('No measured files found')
    rows=[]
    for variant, runs in sorted(groups.items()):
        if any(set(r)!=set(runs[0]) for r in runs):raise ValueError('Metric keys differ')
        for metric in runs[0]:
            values=[r[metric] for r in runs]
            if not all(isinstance(x,(int,float)) and math.isfinite(x) for x in values):
                raise ValueError('Non-finite or nonnumeric metric: '+metric)
            rows.append({'variant':variant,'metric':metric,'seeds':len(values),'mean':statistics.mean(values),
                         'std':statistics.stdev(values) if len(values)>1 else ''})
    return rows

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('files',type=Path,nargs='+')
    p.add_argument('--output',type=Path,default=Path('results/summary.csv'))
    p.add_argument('--plot',type=Path)
    a=p.parse_args()
    rows=summarize(a.files)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    if a.plot:
        import matplotlib.pyplot as plt
        metrics=list(dict.fromkeys(r['metric'] for r in rows))
        fig,axs=plt.subplots(2,3,figsize=(16,9),squeeze=False)
        for ax,metric in zip(axs.flat,metrics):
            selected=[r for r in rows if r['metric']==metric]
            ax.bar([r['variant'] for r in selected],[r['mean'] for r in selected],
                   yerr=[r['std'] or 0 for r in selected],color='#476C9B',capsize=3)
            ax.set_title(metric);ax.tick_params(axis='x',rotation=60);ax.grid(axis='y',alpha=.2)
        fig.suptitle('Measured simulation evaluations: mean and sample SD across training seeds')
        fig.tight_layout();a.plot.parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.plot,dpi=180)
if __name__=='__main__':main()
