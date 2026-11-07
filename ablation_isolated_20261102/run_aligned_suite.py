"""Five-group evaluation anchored to the frozen Table 1 pocket cohort."""
import os
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import sys,argparse,json,random,multiprocessing
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
sys.dont_write_bytecode=True
import run as base
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import RDConfig
from aligned_worker_v1 import worker

def prepare(a):
    out=base.new_directory(a.out);cache=Path(a.table1_cache).resolve();final=Path(a.table1_final).resolve()
    manifest=json.loads((cache/'manifest.json').read_text());entries=manifest['files'];source=json.loads((final/'samples.json').read_text())
    rows=[r for r in source if r['variant']=='balanced']
    expected={(e['index'],j) for e in entries for j in range(manifest['args']['samples'])}
    if len(rows)!=len(expected) or {(r['pocket'],r['sample']) for r in rows}!=expected:raise ValueError('Table 1 final rows do not match cache cohort')
    for e in entries:
        if base.sha(cache/e['name'])!=e['sha256']:raise ValueError('Input cache changed')
    meta=json.loads((final/'run_metadata.json').read_text())
    if Path(meta['cache']).resolve()!=cache:raise ValueError('Final run refers to another cache')
    if len(entries)!=100 or manifest['args']['samples']!=10:raise ValueError('Expected frozen 100-pocket, 10-sample Table 1 cohort')
    # Read and preserve the Full results, without recomputing or changing them.
    full=out/'variant_5';full.mkdir()
    base.write(full/'samples.json',rows)
    summary={k:dict(mean=float(np.mean([r[k] for r in rows])),count=len(rows)) for k in ['qed','sa','vina','strain','clashes']}
    summary.update(attempts=len(rows),valid_pct=100*sum(r['valid'] for r in rows)/len(rows),connectivity_pct=100*sum(r['connected'] for r in rows)/len(rows),uniqueness_pct=100*len({r['smiles'] for r in rows})/len(rows),source='frozen Table 1 Full')
    if abs(summary['vina']['mean']-(-8.312309))>1e-6 or abs(summary['strain']['mean']-242.51840514872904)>1e-6:raise ValueError('Select the frozen Table 1 second-pass result directory')
    base.write(full/'summary.json',summary)
    plan=dict(table1_cache=str(cache),table1_final=str(final),manifest_sha256=base.sha(cache/'manifest.json'),
        table1_samples_sha256=base.sha(final/'samples.json'),checkpoint_sha256=manifest['checkpoint_sha256'],
        samples=10,pockets=100,files=entries,strain_steps=300,
        variants={'1':'unseeded atom cloud, no bridging','2':'unseeded + component bridging','3':'fixed-plane core + bridging','4':'aligned core + sp2 budget + Kekule','5':'frozen complete Table 1 workflow including graph recovery'},
        common_evaluation='20 A box; exhaustiveness 4; cpu 1; entry seed + sample + 10000; 50+300 UFF; contact threshold 1.5 A',
        common_selection='same bounded whole/local/intermediate proposals then local/two torsions for valid connected molecules; retain input; no added torsion-extension or rigid-body search')
    base.write(out/'plan.json',plan)
    print('Prepared Full:',json.dumps(summary,indent=2),flush=True)

def generate(a):
    suite=base.confined(a.suite);plan=json.loads((suite/'plan.json').read_text());out=suite/f'cache_v{a.variant}'
    if a.resume:
        manifest=json.loads((out/'manifest.json').read_text())
        if manifest['variant']!=a.variant or manifest['table1_manifest_sha256']!=plan['manifest_sha256'] or manifest['script_sha256']!=base.sha(__file__):raise ValueError('Generation resume source or code changed')
        for e in manifest['files']:
            if base.sha(out/e['name'])!=e['sha256']:raise ValueError('Saved cache changed')
    else:
        base.new_directory(out);manifest=dict(variant=a.variant,samples=10,files=[],checkpoint_sha256=plan['checkpoint_sha256'],table1_manifest_sha256=plan['manifest_sha256'],script_sha256=base.sha(__file__))
        base.write(out/'manifest.json',manifest)
    if base.sha(a.checkpoint)!=plan['checkpoint_sha256']:raise ValueError('Use the same checkpoint as Table 1')
    device=torch.device(a.device);checkpoint=torch.load(a.checkpoint,map_location=device,weights_only=False);cfg=checkpoint['config']
    model=base.legacy.BACKBONE_REGISTRY.build(cfg['model']).to(device)
    model.load_state_dict(checkpoint['ema_state_dict'] if 'ema_state_dict' in checkpoint else checkpoint['model_state_dict']);model.eval()
    sde=base.legacy.DualSpaceVPSDE(schedule_type='cosine').to(device)
    sampler=base.legacy.ReverseSDESampler(sde=sde,num_steps=100,spring_k=.25,corrector_steps=1,bond_gate_thresh=.35,
        enable_svd_seeding=a.variant>=4,enable_self_healing=False,device=device)
    completed={e['index'] for e in manifest['files']}
    for entry in plan['files']:
        if entry['index'] in completed:continue
        raw=Path(plan['table1_cache'])/entry['name']
        if base.sha(raw)!=entry['sha256']:raise ValueError('Table 1 input changed')
        with np.load(raw) as z:pp,ph,n=[z['pocket_pos'],z['pocket_h'],z['x'].shape[1]]
        seed=entry['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
        p=torch.as_tensor(pp,device=device).unsqueeze(0);hprot=torch.as_tensor(ph,device=device).unsqueeze(0);B=10
        with torch.no_grad():
            if a.variant>=3:
                x,h,_=sampler.sample(model=model,num_samples=B,num_atoms=n,num_atom_types=9,model_kwargs=dict(pocket_pos=p.repeat(B,1,1),pocket_h=hprot.repeat(B,1,1)))
            else:
                x=p.mean(1,keepdim=True)+torch.randn(B,n,3,device=device)*1.5;h=torch.randn(B,n,9,device=device)
                times=torch.linspace(.999,.001,101,device=device)
                for step in range(100):
                    t=torch.full((B,),times[step].item(),device=device);prev=torch.full((B,),times[step+1].item(),device=device)
                    ex,eh,_=model(x_t=x,h_t=h,t=t,pocket_pos=p.repeat(B,1,1),pocket_h=hprot.repeat(B,1,1))
                    alpha=sde.compute_alpha_bar(t).view(B,1,1);aa=torch.sqrt(alpha.clamp(min=1e-4));bb=torch.sqrt((1-alpha).clamp(min=1e-4))
                    x0=(x-bb*ex)/aa;h0=(h-bb*eh)/aa
                    if step==99:x,h=x0,h0
                    else:x,h=sde.q_posterior_step(x0,h0,x,h,t,prev)
        target=out/entry['name'];np.savez_compressed(target,x=x.cpu().numpy(),h=h.cpu().numpy(),pocket_pos=pp,pocket_h=ph)
        manifest['files'].append(dict(entry,sha256=base.sha(target)));base.write(out/'manifest.json',manifest)
        print(f'Variant {a.variant}: generated {len(manifest["files"])}/100 pockets',flush=True)

def extended_summary(rows):
    result=base.summarize(rows);n=len(rows)
    result.update(valid_pct=100*result['valid_count']/n,connectivity_pct=100*result['connected_count']/n)
    valid=[r for r in rows if 'smiles' in r]
    result['uniqueness_pct']=100*len({r['smiles'] for r in valid})/len(valid) if valid else None
    sys.path.append(str(Path(RDConfig.RDContribDir)/'SA_Score'));import sascorer
    values=[];descriptor_errors=[]
    for r in valid:
        try:
            mol=Chem.MolFromSmiles(r['smiles'])
            if mol is None:raise ValueError('SMILES parsing failed')
            values.append((10-sascorer.calculateScore(mol))/9)
        except Exception as exc:descriptor_errors.append(dict(pocket=r['pocket'],sample=r['sample'],error=str(exc)))
    result['descriptor_errors']=descriptor_errors
    result['sa']=dict(mean=float(np.mean(values)) if values else None,count=len(values))
    return result

def compare(a):
    suite=base.confined(a.suite);cache=suite/f'cache_v{a.variant}';out=suite/f'variant_{a.variant}'
    manifest=json.loads((cache/'manifest.json').read_text());plan=json.loads((suite/'plan.json').read_text())
    if len(manifest['files'])!=100 or manifest['table1_manifest_sha256']!=plan['manifest_sha256']:raise ValueError('Cache incomplete or from another cohort')
    identity=dict(manifest_sha256=base.sha(cache/'manifest.json'),script_sha256=base.sha(__file__),worker_sha256=base.sha(Path(__file__).parent/'aligned_worker_v1.py'))
    if a.resume:
        old=json.loads((out/'run_metadata.json').read_text())
        if any(old[k]!=v for k,v in identity.items()):raise ValueError('Resume inputs changed')
    else:base.new_directory(out)
    (out/'tasks').mkdir(exist_ok=True)
    base.write(out/'run_metadata.json',dict(identity,variant=a.variant,strain_steps=300,workers=a.workers,table1_plan=plan))
    done={};tasks=[]
    for e in manifest['files']:
        if base.sha(cache/e['name'])!=e['sha256']:raise ValueError('Variant cache changed')
        for j in range(10):
            file=out/'tasks'/f"p{e['index']}_s{j}.json"
            if a.resume and file.exists():done[e['index'],j]=json.loads(file.read_text())
            else:tasks.append((str(cache),str(out),e,j,a.variant,300))
    def save():
        rows=[done[k] for k in sorted(done)];base.write(out/'samples.json',rows);base.write(out/'summary.json',extended_summary(rows))
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for f in as_completed([pool.submit(worker,t) for t in tasks]):
            r=f.result();done[r['pocket'],r['sample']]=r
            # Raw progress is cheap; descriptor summary is written at completion.
            base.write(out/'samples.json',[done[k] for k in sorted(done)])
            print(f'Variant {a.variant}: completed {len(done)}/1000 molecules',flush=True)
    save();print(json.dumps(json.loads((out/'summary.json').read_text()),indent=2),flush=True)

def collect(a):
    suite=base.confined(a.suite);matrix=[]
    for v in range(1,6):
        file=suite/f'variant_{v}/summary.json'
        if not file.exists():raise ValueError(f'Variant {v} has no completed summary')
        s=json.loads(file.read_text())
        if s['attempts']!=1000:raise ValueError(f'Variant {v} incomplete')
        matrix.append(dict(variant=v,summary=s))
    base.write(suite/'five_group_summary.json',matrix)
    print('Saved five_group_summary.json; Full remains frozen Table 1 result.',flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='stage',required=True)
    x=sub.add_parser('prepare');x.add_argument('--table1-cache',required=True);x.add_argument('--table1-final',required=True);x.add_argument('--out',required=True)
    for name in ['generate','compare']:
        x=sub.add_parser(name);x.add_argument('--suite',required=True);x.add_argument('--variant',type=int,choices=[1,2,3,4],required=True)
        if name=='generate':x.add_argument('--checkpoint',required=True);x.add_argument('--device',default='cuda');x.add_argument('--resume',action='store_true')
        else:x.add_argument('--workers',type=int,default=16);x.add_argument('--resume',action='store_true')
    x=sub.add_parser('collect');x.add_argument('--suite',required=True)
    a=p.parse_args()
    if getattr(a,'workers',1)<1:p.error('workers must be positive')
    {'prepare':prepare,'generate':generate,'compare':compare,'collect':collect}[a.stage](a)
if __name__=='__main__':main()
