"""Independent cached ablation development run. All code and outputs remain in this isolated directory."""
import os
os.environ["PYTHONDONTWRITEBYTECODE"]="1"
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ[key]='1'
import argparse,json,hashlib,sys,multiprocessing,shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem,QED,rdMolDescriptors
ROOT=Path(__file__).resolve().parent
PROJECT_ROOT=ROOT.parent
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'vendor'))
import legacy_ablation_v2 as legacy
from trial_modules.pocket_geometry_bounded_trial import refine_in_pocket as whole_refine
from trial_modules.contact_anchored_local_trial import refine_in_pocket as local_refine
from trial_modules.intermediate_pose_trial import intermediate_candidates
from trial_modules.torsion_recovery_trial import local_candidates

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2),encoding='utf-8');os.replace(tmp,path)
def confined(path):
    path=Path(path).resolve()
    if not path.is_relative_to(ROOT):raise ValueError('Outputs must remain inside isolated directory: '+str(ROOT))
    return path

def new_directory(path):
    path=confined(path)
    if path.exists() and any(path.iterdir()):raise ValueError('Use a new empty output directory: '+str(path))
    path.mkdir(parents=True,exist_ok=True);return path


def generate(a):
    out=new_directory(a.out);device=torch.device(a.device)
    ckpt=torch.load(a.checkpoint,map_location=device);cfg=ckpt['config']
    model=legacy.BACKBONE_REGISTRY.build(cfg['model']).to(device)
    model.load_state_dict(ckpt.get('ema_state_dict',ckpt['model_state_dict']));model.eval()
    sde=legacy.DualSpaceVPSDE(schedule_type=cfg.get('diffusion',{}).get('schedule_type','cosine')).to(device)
    dataset=legacy.CrossDockedLMDBDataset(lmdb_dir=a.lmdb_dir or cfg['dataset']['lmdb_dir'],split='test',split_type='name')
    if a.pockets>len(dataset):raise ValueError('Requested pockets unavailable')
    seeded=a.variant>=3
    sampler=legacy.ReverseSDESampler(sde=sde,num_steps=100,spring_k=.25,corrector_steps=1,bond_gate_thresh=.35,
        enable_svd_seeding=a.variant>=4,enable_self_healing=a.variant==5,device=device)
    manifest=dict(variant=a.variant,pockets=a.pockets,samples=a.samples,seed=a.seed,checkpoint=str(Path(a.checkpoint).resolve()),
        checkpoint_sha256=sha(a.checkpoint),script_sha256=sha(__file__),legacy_sha256=sha(legacy.__file__),files=[])
    write(out/'manifest.json',manifest)
    for i in range(a.pockets):
        seed=a.seed+1009*i;torch.manual_seed(seed);np.random.seed(seed)
        if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
        sample=dataset[i];pp=sample['pocket_pos'].unsqueeze(0).to(device);ph=sample['pocket_h'].unsqueeze(0).to(device)
        n=max(18,int(sample['num_lig_atoms']));B=a.samples
        with torch.no_grad():
            if seeded:
                x,h,_=sampler.sample(model=model,num_samples=B,num_atoms=n,num_atom_types=9,
                    model_kwargs=dict(pocket_pos=pp.repeat(B,1,1),pocket_h=ph.repeat(B,1,1)))
            else:
                x=pp.mean(1,keepdim=True)+torch.randn(B,n,3,device=device)*1.5;h=torch.randn(B,n,9,device=device)
                times=torch.linspace(.999,.001,101,device=device)
                for step in range(100):
                    t=torch.full((B,),times[step].item(),device=device);prev=torch.full((B,),times[step+1].item(),device=device)
                    ex,eh,_=model(x_t=x,h_t=h,t=t,pocket_pos=pp.repeat(B,1,1),pocket_h=ph.repeat(B,1,1))
                    alpha=sde.compute_alpha_bar(t).view(B,1,1)
                    x0=(x-torch.sqrt((1-alpha).clamp(min=1e-4))*ex)/torch.sqrt(alpha.clamp(min=1e-4))
                    h0=(h-torch.sqrt((1-alpha).clamp(min=1e-4))*eh)/torch.sqrt(alpha.clamp(min=1e-4))
                    if step==99:x,h=x0,h0
                    else:x,h=sde.q_posterior_step(x0,h0,x,h,t,prev)
        name=f'pocket_{i:05d}.npz'
        np.savez_compressed(out/name,x=x.cpu().numpy(),h=h.cpu().numpy(),pocket_pos=pp[0].cpu().numpy(),pocket_h=ph[0].cpu().numpy())
        manifest['files'].append(dict(index=i,name=name,seed=seed,sha256=sha(out/name)));write(out/'manifest.json',manifest)
        print(f'Generated {i+1}/{a.pockets} pockets; variant {a.variant}',flush=True)

def strain(mol,steps):
    work=Chem.AddHs(Chem.Mol(mol),addCoords=True);Chem.FastFindRings(work)
    if not AllChem.UFFHasAllMoleculeParams(work):raise ValueError('UFF parameters unavailable')
    AllChem.UFFOptimizeMolecule(work,maxIters=50);ff=AllChem.UFFGetMoleculeForceField(work)
    e0=ff.CalcEnergy();ff.Minimize(maxIts=steps);e1=ff.CalcEnergy()
    if not np.isfinite([e0,e1]).all():raise ValueError('Nonfinite UFF energy')
    return float(min(1000,max(0,e0-e1)))

def worker(task):
    cache,out,entry,j,variant,steps=task;out=Path(out);torch.set_num_threads(1)
    with np.load(Path(cache)/entry['name']) as z:x,h,pp,ph=[z[k] for k in ['x','h','pocket_pos','pocket_h']]
    types=ph.argmax(-1);heavy=types>0;nums=np.array([1,6,7,8,16,34])[types[heavy]]
    if variant==5:
        mol,valid,frags=legacy.BondPerceptionEngine.reconstruct_mol(pos=torch.from_numpy(x[j]),atom_types=torch.from_numpy(h[j]),tolerance=.35)
    else:
        elements=[legacy.SBDD_HEAVY_REVERSE_MAP.get(int(k),6) for k in h[j].argmax(-1)]
        mol,valid,frags=legacy.reconstruct_ablation_v2(x[j],elements,has_seed=variant>=3,use_mst=variant>=2,
            use_sp2_budget=variant>=4,use_kekule=variant>=4,use_forbidden_filter=False,use_triangulation_suppression=False,tolerance=.35)
    result=dict(pocket=entry['index'],sample=j,variant=variant,valid=bool(valid),connected=frags==1,proposals=[])
    path=out/'tasks'/f"p{entry['index']}_s{j}.json"
    if not valid or mol is None:
        result['status']='reconstruction_not_accepted';write(path,result);return result
    try:
        graph=Chem.MolToSmiles(mol);result.update(smiles=graph,qed=float(QED.qed(mol)),aromatic_rings=int(rdMolDescriptors.CalcNumAromaticRings(mol)))
    except Exception as exc:
        result.update(status='scoring_error',error=str(exc));write(path,result);return result
    rec=out/f"receptor_p{entry['index']}_s{j}.pdbqt";legacy.write_pocket_pdbqt_file(pp,types,str(rec))
    def metrics(candidate):
        assert Chem.MolToSmiles(candidate)==graph
        xyz=candidate.GetConformer().GetPositions();clashes=int((np.linalg.norm(xyz[:,None]-pp[heavy][None],axis=-1)<1.5).sum())
        energy=strain(candidate,steps)
        ligand=legacy.mol_to_pdbqt_meeko_safe(candidate)
        if not ligand:raise ValueError('Ligand PDBQT preparation failed')
        v=legacy.Vina(sf_name='vina',cpu=1,seed=entry['seed']+j+10000,verbosity=0)
        v.set_receptor(str(rec));v.set_ligand_from_string(ligand);v.compute_vina_maps(center=pp.mean(0).tolist(),box_size=[20.,20.,20.])
        v.dock(exhaustiveness=4,n_poses=1);score=float(v.energies(n_poses=1)[0][0])
        if not np.isfinite(score):raise ValueError('Nonfinite Vina score')
        return dict(vina=score,strain=energy,clashes=clashes)
    def eligible(candidate,ref):return all(candidate[k]<=ref[k]+1e-8 for k in ['vina','strain','clashes'])
    def evaluate(candidate,name,ref,info):
        try:
            m=metrics(candidate);ok=eligible(m,ref);result['proposals'].append(dict(source=name,metrics=m,eligible=ok,details=info))
            return (candidate,m,name) if ok else None
        except Exception as exc:
            result['proposals'].append(dict(source=name,error=str(exc),details=info));return None
    try:
        base=metrics(mol);result['baseline']=base;best=(mol,base,'input')
        if frags==1:
            pool=[best];whole,wi=whole_refine(mol,pp[heavy],nums)
            local,li=local_refine(mol,pp[heavy],nums)
            for candidate,name,info in [(whole,'whole',wi),(local,'local',li)]:
                if info.get('accepted'):
                    row=evaluate(candidate,name,base,info)
                    if row:pool.append(row)
            best=min(pool,key=lambda t:(t[1]['strain'],t[1]['vina']))
            if wi.get('accepted'):
                middle=[]
                for k,(candidate,info) in enumerate(intermediate_candidates(mol,whole.GetConformer().GetPositions(),pp[heavy])):
                    row=evaluate(candidate,f'intermediate_{k}',best[1],info)
                    if row and row[1]['vina']<best[1]['vina']-1e-6:middle.append(row)
                if middle:best=min([best]+middle,key=lambda t:t[1]['vina'])
            anchor=best;pool=[anchor];local,info=local_refine(anchor[0],pp[heavy],nums)
            proposals=[('second_local',local,info)] if info.get('accepted') else []
            proposals += [(f'torsion_{k}',candidate,info) for k,(candidate,info) in enumerate(local_candidates(anchor[0]))]
            for name,candidate,info in proposals:
                row=evaluate(candidate,name,anchor[1],info)
                if row:pool.append(row)
            best=min(pool,key=lambda t:(t[1]['strain'],t[1]['vina'],t[1]['clashes']))
        result.update(status='scored',selected=best[1],selected_source=best[2])
        for name,candidate in [('input',mol),('selected',best[0])]:
            (out/f"p{entry['index']}_s{j}_{name}.mol").write_text(Chem.MolToMolBlock(candidate))
            np.savez_compressed(out/f"p{entry['index']}_s{j}_{name}_xyz.npz",xyz=candidate.GetConformer().GetPositions())
    except Exception as exc:result.update(status='scoring_error',error=str(exc))
    write(path,result);return result

def summarize(rows):
    scored=[r for r in rows if r['status']=='scored'];out=dict(attempts=len(rows),valid_count=sum(r['valid'] for r in rows),
        connected_count=sum(r['connected'] for r in rows),scored_count=len(scored),errors=[r for r in rows if r['status']=='scoring_error'])
    for stage in ['baseline','selected']:
        out[stage]={k:dict(mean=float(np.mean([r[stage][k] for r in scored])) if scored else None,count=len(scored)) for k in ['vina','strain','clashes']}
    out['selected_changed']=sum(r['selected_source']!='input' for r in scored)
    out['candidate_errors']=sum('error' in t for r in rows for t in r['proposals'])
    for key in ['qed','aromatic_rings']:
        vals=[r[key] for r in rows if key in r];out[key]=dict(mean=float(np.mean(vals)) if vals else None,count=len(vals))
    return out

def compare(a):
    cache=Path(a.cache);manifest=json.loads((cache/'manifest.json').read_text());entries=manifest['files'][:a.pockets]
    if len(entries)!=a.pockets:raise ValueError('Requested pocket count not cached')
    out=confined(a.out)
    if a.resume:
        meta=json.loads((out/'run_metadata.json').read_text())
        if meta['manifest_sha256']!=sha(cache/'manifest.json') or meta['script_sha256']!=sha(__file__) or meta['pockets']!=a.pockets or meta['strain_steps']!=a.strain_steps:raise ValueError('Resume inputs or policy changed')
    else:new_directory(out)
    (out/'tasks').mkdir(exist_ok=True)
    for e in entries:
        if sha(cache/e['name'])!=e['sha256']:raise ValueError('Cache file changed: '+e['name'])
    write(out/'run_metadata.json',dict(cache=str(cache.resolve()),variant=manifest['variant'],pockets=a.pockets,workers=a.workers,
        manifest_sha256=sha(cache/'manifest.json'),script_sha256=sha(__file__),strain_steps=a.strain_steps,
        docking=dict(cpu=1,exhaustiveness=4,box=[20,20,20],seed='pocket seed + sample + 10000'),
        candidate_policy='whole/local + two intermediates, then local + two torsions; graph unchanged; each selection nonworsening in three metrics',scope='new cached ablation development cohort; no merge into historical table'))
    rows=[];tasks=[]
    for e in entries:
        for j in range(manifest['samples']):
            saved=out/'tasks'/f"p{e['index']}_s{j}.json"
            if a.resume and saved.exists():rows.append(json.loads(saved.read_text()))
            else:tasks.append((str(cache),str(out),e,j,manifest['variant'],a.strain_steps))
    def save():
        rows.sort(key=lambda r:(r['pocket'],r['sample']));write(out/'samples.json',rows);write(out/'summary.json',summarize(rows))
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for future in as_completed([pool.submit(worker,t) for t in tasks]):
            rows.append(future.result());save();print(f'Completed {len(rows)}/{a.pockets*manifest["samples"]} molecules',flush=True)
    save();print(json.dumps(summarize(rows),indent=2),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    g=sub.add_parser('generate');g.add_argument('--out',required=True);g.add_argument('--checkpoint',default=str(PROJECT_ROOT/'checkpoints/sbdd/latest_sbdd_model.pt'));g.add_argument('--lmdb-dir');g.add_argument('--device',default='cuda');g.add_argument('--variant',type=int,choices=range(1,6),default=5);g.add_argument('--pockets',type=int,default=20);g.add_argument('--samples',type=int,default=5);g.add_argument('--seed',type=int,default=42)
    c=sub.add_parser('compare');c.add_argument('--cache',required=True);c.add_argument('--out',required=True);c.add_argument('--pockets',type=int,default=20);c.add_argument('--workers',type=int,default=12);c.add_argument('--strain-steps',type=int,choices=[200,300],default=200);c.add_argument('--resume',action='store_true')
    a=p.parse_args()
    if getattr(a,'pockets',1)<1 or getattr(a,'workers',1)<1 or getattr(a,'samples',1)<1:p.error('Counts must be positive')
    {'generate':generate,'compare':compare}[a.command](a)
if __name__=='__main__':main()
