"""Parallel/resumable execution of the same fixed topology trial."""
import os
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ[name]='1'
import argparse, hashlib, importlib.util, json, sys, time, multiprocessing
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import QED,RDConfig
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from project_build.postprocess.bond_perception import BondPerceptionEngine
from project_build.postprocess.pocket_geometry_bounded_trial import refine_in_pocket
from project_build.postprocess.repair_bridge_trial_frozen import repair_with_terminal_geometry
VARIANTS=['original','pocket_refine']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic_json(path,value):
    temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value,indent=2));os.replace(temp,path)
def initialize(dock):
    global MASTER,SASCORER,Vina
    torch.set_num_threads(1)
    sys.path.append(str(Path(RDConfig.RDContribDir)/'SA_Score'));import sascorer
    SASCORER=sascorer;MASTER=None
    if dock:
        spec=importlib.util.spec_from_file_location('master',ROOT/'scripts/evaluate_official_vina_large_scale.py')
        MASTER=importlib.util.module_from_spec(spec);spec.loader.exec_module(MASTER)
        from vina import Vina

def one_sample(task):
    cache,report,entry,j=task;cache=Path(cache);report=Path(report)
    file=cache/entry['name'];assert sha(file)==entry['sha256']
    with np.load(file) as z:x,h,pp,ph=[z[k] for k in ['x','h','pocket_pos','pocket_h']]
    types=ph.argmax(-1);heavy=types>0;nums=np.array([1,6,7,8,16,34])[types[heavy]]
    master=MASTER;sascorer=SASCORER
    receptor=report/f"receptor_{entry['index']}_s{j}.pdbqt"
    if master:master.write_pocket_pdbqt_file(pp,types,str(receptor))
    mol,valid,nf=BondPerceptionEngine.reconstruct_mol(torch.from_numpy(x[j]),torch.from_numpy(h[j]))
    repair_info={'reason':'original_accepted'}
    if not valid:
        repaired,repair_info=repair_with_terminal_geometry(mol,pp[heavy],nums)
        if repaired is not None:mol,valid=repaired,True
    candidate,refinement=refine_in_pocket(mol,pp[heavy],nums) if valid else (mol,dict(accepted=False,reason='invalid_baseline'))
    # MMFF energy decrease is not identical to the reported UFF strain change.
    # Explicit same-definition acceptance gate; keep both values in diagnostics.
    if master and refinement.get('accepted'):
        before_strain=master.compute_clean_strain_energy(mol)
        after_strain=master.compute_clean_strain_energy(candidate)
        refinement.update(strain_gate_before=before_strain,strain_gate_candidate=after_strain)
        if after_strain>=before_strain:
            candidate=Chem.Mol(mol)
            refinement.update(accepted=False,reason='reported_strain_not_lower')
    score_cache={};rows=[]
    for name,m in [('original',mol),('pocket_refine',candidate)]:
        row=dict(pocket=entry['index'],sample=j,variant=name,valid=bool(valid),connected=m is not None and len(Chem.GetMolFrags(m))==1,repair=repair_info)
        if name=='pocket_refine':row['refinement']=refinement
        try:
            xyz=m.GetConformer().GetPositions() if m is not None else x[j]
            row['clashes']=int((np.linalg.norm(xyz[:,None]-pp[heavy][None,:],axis=-1)<1.5).sum())
            if valid:
                smiles=Chem.MolToSmiles(m);rings=m.GetRingInfo().AtomRings()
                row.update(smiles=smiles,qed=QED.qed(m),sa=(10-sascorer.calculateScore(m))/9,ring3=sum(len(r)==3 for r in rings),ring4=sum(len(r)==4 for r in rings),graph_changed_vs_baseline=False)
                if name=='pocket_refine':
                    assert smiles==rows[0]['smiles']
                    assert abs(row['qed']-rows[0]['qed'])<1e-10 and abs(row['sa']-rows[0]['sa'])<1e-10
                if master:
                    pdbqt=master.mol_to_pdbqt_meeko(m)
                    signature=hashlib.sha256(smiles.encode()+xyz.tobytes()).hexdigest()
                    if signature not in score_cache:
                        v=Vina(sf_name='vina',cpu=1,seed=entry['seed']+j+10000,verbosity=0)
                        v.set_receptor(str(receptor));v.set_ligand_from_string(pdbqt)
                        v.compute_vina_maps(center=pp.mean(0).tolist(),box_size=[20.,20.,20.]);v.dock(exhaustiveness=4,n_poses=1)
                        score_cache[signature]=(float(v.energies(n_poses=1)[0][0]),master.compute_clean_strain_energy(m))
                    row['vina'],row['strain']=score_cache[signature]
            if m is not None:(report/f"p{entry['index']}_s{j}_{name}.mol").write_text(Chem.MolToMolBlock(m,kekulize=False))
        except AssertionError:raise
        except Exception as exc:row['error']=str(exc)
        rows.append(row)
    result=dict(rows=rows,baseline_checks=1)
    atomic_json(report/'tasks'/f"p{entry['index']}_s{j}.json",result)
    return result

def summarize(rows):
    summary={}
    for w in VARIANTS:
        rr=[r for r in rows if r['variant']==w];n=len(rr);smiles=[r['smiles'] for r in rr if 'smiles' in r]
        s=dict(attempts=n,valid_pct=100*sum(r['valid'] for r in rr)/n,connectivity_pct=100*sum(r['connected'] for r in rr)/n,
               uniqueness_pct=100*len(set(smiles))/len(smiles) if smiles else None,
               changed_graphs=sum(r.get('graph_changed_vs_baseline',False) for r in rr),errors=[r for r in rr if 'error' in r])
        for k in ['qed','sa','clashes','vina','strain','ring3','ring4']:
            values=[r[k] for r in rr if k in r];s[k]=dict(mean=float(np.mean(values)) if values else None,count=len(values))
        base={(r['pocket'],r['sample']):r for r in rows if r['variant']=='original'}
        delta=[r['vina']-base[(r['pocket'],r['sample'])]['vina'] for r in rr if 'vina' in r and 'vina' in base[(r['pocket'],r['sample'])]]
        s['paired_vina_delta_candidate_minus_baseline']=dict(mean=float(np.mean(delta)) if delta else None,count=len(delta))
        ds=[r['strain']-base[(r['pocket'],r['sample'])]['strain'] for r in rr if 'strain' in r and 'strain' in base[(r['pocket'],r['sample'])]]
        s['paired_strain_delta_candidate_minus_baseline']=dict(mean=float(np.mean(ds)) if ds else None,count=len(ds))
        s['accepted_refinements']=sum(r.get('refinement',{}).get('accepted',False) for r in rr)
        s['optimizer_converged']=sum(r.get('refinement',{}).get('converged',False) for r in rr)
        summary[str(w)]=s
    return summary

def groups(rows):
    result={}
    for r in rows:result.setdefault((r['pocket'],r['sample']),[]).append(r)
    return {k:v for k,v in result.items() if len(v)==2 and {r['variant'] for r in v}==set(VARIANTS)}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cache',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--dock',action='store_true');ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--pockets',type=int,default=10)
    ap.add_argument('--resume',action='store_true');ap.add_argument('--snapshot-only',action='store_true')
    a=ap.parse_args();cache=Path(a.cache);out=Path(a.out);report=out/('comparison_docked' if a.dock else 'comparison')
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    if a.snapshot_only:
        for attempt in range(20):
            try:
                rows=json.loads((report/'samples.json').read_text());break
            except (ValueError,FileNotFoundError):time.sleep(.1)
        else:raise RuntimeError('No readable progress file yet. Keep original job running and try later.')
        path=report/f'serial_snapshot_{stamp}.json';atomic_json(path,rows)
        print('Saved snapshot:',path,'completed:',len(groups(rows)));return
    if a.workers<1:raise ValueError('--workers must be positive')
    manifest=json.loads((cache/'manifest.json').read_text());entries=manifest['files'][:a.pockets]
    if a.pockets<1 or len(entries)!=a.pockets:raise ValueError('Requested pocket count unavailable in cache')
    allowed={(e['index'],j) for e in entries for j in range(manifest['args']['samples'])}
    complete={};resumed_checks=0
    if a.resume:
        if not report.exists():raise ValueError('Resume report not found')
        paths=[report/'samples.json']+sorted(report.glob('serial_snapshot_*.json'),reverse=True)
        for path in paths:
            if not path.exists():continue
            try:
                raw=json.loads(path.read_text());complete=groups(raw)
                atomic_json(report/f'progress_before_parallel_{stamp}.json',raw);break
            except ValueError:continue
        for path in (report/'tasks').glob('*.json'):
            data=json.loads(path.read_text());complete.update(groups(data['rows']))
        if not set(complete)<=allowed:raise ValueError('Saved samples do not match selected cache indices')
    else:
        out.mkdir(parents=True,exist_ok=True)
        if any(out.iterdir()):raise ValueError('Use --resume for existing results')
        report.mkdir()
    (report/'tasks').mkdir(exist_ok=True)
    resumed=len(complete);checks=0
    tasks=[(str(cache),str(report),e,j) for e in entries for j in range(manifest['args']['samples']) if (e['index'],j) not in complete]
    def all_rows():
        return [r for k in sorted(complete) for r in sorted(complete[k],key=lambda r:VARIANTS.index(r['variant']))]
    print(f'Resumed {resumed}/{len(allowed)} molecules; pending {len(tasks)}; workers={a.workers}',flush=True)
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn'),initializer=initialize,initargs=(a.dock,)) as pool:
        futures=[pool.submit(one_sample,t) for t in tasks]
        for future in as_completed(futures):
            data=future.result();complete.update(groups(data['rows']));checks+=data['baseline_checks']
            atomic_json(report/'samples.json',all_rows())
            print(f'Completed {len(complete)}/{len(allowed)} molecules (three variants)',flush=True)
    rows=all_rows();atomic_json(report/'summary.json',summarize(rows))
    atomic_json(report/'run_metadata.json',dict(scope='Same fixed cached development comparison',workers=a.workers,
        resumed_molecules=resumed,baseline_reconstructions_this_execution=checks,docking=a.dock,
        source_split=manifest['args']['split'],cache=str(cache),pockets=a.pockets,script_sha256=sha(__file__),
        refinement_sha256=sha(ROOT/'project_build/postprocess/pocket_geometry_bounded_trial.py'),
        repair_sha256=sha(ROOT/'project_build/postprocess/repair_bridge_trial_frozen.py')))
    print('Finished. Results:',report,flush=True)

if __name__=='__main__':main()
