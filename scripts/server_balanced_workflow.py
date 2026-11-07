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
from project_build.postprocess.torsion_recovery_trial import candidate_strain
from project_build.postprocess.intermediate_pose_trial import intermediate_candidates
from project_build.postprocess.contact_anchored_local_trial import refine_in_pocket
from project_build.postprocess.balanced_selection_trial import choose_geometry, choose_intermediate
from project_build.postprocess.repair_bridge_trial_frozen import repair_with_terminal_geometry
VARIANTS=['original','balanced']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic_json(path,value):
    temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value,indent=2));os.replace(temp,path)
def initialize(dock,baseline):
    global MASTER,SASCORER,Vina,SOURCE_ROWS,GLOBAL_ROWS,BASELINE_DIR
    BASELINE_DIR=Path(baseline);data=json.loads((BASELINE_DIR/'samples.json').read_text())
    SOURCE_ROWS={(r['pocket'],r['sample']):r for r in data if r['variant']=='original'}
    GLOBAL_ROWS={(r['pocket'],r['sample']):r for r in data if r['variant']=='pocket_refine'}
    torch.set_num_threads(1)
    sys.path.append(str(Path(RDConfig.RDContribDir)/'SA_Score'));import sascorer
    SASCORER=sascorer
    spec=importlib.util.spec_from_file_location('master',ROOT/'scripts/evaluate_official_vina_large_scale.py')
    MASTER=importlib.util.module_from_spec(spec);spec.loader.exec_module(MASTER)
    from vina import Vina

def one_sample(task):
    cache,report,entry,j=task;cache=Path(cache);report=Path(report)
    file=cache/entry['name'];assert sha(file)==entry['sha256']
    with np.load(file) as z:x,h,pp,ph=[z[k] for k in ['x','h','pocket_pos','pocket_h']]
    types=ph.argmax(-1);heavy=types>0;nums=np.array([1,6,7,8,16,34])[types[heavy]]
    key=entry['index'],j;baseline=dict(SOURCE_ROWS[key],source='original')
    raw,valid,_=BondPerceptionEngine.reconstruct_mol(torch.from_numpy(x[j]),torch.from_numpy(h[j]))
    if not valid:
        fixed,_=repair_with_terminal_geometry(raw,pp[heavy],nums)
        if fixed is not None:raw,valid=fixed,True
    assert valid and Chem.MolToSmiles(raw)==baseline['smiles']
    saved=Chem.MolFromMolBlock((BASELINE_DIR/f"p{entry['index']}_s{j}_pocket_refine.mol").read_text(),sanitize=False,removeHs=False)
    assert saved.GetNumAtoms()==raw.GetNumAtoms()
    endpoint=Chem.Mol(raw)
    for i,pos in enumerate(saved.GetConformer().GetPositions()):endpoint.GetConformer().SetAtomPosition(i,tuple(float(a) for a in pos))
    global_row=dict(GLOBAL_ROWS[key],source='global');assert global_row['smiles']==baseline['smiles']
    molecules={'original':raw,'global':endpoint};candidates=[]
    receptor=report/f"receptor_{entry['index']}_s{j}.pdbqt"
    MASTER.write_pocket_pdbqt_file(pp,types,str(receptor))
    def evaluate(mol,source,reference,details):
        trial=dict(source=source,details=details)
        try:
            assert Chem.MolToSmiles(mol)==baseline['smiles']
            xyz=mol.GetConformer().GetPositions();contacts=int((np.linalg.norm(xyz[:,None]-pp[heavy][None,:],axis=-1)<1.5).sum())
            trial['clashes']=contacts
            if contacts>reference['clashes']:trial['reason']='more_contacts';return None,trial
            strain=candidate_strain(mol);trial['strain']=strain
            if strain>reference['strain']+1e-8:trial['reason']='strain_increased';return None,trial
            assert abs(QED.qed(mol)-baseline['qed'])<1e-10
            assert abs((10-SASCORER.calculateScore(mol))/9-baseline['sa'])<1e-10
            v=Vina(sf_name='vina',cpu=1,seed=entry['seed']+j+10000,verbosity=0)
            v.set_receptor(str(receptor));v.set_ligand_from_string(MASTER.mol_to_pdbqt_meeko(mol))
            v.compute_vina_maps(center=pp.mean(0).tolist(),box_size=[20.,20.,20.]);v.dock(exhaustiveness=4,n_poses=1)
            score=float(v.energies(n_poses=1)[0][0]);trial['vina']=score
            if score>=reference['vina']-1e-6:trial['reason']='vina_not_lower';return None,trial
            trial['reason']='eligible'
            row=dict(baseline,source=source,vina=score,strain=strain,clashes=contacts)
            molecules[source]=mol
            return row,trial
        except AssertionError:raise
        except Exception as exc:trial.update(reason='candidate_error',error=str(exc));return None,trial
    # Stage A exactly follows the already-explored original/global/local pool.
    alternatives=[global_row]
    local,info=refine_in_pocket(raw,pp[heavy],nums)
    if info.get('accepted'):
        row,trial=evaluate(local,'local',baseline,info);candidates.append(trial)
        if row is not None:alternatives.append(row)
    else:candidates.append(dict(source='local',reason='optimization_returned_original',details=info))
    stage_a=choose_geometry(baseline,alternatives)
    # Stage B uses exactly the two already-fixed intermediate proposals.
    middle=[]
    for i,(mol,info) in enumerate(intermediate_candidates(raw,endpoint.GetConformer().GetPositions(),pp[heavy])):
        row,trial=evaluate(mol,f'intermediate_{i}',stage_a,info);candidates.append(trial)
        if row is not None:middle.append(row)
    selected=choose_intermediate(stage_a,middle)
    original=dict(baseline,variant='original');original.pop('refinement',None);original.pop('recovery',None)
    best=dict(selected,variant='balanced');best.pop('refinement',None);best.pop('recovery',None)
    best['selection']=dict(source=selected['source'],stage_a_source=stage_a['source'],
        stage_a_metrics={k:stage_a[k] for k in ['vina','strain','clashes']},candidates=candidates)
    assert best['vina']<=original['vina']+1e-8 and best['strain']<=original['strain']+1e-8 and best['clashes']<=original['clashes']
    winner=molecules[selected['source']]
    (report/f"p{entry['index']}_s{j}_balanced.mol").write_text(Chem.MolToMolBlock(winner))
    np.savez_compressed(report/f"p{entry['index']}_s{j}_balanced_coordinates.npz",xyz=winner.GetConformer().GetPositions())
    result=dict(rows=[original,best],baseline_checks=1)
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
        s['selected_nonoriginal']=sum(r.get('selection',{}).get('source','original')!='original' for r in rr)
        s['optimizer_converged']=sum(r.get('refinement',{}).get('converged',False) for r in rr)
        summary[str(w)]=s
    return summary

def groups(rows):
    result={}
    for r in rows:result.setdefault((r['pocket'],r['sample']),[]).append(r)
    return {k:v for k,v in result.items() if len(v)==2 and {r['variant'] for r in v}==set(VARIANTS)}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cache',required=True);ap.add_argument('--out',required=True);ap.add_argument('--baseline',required=True)
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
    if not a.dock:raise ValueError('This trial requires --dock')
    baseline_meta=json.loads((Path(a.baseline)/'run_metadata.json').read_text())
    if Path(baseline_meta['cache']).resolve()!=cache.resolve():
        raise ValueError('Baseline and requested cache refer to different runs')
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
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn'),initializer=initialize,initargs=(a.dock,a.baseline)) as pool:
        futures=[pool.submit(one_sample,t) for t in tasks]
        for future in as_completed(futures):
            data=future.result();complete.update(groups(data['rows']));checks+=data['baseline_checks']
            atomic_json(report/'samples.json',all_rows())
            print(f'Completed {len(complete)}/{len(allowed)} molecules (three variants)',flush=True)
    rows=all_rows();atomic_json(report/'summary.json',summarize(rows))
    atomic_json(report/'run_metadata.json',dict(scope='Same fixed cached development comparison',workers=a.workers,
        resumed_molecules=resumed,baseline_reconstructions_this_execution=checks,docking=a.dock,
        source_split=manifest['args']['split'],cache=str(cache),pockets=a.pockets,script_sha256=sha(__file__),
        candidate_generator_sha256=sha(ROOT/'project_build/postprocess/intermediate_pose_trial.py'),local_optimizer_sha256=sha(ROOT/'project_build/postprocess/contact_anchored_local_trial.py'),selector_sha256=sha(ROOT/'project_build/postprocess/balanced_selection_trial.py'),baseline=str(a.baseline),baseline_samples_sha256=sha(Path(a.baseline)/'samples.json'),baseline_metrics_reused=True,
        repair_sha256=sha(ROOT/'project_build/postprocess/repair_bridge_trial_frozen.py')))
    print('Finished. Results:',report,flush=True)

if __name__=='__main__':main()
