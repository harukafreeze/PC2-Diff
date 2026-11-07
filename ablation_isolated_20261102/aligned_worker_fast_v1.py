import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
import sys,time
sys.dont_write_bytecode=True
from pathlib import Path
import numpy as np
from rdkit import Chem
from rdkit.Chem import QED,rdMolDescriptors
from run import torch,legacy,strain,write,whole_refine,local_refine
from trial_modules.aligned_intermediates import intermediate_candidates
from trial_modules.aligned_torsions import local_candidates

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
    timings=dict(optimizer=0.,uff=0.,docking=0.,skipped_docking=0,metric_cache_hits=0,local_optimizer_reuse=0)
    metric_cache={}
    def timed_refine(fn,*args):
        started=time.perf_counter()
        try:return fn(*args)
        finally:timings['optimizer']+=time.perf_counter()-started
    def metrics(candidate,reference=None):
        assert Chem.MolToSmiles(candidate)==graph
        xyz=candidate.GetConformer().GetPositions()
        key=xyz.tobytes()
        if key in metric_cache:
            timings['metric_cache_hits']+=1;return dict(metric_cache[key])
        clashes=int((np.linalg.norm(xyz[:,None]-pp[heavy][None],axis=-1)<1.5).sum())
        if reference is not None and clashes>reference['clashes']:
            timings['skipped_docking']+=1;return dict(clashes=clashes,skipped='more_contacts')
        started=time.perf_counter()
        energy=strain(candidate,steps)
        timings['uff']+=time.perf_counter()-started
        if reference is not None and energy>reference['strain']+1e-8:
            timings['skipped_docking']+=1;return dict(clashes=clashes,strain=energy,skipped='strain_increased')
        started=time.perf_counter()
        ligand=legacy.mol_to_pdbqt_meeko_safe(candidate)
        if not ligand:raise ValueError('Ligand PDBQT preparation failed')
        v=legacy.Vina(sf_name='vina',cpu=1,seed=entry['seed']+j+10000,verbosity=0)
        v.set_receptor(str(rec));v.set_ligand_from_string(ligand);v.compute_vina_maps(center=pp.mean(0).tolist(),box_size=[20.,20.,20.])
        v.dock(exhaustiveness=4,n_poses=1);score=float(v.energies(n_poses=1)[0][0])
        if not np.isfinite(score):raise ValueError('Nonfinite Vina score')
        timings['docking']+=time.perf_counter()-started
        value=dict(vina=score,strain=energy,clashes=clashes);metric_cache[key]=value
        return dict(value)
    def eligible(candidate,ref):return all(candidate[k]<=ref[k]+1e-8 for k in ['vina','strain','clashes'])
    def evaluate(candidate,name,ref,info):
        try:
            m=metrics(candidate,ref)
            if 'skipped' in m:
                result['proposals'].append(dict(source=name,metrics=m,eligible=False,reason=m['skipped'],details=info));return None
            ok=eligible(m,ref) and (m['vina']<ref['vina']-1e-6 if name=='local' or name.startswith('intermediate_') else True);result['proposals'].append(dict(source=name,metrics=m,eligible=ok,details=info))
            return (candidate,m,name) if ok else None
        except Exception as exc:
            result['proposals'].append(dict(source=name,error=str(exc),details=info));return None
    try:
        base=metrics(mol);result['baseline']=base;best=(mol,base,'input')
        if frags==1:
            pool=[best];whole,wi=timed_refine(whole_refine,mol,pp[heavy],nums)
            local,li=timed_refine(local_refine,mol,pp[heavy],nums)
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
            anchor=best;pool=[anchor]
            if np.array_equal(anchor[0].GetConformer().GetPositions(),mol.GetConformer().GetPositions()):
                local,info=local,li;timings['local_optimizer_reuse']+=1
            else:local,info=timed_refine(local_refine,anchor[0],pp[heavy],nums)
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
    result['timing_seconds']=timings
    write(path,result);return result
