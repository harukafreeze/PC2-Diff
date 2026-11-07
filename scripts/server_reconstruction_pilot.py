"""Generate once, compare reconstruction on immutable cached coordinates."""
import argparse, hashlib, importlib.util, json, random, sys
from pathlib import Path
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import QED, RDConfig

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from project_build.postprocess.bond_perception import BondPerceptionEngine
from project_build.postprocess.repair_experiment import repair_failed, repair_with_terminal_geometry, diagnostics

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['generate','compare'])
    ap.add_argument('--out', required=True)
    ap.add_argument('--checkpoint', default='checkpoints/sbdd/latest_sbdd_model.pt')
    ap.add_argument('--lmdb-dir')
    ap.add_argument('--split', choices=['train','test'], default='train')
    ap.add_argument('--pockets', type=int, default=10)
    ap.add_argument('--samples', type=int, default=10)
    ap.add_argument('--start-index', type=int, default=0)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--dock', action='store_true', help='Full original-protocol docking for both arms')
    ap.add_argument('--report-name', help='New comparison subdirectory, preserving previous reports')
    ap.add_argument('--geometry-repair', action='store_true', help='Enable V3 terminal-halogen geometry fallback')
    args = ap.parse_args(); out = Path(args.out)
    if args.stage == 'generate':
        out.mkdir(parents=True, exist_ok=True)
        if any(out.iterdir()):
            raise RuntimeError('Output directory must be empty; use a new run directory.')
        import project_build.models.backbones, project_build.physics
        from project_build.core.registry import BACKBONE_REGISTRY
        from project_build.data.datasets.crossdocked import CrossDockedLMDBDataset
        from project_build.diffusion.sde.vp_sde import DualSpaceVPSDE
        from project_build.diffusion.solvers.reverse_sde_sampler import ReverseSDESampler
        ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
        cfg = ckpt['config']; model = BACKBONE_REGISTRY.build(cfg['model']).to(args.device)
        model.load_state_dict(ckpt['ema_state_dict'] if 'ema_state_dict' in ckpt else ckpt['model_state_dict']); model.eval()
        data = CrossDockedLMDBDataset(lmdb_dir=args.lmdb_dir or cfg['dataset']['lmdb_dir'], split=args.split, split_type='name')
        if args.start_index < 0 or args.start_index + args.pockets > len(data):
            raise ValueError('Requested pocket range is outside the selected split.')
        sde = DualSpaceVPSDE(schedule_type='cosine').to(args.device)
        sampler = ReverseSDESampler(sde=sde, num_steps=100, spring_k=0.25, corrector_steps=1,
                     bond_gate_thresh=0.35, enable_svd_seeding=True, enable_self_healing=True, device=args.device)
        manifest = dict(args=vars(args), checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
                        engine_sha256=hashlib.sha256((ROOT/'project_build/postprocess/bond_perception.py').read_bytes()).hexdigest(),files=[])
        for idx in range(args.start_index, args.start_index + args.pockets):
            seed = args.seed + idx; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
            s = data[idx]; pp = s['pocket_pos'].unsqueeze(0).to(args.device); ph = s['pocket_h'].unsqueeze(0).to(args.device)
            with torch.no_grad():
                x,h,_ = sampler.sample(model=model,num_samples=args.samples,num_atoms=max(18,s['num_lig_atoms']),num_atom_types=9,
                       model_kwargs={'pocket_pos':pp.repeat(args.samples,1,1),'pocket_h':ph.repeat(args.samples,1,1)})
            name = f'pocket_{idx:05d}.npz'; path = out/name
            np.savez_compressed(path,x=x.cpu().numpy(),h=h.cpu().numpy(),pocket_pos=pp[0].cpu().numpy(),pocket_h=ph[0].cpu().numpy())
            manifest['files'].append(dict(name=name,index=idx,key=str(data.keys[idx]),seed=seed,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
            print(f'Saved {name}',flush=True)
        return

    manifest = json.loads((out/'manifest.json').read_text())
    if len(manifest['files']) != manifest['args']['pockets']:
        raise RuntimeError('Generation incomplete; do not summarize a partial run.')
    if hashlib.sha256((ROOT/'project_build/postprocess/bond_perception.py').read_bytes()).hexdigest()!=manifest['engine_sha256']:
        raise RuntimeError('Reconstruction engine changed since generation.')
    if args.report_name and Path(args.report_name).name != args.report_name:
        raise ValueError('report-name must be a single directory name')
    report = out/(args.report_name or ('comparison_docked' if args.dock else 'comparison'))
    report.mkdir(exist_ok=False)
    (report/'run_metadata.json').write_text(json.dumps(dict(args=vars(args),
        repair_sha256=hashlib.sha256((ROOT/'project_build/postprocess/repair_experiment.py').read_bytes()).hexdigest()),indent=2))
    # Import the original evaluator only when docking is requested.
    master = None
    if args.dock:
        spec = importlib.util.spec_from_file_location('master', ROOT/'scripts/evaluate_official_vina_large_scale.py')
        master = importlib.util.module_from_spec(spec); spec.loader.exec_module(master)
        from vina import Vina
    sys.path.append(str(Path(RDConfig.RDContribDir)/'SA_Score')); import sascorer
    totals = {v:dict(attempts=0,valid=0,connected=0,smiles=[],qed=[],sa=[],vina=[],strain=[],clashes=[],failures=[]) for v in ['original','repair']}
    records=[]
    for entry in manifest['files']:
        path=out/entry['name']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['sha256']
        with np.load(path) as z:
            x,h,pp,ph=[z[k] for k in ['x','h','pocket_pos','pocket_h']]
        assert len(x)==len(h)==manifest['args']['samples']
        receptor=report/'receptor.pdbqt'
        if master: master.write_pocket_pdbqt_file(pp,ph.argmax(-1),str(receptor))
        for j in range(len(x)):
            record=dict(pocket=entry['index'],sample=j)
            try:
                mol,valid,nf=BondPerceptionEngine.reconstruct_mol(torch.from_numpy(x[j]),torch.from_numpy(h[j]),tolerance=.35)
                err=None
            except Exception as exc:
                mol,valid,nf=None,False,0;err=str(exc)
            record.update(original_valid=bool(valid),original_error=err,chemistry_problems=diagnostics(mol))
            # Never repair accepted molecules or reduce the validity standard.
            if valid:
                repaired,info=mol,{'reason':'original_accepted'}
            elif args.geometry_repair:
                protein_map=np.array([1,6,7,8,16,34]);protein_types=ph.argmax(-1);heavy=protein_types>0
                repaired,info=repair_with_terminal_geometry(mol,pp[heavy],protein_map[protein_types[heavy]])
            else:
                repaired,info=repair_failed(mol)
            record['repair']=info
            for variant,m,ok in [('original',mol,bool(valid)),('repair',repaired if repaired is not None else mol,bool(valid or repaired is not None))]:
                t=totals[variant];t['attempts']+=1;t['valid']+=int(ok)
                t['connected']+=int(m is not None and len(Chem.GetMolFrags(m))==1)
                actual_xyz=m.GetConformer().GetPositions() if m is not None else x[j]
                t['clashes'].append(int((np.linalg.norm(actual_xyz[:,None]-pp[ph.argmax(-1)>0][None,:],axis=-1)<1.5).sum()))
                record.setdefault('coordinate_changes',{})[variant]=float(np.linalg.norm(actual_xyz-x[j],axis=1).max())
                if m is not None:
                    try:
                        (report/f"p{entry['index']}_s{j}_{variant}.mol").write_text(Chem.MolToMolBlock(m,kekulize=False))
                    except Exception as exc:
                        record.setdefault('export_errors',[]).append(str(exc))
                if not ok: continue
                try:
                    t['smiles'].append(Chem.MolToSmiles(m));t['qed'].append(QED.qed(m));t['sa'].append((10-sascorer.calculateScore(m))/9)
                except Exception as exc:t['failures'].append(dict(pocket=entry['index'],sample=j,stage='descriptors',error=str(exc)))
                if master:
                    try:
                        # Same seed per paired sample, same box/exhaustiveness as original script.
                        v=Vina(sf_name='vina',cpu=1,seed=manifest['args']['seed']+entry['index']*10000+j,verbosity=0)
                        v.set_receptor(str(receptor));v.set_ligand_from_string(master.mol_to_pdbqt_meeko(m))
                        v.compute_vina_maps(center=pp.mean(0).tolist(),box_size=[20.,20.,20.]);v.dock(exhaustiveness=4,n_poses=1)
                        t['vina'].append(float(v.energies(n_poses=1)[0][0]));t['strain'].append(master.compute_clean_strain_energy(m))
                    except Exception as exc:t['failures'].append(dict(pocket=entry['index'],sample=j,stage='docking_or_strain',error=str(exc)))
            records.append(record)
        print(f"Compared {entry['name']}",flush=True)
    summary={}
    for variant,t in totals.items():
        n=t['attempts'];summary[variant]=dict(attempts=n,valid_count=t['valid'],valid_pct=100*t['valid']/n,
            connectivity_pct=100*t['connected']/n,uniqueness_pct=100*len(set(t['smiles']))/len(t['smiles']) if t['smiles'] else None,
            descriptor_smiles_count=len(t['smiles']),failures=t['failures'])
        for k in ['qed','sa','vina','strain','clashes']:
            summary[variant][k]=dict(mean=float(np.mean(t[k])) if t[k] else None,count=len(t[k]))
    (report/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    (report/'sample_diagnostics.json').write_text(json.dumps(records,indent=2),encoding='utf8')
    (report/'raw_metric_lists.json').write_text(json.dumps(totals,indent=2),encoding='utf8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
