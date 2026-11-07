"""Bounded second pass from saved balanced coordinates; three proposals/sample."""
import os
for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[name] = '1'
import argparse
import json
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import QED

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
import server_balanced_workflow as previous
from project_build.postprocess.contact_anchored_local_trial import refine_in_pocket
from project_build.postprocess.torsion_recovery_trial import local_candidates, candidate_strain
from project_build.postprocess.balanced_selection_trial import eligible
from project_build.postprocess.bond_perception import BondPerceptionEngine
from project_build.postprocess.repair_bridge_trial_frozen import repair_with_terminal_geometry


def initialize(source):
    global SOURCE, ROWS
    SOURCE = Path(source)
    metadata = json.loads((SOURCE / 'run_metadata.json').read_text())
    previous.initialize(True, metadata['baseline'])
    ROWS = {(r['pocket'], r['sample']): r for r in
            json.loads((SOURCE / 'samples.json').read_text()) if r['variant'] == 'balanced'}


def one_sample(task):
    cache, report, entry, j = task
    cache, report = Path(cache), Path(report)
    assert previous.sha(cache / entry['name']) == entry['sha256']
    with np.load(cache / entry['name']) as z:
        x, h, pp, ph = [z[k] for k in ['x', 'h', 'pocket_pos', 'pocket_h']]
    types = ph.argmax(-1)
    heavy = types > 0
    numbers = np.array([1, 6, 7, 8, 16, 34])[types[heavy]]
    base = dict(ROWS[entry['index'], j])
    mol, valid, _ = BondPerceptionEngine.reconstruct_mol(torch.from_numpy(x[j]), torch.from_numpy(h[j]))
    if not valid:
        fixed, _ = repair_with_terminal_geometry(mol, pp[heavy], numbers)
        if fixed is not None:
            mol, valid = fixed, True
    assert valid and Chem.MolToSmiles(mol) == base['smiles']
    with np.load(SOURCE / f"p{entry['index']}_s{j}_balanced_coordinates.npz") as z:
        xyz = z['xyz']
    assert xyz.shape == (mol.GetNumAtoms(), 3) and np.isfinite(xyz).all()
    for i, pos in enumerate(xyz):
        mol.GetConformer().SetAtomPosition(i, tuple(float(v) for v in pos))
    receptor = report / f"receptor_{entry['index']}_s{j}.pdbqt"
    previous.MASTER.write_pocket_pdbqt_file(pp, types, str(receptor))
    winner, best = mol, base
    diagnostics = []
    proposals = []
    local, info = refine_in_pocket(mol, pp[heavy], numbers)
    if info.get('accepted'):
        proposals.append(('local', local, info))
    else:
        diagnostics.append(dict(source='local', reason='optimizer_returned_input', details=info))
    proposals.extend((f'torsion_{i}', m, info) for i, (m, info) in enumerate(local_candidates(mol)))
    for source, candidate, info in proposals:
        trial = dict(source=source, details=info)
        try:
            assert Chem.MolToSmiles(candidate) == base['smiles']
            assert abs(QED.qed(candidate) - base['qed']) < 1e-10
            assert abs((10 - previous.SASCORER.calculateScore(candidate)) / 9 - base['sa']) < 1e-10
            coords = candidate.GetConformer().GetPositions()
            clashes = int((np.linalg.norm(coords[:, None] - pp[heavy][None], axis=-1) < 1.5).sum())
            trial['clashes'] = clashes
            if clashes > base['clashes']:
                trial['reason'] = 'more_contacts'
                continue
            strain = candidate_strain(candidate)
            trial['strain'] = strain
            if strain > base['strain'] + 1e-8:
                trial['reason'] = 'strain_increased'
                continue
            v = previous.Vina(sf_name='vina', cpu=1, seed=entry['seed'] + j + 10000, verbosity=0)
            v.set_receptor(str(receptor))
            v.set_ligand_from_string(previous.MASTER.mol_to_pdbqt_meeko(candidate))
            v.compute_vina_maps(center=pp.mean(0).tolist(), box_size=[20., 20., 20.])
            v.dock(exhaustiveness=4, n_poses=1)
            vina = float(v.energies(n_poses=1)[0][0])
            trial['vina'] = vina
            row = dict(base, vina=vina, strain=strain, clashes=clashes)
            if not eligible(row, base):
                trial['reason'] = 'vina_increased'
                continue
            improves = (vina < base['vina'] - 1e-6 or strain < base['strain'] - 1e-6 or clashes < base['clashes'])
            if not improves:
                trial['reason'] = 'unchanged'
                continue
            trial['reason'] = 'eligible'
            # Fixed geometry-first ordering, all candidates must dominate input.
            if (strain, vina, clashes) < (best['strain'], best['vina'], best['clashes']):
                winner, best = candidate, row
                best = dict(best, second_pass_source=source)
        except AssertionError:
            raise
        except Exception as exc:
            trial.update(reason='candidate_error', error=str(exc))
        finally:
            diagnostics.append(trial)
    original = dict(base, variant='original')
    selected = dict(best, variant='balanced')
    selected['second_pass'] = dict(source=best.get('second_pass_source', 'retained'), candidates=diagnostics)
    assert eligible(selected, base)
    (report / f"p{entry['index']}_s{j}_balanced.mol").write_text(Chem.MolToMolBlock(winner))
    np.savez_compressed(report / f"p{entry['index']}_s{j}_balanced_coordinates.npz", xyz=winner.GetConformer().GetPositions())
    result = dict(rows=[original, selected])
    previous.atomic_json(report / 'tasks' / f"p{entry['index']}_s{j}.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', required=True)
    parser.add_argument('--baseline', required=True, help='Completed balanced comparison_docked directory')
    parser.add_argument('--out', required=True)
    parser.add_argument('--pockets', type=int, default=10)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--resume', action='store_true')
    a = parser.parse_args()
    cache, source, out = Path(a.cache), Path(a.baseline), Path(a.out)
    meta = json.loads((source / 'run_metadata.json').read_text())
    if Path(meta['cache']).resolve() != cache.resolve():
        raise ValueError('Source and requested cache differ')
    if a.workers < 1 or a.pockets < 1:
        raise ValueError('Workers and pockets must be positive')
    manifest = json.loads((cache / 'manifest.json').read_text())
    entries = manifest['files'][:a.pockets]
    if len(entries) != a.pockets:
        raise ValueError('Requested pockets unavailable')
    report = out / 'comparison_docked'
    if not a.resume and out.exists() and any(out.iterdir()):
        raise ValueError('Use a new output directory or --resume')
    if a.resume and not (report / 'run_metadata.json').exists():
        raise ValueError('Resume metadata missing')
    if a.resume:
        old = json.loads((report / 'run_metadata.json').read_text())
        if old['baseline_samples_sha256'] != previous.sha(source / 'samples.json') or old['script_sha256'] != previous.sha(__file__) or old['pockets'] != a.pockets:
            raise ValueError('Resume source, code or pocket count changed')
    (report / 'tasks').mkdir(parents=True, exist_ok=True)
    allowed = {(e['index'], j) for e in entries for j in range(manifest['args']['samples'])}
    complete = {}
    if a.resume:
        for path in (report / 'tasks').glob('*.json'):
            complete.update(previous.groups(json.loads(path.read_text())['rows']))
        if not set(complete) <= allowed:
            raise ValueError('Saved tasks differ from requested samples')
    previous.atomic_json(report / 'run_metadata.json', dict(
        scope='Second pass from balanced poses; at most three proposals per molecule',
        cache=str(cache.resolve()), baseline=str(source.resolve()), pockets=a.pockets,
        baseline_samples_sha256=previous.sha(source / 'samples.json'), script_sha256=previous.sha(__file__),
        selection='No metric worsening versus balanced input; minimize (strain, vina, clashes)',
        baseline_variant_in_report='original denotes balanced input, not original V3', workers=a.workers,
        docking=dict(exhaustiveness=4, box_size=[20, 20, 20], seed='entry seed + sample + 10000')))
    tasks = [(str(cache), str(report), e, j) for e in entries for j in range(manifest['args']['samples']) if (e['index'], j) not in complete]
    def save():
        rows = [r for key in sorted(complete) for r in sorted(complete[key], key=lambda r: previous.VARIANTS.index(r['variant']))]
        previous.atomic_json(report / 'samples.json', rows)
        if rows:
            summary = previous.summarize(rows)
            for variant in previous.VARIANTS:
                rr = [r for r in rows if r['variant'] == variant]
                for inherited in ['selected_nonoriginal', 'optimizer_converged']:
                    summary[variant].pop(inherited, None)
                summary[variant]['second_pass_replaced'] = sum(r.get('second_pass', {}).get('source', 'retained') != 'retained' for r in rr)
            previous.atomic_json(report / 'summary.json', summary)
    print(f'Resumed {len(complete)}/{len(allowed)}; pending {len(tasks)}; workers={a.workers}', flush=True)
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=multiprocessing.get_context('spawn'), initializer=initialize, initargs=(str(source),)) as pool:
        futures = [pool.submit(one_sample, task) for task in tasks]
        for future in as_completed(futures):
            complete.update(previous.groups(future.result()['rows']))
            save()
            print(f'Completed {len(complete)}/{len(allowed)} molecules (up to three proposals)', flush=True)
    save()
    print('Finished:', report, flush=True)


if __name__ == '__main__':
    main()
