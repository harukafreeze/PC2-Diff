"""Bounded, exploratory Vina seed sensitivity; never edits manuscript inputs."""
from pathlib import Path
import concurrent.futures as cf
import csv
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
CACHE = Path(os.environ.get('BUILD_POSE_CACHE', str(Path.home() / '.cache' / 'build-pose-exploration'))).expanduser()
sys.path.insert(0, str(CACHE / 'python'))
from meeko import MoleculePreparation, PDBQTWriterLegacy

SEEDS = [11, 22, 33, 44, 55]
CENTER = [-23.883934020996094, 31.168371200561523, 11.053484916687012]
SOURCE = ROOT / 'real_world_study/results/allosteric_screen_100k'
DATA = ROOT / 'real_world_study/data'
RUN = CACHE / ('run_' + time.strftime('%Y%m%d_%H%M%S'))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atoms(text):
    result = {}
    for line in text.splitlines():
        if line.startswith(('ATOM', 'HETATM')):
            elem = line.split()[-1].upper()
            if elem in ('H', 'HD', 'HS'):
                continue
            result[int(line[6:11])] = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return result


def graph_and_mapping(text):
    smiles = next(line[len('REMARK SMILES '):] for line in text.splitlines()
                  if line.startswith('REMARK SMILES ') and not line.startswith('REMARK SMILES IDX'))
    mapping = {}
    for line in text.splitlines():
        if line.startswith('REMARK SMILES IDX '):
            values = [int(v) for v in line.split()[3:]]
            mapping.update({values[i + 1]: values[i] - 1 for i in range(0, len(values), 2)})
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or set(mapping.values()) != set(range(mol.GetNumAtoms())):
        raise ValueError('SMILES/serial mapping incomplete')
    return mol, mapping


def posed_mol(template, mapping, text):
    positions = atoms(text)
    if set(positions) != set(mapping):
        raise ValueError('Docked pose changed heavy atom serials')
    mol = Chem.Mol(template)
    conf = Chem.Conformer(mol.GetNumAtoms())
    for serial, idx in mapping.items():
        conf.SetAtomPosition(idx, positions[serial])
    mol.RemoveAllConformers()
    mol.AddConformer(conf)
    return mol


def rmsd(a, b):
    # CalcRMS leaves coordinates in the receptor frame; it does NOT fit ligands.
    return float(rdMolAlign.CalcRMS(a, b, maxMatches=10000))


def prepare():
    if (OUT / 'plan.json').exists():
        raise RuntimeError('This output folder already has a plan; use a fresh folder')
    RUN.mkdir(parents=True, exist_ok=False)
    (RUN / 'inputs').mkdir()
    (RUN / 'poses').mkdir()
    (RUN / 'logs').mkdir()
    receptor = DATA / 'pocket_EGFR_Allosteric.pdbqt'
    shutil.copy2(receptor, RUN / 'inputs/receptor.pdbqt')
    inputs = {}
    sources = {'receptor': {'path': str(receptor), 'sha256': sha(receptor)}}
    for name, filename in [('candidate_27174', 'lead_3_pose.pdbqt'), ('candidate_58449', 'lead_1_pose.pdbqt')]:
        path = SOURCE / filename
        text = path.read_text()
        text = '\n'.join(line for line in text.splitlines()
                         if not line.startswith(('MODEL', 'ENDMDL', 'REMARK VINA RESULT',
                                                 'REMARK INTER', 'REMARK INTRA', 'REMARK UNBOUND'))) + '\n'
        (RUN / f'inputs/{name}.pdbqt').write_text(text)
        inputs[name] = text
        sources[name] = {'path': str(path), 'sha256': sha(path)}
    sdf = DATA / '57N_ideal.sdf'
    ref = next(Chem.ForwardSDMolSupplier(io.BytesIO(sdf.read_bytes()), removeHs=False))
    ref = Chem.AddHs(Chem.RemoveHs(ref))
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    if AllChem.EmbedMolecule(ref, params) != 0:
        raise RuntimeError('Reference conformer embedding failed')
    uff_status = AllChem.UFFOptimizeMolecule(ref, maxIters=200)
    text, ok, error = PDBQTWriterLegacy.write_string(MoleculePreparation().prepare(ref)[0])
    if not ok:
        raise RuntimeError(error)
    inputs['EAI001'] = text
    (RUN / 'inputs/EAI001.pdbqt').write_text(text)
    sources['EAI001'] = {'path': str(sdf), 'sha256': sha(sdf),
                        'preparation': 'RDKit ETKDGv3 seed42, UFF200, Meeko0.7.1 default',
                        'uff_return_code': uff_status}
    plan = {
        'question': 'Sensitivity of top-ranked EGFR allosteric docking poses to the search seed',
        'seeds': SEEDS, 'molecules': list(inputs), 'number_of_docking_jobs': 15,
        'engine': 'official AutoDock Vina 1.2.7 Windows CLI',
        'binary_sha256': sha(CACHE / 'vina_1.2.7_win.exe'),
        'center_angstrom': CENTER, 'box_angstrom': [16.0] * 3,
        'exhaustiveness': 8, 'num_modes': 1, 'cpu_per_job': 2,
        'parallel_jobs': 2, 'job_timeout_seconds': 300,
        'primary_readouts': ['score range', 'symmetry-corrected heavy-atom RMSD without alignment',
                             'receptor heavy-atom contact fingerprint at 4.2 angstrom'],
        'scope': 'Fixed ligand preparation and fixed receptor; search-seed sensitivity only. No MD or activity inference.',
        'selection_policy': 'Keep all 15 scheduled outcomes, including errors. No score-based rerun or seed expansion.',
        'sources': sources, 'ascii_run_folder': str(RUN),
        'versions': {'rdkit': Chem.rdBase.rdkitVersion, 'numpy': np.__version__, 'python': sys.version},
    }
    write_json(OUT / 'plan.json', plan)
    shutil.copytree(RUN / 'inputs', OUT / 'inputs')
    return inputs


def run_job(name, seed):
    command = [str(CACHE / 'vina_1.2.7_win.exe'), '--receptor', 'inputs/receptor.pdbqt',
               '--ligand', f'inputs/{name}.pdbqt', '--out', f'poses/{name}_seed{seed}.pdbqt',
               '--center_x', str(CENTER[0]), '--center_y', str(CENTER[1]), '--center_z', str(CENTER[2]),
               '--size_x', '16', '--size_y', '16', '--size_z', '16', '--exhaustiveness', '8',
               '--num_modes', '1', '--cpu', '2', '--seed', str(seed), '--verbosity', '1']
    start = time.time()
    try:
        proc = subprocess.run(command, cwd=str(RUN), capture_output=True, text=True, timeout=300)
        log = proc.stdout + '\n' + proc.stderr
        (RUN / f'logs/{name}_seed{seed}.txt').write_text(log, encoding='utf-8')
        if proc.returncode != 0:
            raise RuntimeError('Vina returned ' + str(proc.returncode) + ': ' + log[-800:])
        text = (RUN / f'poses/{name}_seed{seed}.pdbqt').read_text()
        score = float(next(line.split()[3] for line in text.splitlines() if line.startswith('REMARK VINA RESULT:')))
        result = {'molecule': name, 'seed': seed, 'status': 'ok', 'vina': score,
                  'elapsed_seconds': round(time.time() - start, 2)}
    except Exception as exc:
        result = {'molecule': name, 'seed': seed, 'status': 'error', 'error': str(exc),
                  'elapsed_seconds': round(time.time() - start, 2)}
    print(json.dumps(result), flush=True)
    return result


def analyze(inputs, outcomes):
    receptor = np.array(list(atoms((RUN / 'inputs/receptor.pdbqt').read_text()).values()))
    results = []
    summaries = {}
    for name, input_text in inputs.items():
        template, mapping = graph_and_mapping(input_text)
        archived = posed_mol(template, mapping, input_text) if name != 'EAI001' else None
        poses, fingerprints, group = [], [], []
        for row in sorted([dict(x) for x in outcomes if x['molecule'] == name], key=lambda x: x['seed']):
            if row['status'] == 'ok':
                text = (RUN / f"poses/{name}_seed{row['seed']}.pdbqt").read_text()
                mol = posed_mol(template, mapping, text)
                xyz = mol.GetConformer().GetPositions()
                distances = np.linalg.norm(xyz[:, None] - receptor[None, :], axis=2)
                row['minimum_receptor_distance'] = float(distances.min())
                row['receptor_pairs_below_1p50'] = int((distances < 1.5).sum())
                row['receptor_atoms_within_4p2'] = int((distances <= 4.2).any(axis=0).sum())
                row['centroid'] = xyz.mean(axis=0).tolist()
                if archived is not None:
                    row['rmsd_to_archived_pose'] = rmsd(mol, archived)
                poses.append(mol)
                fingerprints.append(set(np.where((distances <= 4.2).any(axis=0))[0].tolist()))
                writer = Chem.SDWriter(str(RUN / f"poses/{name}_seed{row['seed']}.sdf"))
                mol.SetProp('_Name', f"{name}_seed{row['seed']}")
                mol.SetProp('Vina', str(row['vina']))
                writer.write(mol)
                writer.close()
            group.append(row)
        good = [row for row in group if row['status'] == 'ok']
        n = len(poses)
        matrix = np.array([[rmsd(a, b) for b in poses] for a in poses]) if n else np.empty((0, 0))
        pairs = matrix[np.triu_indices(n, 1)]
        jaccard = [len(a & b) / len(a | b) if a | b else 1.0
                   for i, a in enumerate(fingerprints) for b in fingerprints[i + 1:]]
        if good:
            medoid = int(np.argmin(matrix.sum(axis=1)))
            scores = np.array([row['vina'] for row in good])
            summaries[name] = {'completed': n, 'scheduled': 5, 'vina_mean': float(scores.mean()),
                               'vina_std_sample': float(scores.std(ddof=1)) if n > 1 else None,
                               'vina_range': [float(scores.min()), float(scores.max())],
                               'median_pairwise_rmsd': float(np.median(pairs)) if len(pairs) else None,
                               'max_pairwise_rmsd': float(pairs.max()) if len(pairs) else None,
                               'medoid_seed': good[medoid]['seed'],
                               'within_2A_of_medoid': int((matrix[medoid] <= 2).sum()),
                               'median_receptor_atom_contact_jaccard': float(np.median(jaccard)) if jaccard else None,
                               'rmsd_matrix': matrix.tolist(), 'matrix_seed_order': [r['seed'] for r in good]}
        results.extend(group)
    write_json(OUT / 'results.json', {'summary': summaries, 'runs': results})
    columns = sorted(set().union(*(row.keys() for row in results)))
    with (OUT / 'runs.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(results)
    shutil.copytree(RUN / 'poses', OUT / 'poses')
    shutil.copytree(RUN / 'logs', OUT / 'logs')
    print('SUMMARY ' + json.dumps(summaries), flush=True)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    inputs = prepare()
    print('PLAN WRITTEN. Starting fixed 15-job exploration.', flush=True)
    outcomes = []
    with cf.ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(run_job, name, seed) for name in inputs for seed in SEEDS]
        for future in cf.as_completed(jobs):
            outcomes.append(future.result())
            write_json(OUT / 'progress.json', outcomes)
    analyze(inputs, outcomes)
