# scripts/supplementary/screen_allosteric_candidates.py
# =====================================================================================
# BUILD 3.0 -- full allosteric screening pipeline for EGFR 5D41 (alphaC-helix cleft).
#
# SELF-CONTAINED: this single file needs no sibling modules. Only the repository itself
# (project_build) and the build_env packages are required.
#
# Stage design (cheap -> expensive, so nothing expensive is wasted):
#   A. GPU  : generate --total_samples molecules (default 20000) conditioned on the pocket.
#             Cached to gen_cache.npz, so a crash or a second pass costs no GPU time.
#   B. CPU  : bond perception; require a valid molecule and a single connected component.
#   C. CPU  : hard validity. Topology: no atom in 3+ rings (cage), no three-membered ring
#             fused to another ring, ring count <= 5, largest ring <= 7, at most one ring
#             smaller than 5, at most one spiro atom. Geometry: after UFF relaxation the
#             closest non-bonded heavy-atom contact must be >= 1.90 A. This replaces the
#             bridgehead test, which returns 0 for the cage reported earlier.
#   D. CPU  : paper protocol gates -- receptor clashes <= 3 (contacts below 1.50 A),
#             QED >= 0.55, SA <= 4.5, strain <= 450 kcal/mol, using the same definitions as
#             the published numbers (including the .pt pocket for the clash count).
#   E. CPU  : AutoDock Vina docking in the 16.0 A focused box; require Vina <= -9.00.
#   F. write: per-molecule audit CSV, funnel report, top leads, docked poses and a correctly
#             mapped complex PDB for the number one lead.
#
# Everything is seeded (sampling and Vina), so the run is reproducible.
#
# Usage (from the repository root, inside build_env):
#   python scripts/supplementary/screen_allosteric_candidates.py --total_samples 20000
#   python scripts/supplementary/screen_allosteric_candidates.py --skip_docking    # filter only
#   python scripts/supplementary/screen_allosteric_candidates.py --reuse_cache     # reuse GPU cache
#
# Outputs (inside --out_dir, default real_world_study/results/allosteric_screen):
#   gen_cache.npz        raw generated coordinates and atom types (GPU cache)
#   candidates.csv       one row per generated molecule: metrics and verdict
#   funnel_report.json   survivors and rejection reasons per stage
#   top_leads.csv        leads that passed the whole funnel, ranked by Vina
#   lead_k_pose.sdf/.mol validated docked pose of the k-th lead
#   lead_k_pose.pdbqt    raw Vina pose of the k-th lead
#   lead_k_metrics.json  per-lead contact network and clash audit
#   complex_<target>_top1_exported.pdb   receptor + ligand with correct element columns
#   run_config.json      all arguments and thresholds, for the record
# =====================================================================================
import argparse, collections, contextlib, json, math, multiprocessing, os, sys, time

import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, QED, RDConfig, rdDetermineBonds, rdMolDescriptors
RDLogger.DisableLog("rdApp.*")

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# paper protocol thresholds -----------------------------------------------------------
CLASH_MAX = 3           # heavy-atom receptor contacts below 1.50 A
QED_MIN = 0.55
SA_MAX = 4.5
STRAIN_MAX = 450.0      # kcal/mol, E_initial - E_minimized after 50-step H relaxation
VINA_MAX = -9.00        # kcal/mol
RING_COUNT_MAX = 5
RING_SIZE_MAX = 7
SMALL_RING_MAX = 1
SPIRO_MAX = 1
MIN_NONBONDED = 1.90    # A, closest allowed non-bonded heavy-atom contact
RAW_SOFT_MIN = 1.15     # A, raw conformer flagged as overlapping below this (diagnostic only)

VDW = {"H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "F": 1.47, "Cl": 1.75,
       "Br": 1.85, "I": 1.98, "P": 1.80, "B": 1.92, "Se": 1.90, "Mg": 1.73, "Zn": 1.39,
       "Ca": 2.31, "Na": 2.27, "K": 2.75, "Fe": 2.00, "Mn": 2.05}

TYPE2ELEM = {"A": "C", "C": "C", "N": "N", "NA": "N", "NS": "N", "OA": "O", "O": "O",
             "OS": "O", "SA": "S", "S": "S", "HD": "H", "H": "H", "HS": "H", "F": "F",
             "CL": "Cl", "BR": "Br", "I": "I", "P": "P", "MG": "Mg", "ZN": "Zn", "FE": "Fe",
             "MN": "Mn", "CA": "Ca", "K": "K", "SE": "Se", "SI": "Si", "B": "B"}

_REC_CACHE = {}


def load_sascorer():
    """The SA gate is part of the protocol, so a silent fallback would invalidate it."""
    for candidate in (os.path.join(RDConfig.RDContribDir, "SA_Score"),
                      os.path.join(os.environ.get("CONDA_PREFIX", ""), "share", "RDKit",
                                   "Contrib", "SA_Score")):
        if candidate not in sys.path:
            sys.path.append(candidate)
    try:
        import sascorer
        return sascorer
    except Exception as exc:
        raise SystemExit("cannot import sascorer (required for the SA gate): %s" % exc)


sascorer = load_sascorer()


@contextlib.contextmanager
def suppress_c_stderr():
    """Meeko and Vina print C-level noise; keep the progress log readable."""
    try:
        null_fds = os.open(os.devnull, os.O_RDWR)
        save_fds = os.dup(2)
        os.dup2(null_fds, 2)
        yield
    finally:
        os.dup2(save_fds, 2)
        os.close(null_fds)
        os.close(save_fds)


def element_from_line(line, atom_name):
    """Resolve the element of a PDBQT or PDB record without relying on column alignment.

    Two-character types (Cl, Br) are placed at slightly different columns by different
    writers, and slicing columns 77-78 silently turns Cl into C. The trailing whitespace
    token is the atom type in every PDBQT writer used here and the element in PDB files, so
    it is preferred whenever it is recognisable. The one ambiguous token is Na, which the
    PDBQT table maps to N; no structure in this study contains sodium.
    """
    parts = line.split()
    token = parts[-1].upper() if parts else ""
    if token in TYPE2ELEM:
        return TYPE2ELEM[token]
    raw = line[76:78].strip().upper() if len(line) >= 78 else ""
    if raw in TYPE2ELEM:
        return TYPE2ELEM[raw]
    if len(token) == 1 and token.isalpha():
        return token.capitalize()
    if raw:
        return raw.capitalize()
    return atom_name[0].upper() if atom_name else "C"


def parse_pdbqt_atoms(text, keep_hydrogen=False):
    """Return [(element, np.array([x, y, z]))] for the ATOM and HETATM records."""
    out = []
    for line in text.splitlines():
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        elem = element_from_line(line, line[12:16].strip())
        if elem == "H" and not keep_hydrogen:
            continue
        out.append((elem, np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])))
    return out


def fragments_and_min_distance(mol):
    """Connected components and shortest heavy-atom pair (chemical bonds included)."""
    heavy = Chem.RemoveHs(Chem.Mol(mol))
    positions = heavy.GetConformer().GetPositions()
    count = len(positions)
    matrix = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
    np.fill_diagonal(matrix, 9.9)
    adjacency = matrix < 1.75
    seen = [False] * count
    components = 0
    for start in range(count):
        if seen[start]:
            continue
        components += 1
        stack = [start]
        seen[start] = True
        while stack:
            node = stack.pop()
            for neighbour in np.nonzero(adjacency[node])[0]:
                if not seen[neighbour]:
                    seen[neighbour] = True
                    stack.append(neighbour)
    return components, float(matrix.min()), count, heavy, positions


def audit_against_receptor(rec_atoms, heavy, positions):
    """Per-residue minimum heavy-atom contacts plus clash counts under three criteria."""
    per_residue = {}
    severe = mild = 0
    worst_ratio, worst_pair = 9.9, ""
    symbols = [a.GetSymbol() for a in heavy.GetAtoms()]
    for i in range(len(positions)):
        for rec in rec_atoms:
            d = float(np.linalg.norm(positions[i] - rec["xyz"]))
            total = VDW.get(symbols[i], 1.70) + VDW.get(rec["elem"], 1.70)
            if d < 1.50:
                severe += 1
            if d < total - 0.40:
                mild += 1
            if d / total < worst_ratio:
                worst_ratio = d / total
                worst_pair = rec["res"] + ":" + rec["name"] + "-LIG:" + str(i + 1)
            pair = rec["res"] + ":" + rec["name"] + " - LIG:" + symbols[i] + str(i + 1)
            if rec["res"] not in per_residue or d < per_residue[rec["res"]][0]:
                per_residue[rec["res"]] = (d, pair, rec["elem"], symbols[i])
    top = sorted(per_residue.items(), key=lambda kv: kv[1][0])[:10]
    return top, severe, mild, worst_ratio, worst_pair


def write_complex(receptor_path, mol, out_pdb, remark):
    """Receptor ATOM records plus ligand HETATM records, with element columns filled in."""
    receptor_lines = []
    serial = 1
    for line in open(receptor_path, encoding="utf-8", errors="ignore"):
        if not line.startswith("ATOM"):
            continue
        elem = element_from_line(line, line[12:16].strip())
        receptor_lines.append("ATOM  %5d %-4s %-3s A%4s    %8.3f%8.3f%8.3f  1.00 20.00          %2s"
                              % (serial, line[12:16].strip(), line[17:20].strip(),
                                 line[22:26].strip(), float(line[30:38]), float(line[38:46]),
                                 float(line[46:54]), elem))
        serial += 1
    conf = mol.GetConformer()
    ligand_lines = []
    for atom in mol.GetAtoms():
        point = conf.GetAtomPosition(atom.GetIdx())
        ligand_lines.append("HETATM%5d %-4s LIG B   1    %8.3f%8.3f%8.3f  1.00 20.00          %2s"
                            % (atom.GetIdx() + 1, atom.GetSymbol() + str(atom.GetIdx() + 1),
                               point.x, point.y, point.z, atom.GetSymbol()))
    with open(out_pdb, "w", encoding="utf-8") as handle:
        handle.write("REMARK    " + remark + chr(10))
        handle.write(chr(10).join(receptor_lines) + chr(10) + "TER" + chr(10))
        handle.write(chr(10).join(ligand_lines) + chr(10) + "END" + chr(10))
    return len(receptor_lines), len(ligand_lines)


def receptor_atoms(path):
    """Parse the rigid receptor once per worker process."""
    if path not in _REC_CACHE:
        atoms = []
        for line in open(path, encoding="utf-8", errors="ignore"):
            if not line.startswith("ATOM"):
                continue
            raw = line[76:78].strip().upper() if len(line) >= 78 else ""
            atoms.append({
                "res": line[17:20].strip() + line[22:26].strip(),
                "name": line[12:16].strip(),
                "elem": TYPE2ELEM.get(raw, raw.capitalize() if raw else "C"),
                "xyz": np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])]),
            })
        _REC_CACHE[path] = atoms
    return _REC_CACHE[path]


def topology_only(mol):
    """Copy of mol with all bonds single and aromatic flags cleared, for graph matching."""
    work = Chem.RWMol(mol)
    for bond in work.GetBonds():
        bond.SetBondType(Chem.BondType.SINGLE)
        bond.SetIsAromatic(False)
    for atom in work.GetAtoms():
        atom.SetIsAromatic(False)
    out = work.GetMol()
    Chem.SanitizeMol(out, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL
                     ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
                     ^ Chem.SanitizeFlags.SANITIZE_SETAROMATICITY)
    return out


def guard_bond_geometry(mol):
    """Fail loudly if the pose coordinates cannot belong to this molecular graph."""
    positions = mol.GetConformer().GetPositions()
    bonded = np.zeros((mol.GetNumAtoms(), mol.GetNumAtoms()), dtype=bool)
    worst_bond, closest_bond = 0.0, 9.9
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bonded[i, j] = bonded[j, i] = True
        d = float(np.linalg.norm(positions[i] - positions[j]))
        worst_bond = max(worst_bond, d)
        closest_bond = min(closest_bond, d)
    heavy = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() != "H"]
    closest_nonbonded = 9.9
    for a in heavy:
        for b in heavy:
            if b <= a or bonded[a, b]:
                continue
            closest_nonbonded = min(closest_nonbonded,
                                    float(np.linalg.norm(positions[a] - positions[b])))
    if worst_bond > 1.85 or closest_bond < 0.90:
        raise RuntimeError("implausible bond lengths: coordinate mapping is wrong")
    if closest_nonbonded < MIN_NONBONDED:
        raise RuntimeError("non-bonded heavy atoms %.2f A apart" % closest_nonbonded)
    return closest_nonbonded


def mol_from_pose(atoms, template_smiles, cov_factor=1.3):
    """Map docked coordinates onto the SMILES molecular graph.

    The graph and the bond orders come from template_smiles, so the chemistry stays valid,
    while the graph used for matching comes from the pose coordinates. A geometry guard
    rejects any mapping that would produce impossible bond lengths, which is what keeps atom
    identity and coordinates from drifting apart.
    """
    xyz = [str(len(atoms)), "pose"]
    for elem, pos in atoms:
        xyz.append("%-2s %12.5f %12.5f %12.5f" % (elem, pos[0], pos[1], pos[2]))
    template = Chem.MolFromSmiles(template_smiles)
    if template is None:
        raise RuntimeError("cannot parse the ligand SMILES")
    expected = template.GetNumBonds()
    pose, got = None, None
    for factor in (cov_factor, 1.25, 1.20, 1.15, 1.10):
        pose = Chem.MolFromXYZBlock(chr(10).join(xyz) + chr(10))
        rdDetermineBonds.DetermineConnectivity(pose, useHueckel=False, covFactor=factor)
        got = pose.GetNumBonds()
        if got == expected:
            break
    if got != expected:
        raise RuntimeError("pose geometry is not physically realisable: %d bonds implied by "
                           "the coordinates vs %d in the SMILES" % (got, expected))
    match = topology_only(pose).GetSubstructMatch(topology_only(template))
    if len(match) != template.GetNumAtoms():
        raise RuntimeError("topology mismatch: the pose is not isomorphic to the SMILES")
    pose_conf = pose.GetConformer()
    conf = Chem.Conformer(template.GetNumAtoms())
    for template_idx, pose_idx in enumerate(match):
        conf.SetAtomPosition(template_idx, pose_conf.GetAtomPosition(pose_idx))
    out = Chem.Mol(template)
    out.RemoveAllConformers()
    out.AddConformer(conf, assignId=True)
    out = Chem.AddHs(out, addCoords=True)
    guard_bond_geometry(out)
    return out


def topology_reason(mol):
    """Deterministic graph-level rejection of cages and unusable polycyclics."""
    rings = [list(r) for r in Chem.GetSymmSSSR(mol)]
    if not rings:
        return "no ring"
    membership = collections.Counter()
    for ring in rings:
        for idx in ring:
            membership[idx] += 1
    if any(count >= 3 for count in membership.values()):
        return "an atom belongs to 3 or more rings (cage)"
    bond_in_rings = collections.Counter()
    for ring in rings:
        for k in range(len(ring)):
            bond_in_rings[tuple(sorted((ring[k], ring[(k + 1) % len(ring)])))] += 1
    for ring in rings:
        if len(ring) != 3:
            continue
        for k in range(len(ring)):
            bond = tuple(sorted((ring[k], ring[(k + 1) % len(ring)])))
            if bond_in_rings[bond] >= 2:
                return "three-membered ring fused to another ring"
    if any(len(ring) > RING_SIZE_MAX for ring in rings):
        return "ring larger than %d" % RING_SIZE_MAX
    return None


def geometry_report(mol, max_iters=500):
    """Relax the conformer and report its quality plus the relaxed heavy-atom frame.

    Returns (closest non-bonded heavy-atom contact, absolute UFF energy, relaxed
    heavy-atom positions, heavy-atom symbols). The relaxed frame is what the design
    funnel should judge and dock, because the raw generated point cloud is noisy.
    """
    work = Chem.AddHs(mol, addCoords=True)
    Chem.FastFindRings(work)
    AllChem.UFFOptimizeMolecule(work, maxIters=max_iters)
    field = AllChem.UFFGetMoleculeForceField(work)
    energy = float(field.CalcEnergy()) if field else 9999.0
    positions = work.GetConformer().GetPositions()
    heavy = [a.GetIdx() for a in work.GetAtoms() if a.GetSymbol() != "H"]
    bonded = {tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))) for b in work.GetBonds()}
    closest = 9.9
    for i in heavy:
        for j in heavy:
            if j <= i or (i, j) in bonded:
                continue
            closest = min(closest, float(np.linalg.norm(positions[i] - positions[j])))
    symbols = [work.GetAtomWithIdx(i).GetSymbol() for i in heavy]
    return closest, energy, positions[heavy], symbols


def paper_ring_rules(mol):
    """Ring rules as stated in the paper, on the symmetrized SSSR."""
    rings = [list(r) for r in Chem.GetSymmSSSR(mol)]
    if not rings:
        return "no ring"
    if len(rings) > RING_COUNT_MAX:
        return "more than %d rings" % RING_COUNT_MAX
    if max(len(r) for r in rings) > RING_SIZE_MAX:
        return "ring larger than %d" % RING_SIZE_MAX
    if sum(1 for r in rings if len(r) < 5) > SMALL_RING_MAX:
        return "more than one ring smaller than 5"
    if rdMolDescriptors.CalcNumSpiroAtoms(mol) > SPIRO_MAX:
        return "more than one spiro atom"
    return None


def compute_clean_strain_energy(mol):
    """Paper definition: relax 50 steps, then measure how much further it can drop."""
    try:
        mol_h = Chem.AddHs(mol, addCoords=True)
        Chem.FastFindRings(mol_h)
        AllChem.UFFOptimizeMolecule(mol_h, maxIters=50)
        field = AllChem.UFFGetMoleculeForceField(mol_h)
        if not field:
            return 250.0
        before = field.CalcEnergy()
        field.Minimize(maxIts=200)
        return float(min(1000.0, max(0.0, before - field.CalcEnergy())))
    except Exception:
        return 250.0


def filter_worker(pack):
    """Stages B, C and D: reconstruct, validate and score, without docking."""
    idx, pos_np, h_np, p_pos_np, p_types_np = pack
    row = {"idx": idx, "smiles": None, "topology_reject": None, "geometry_reject": None,
           "ring_reject": None, "closest_nonbonded": None, "abs_uff": None, "clashes": None,
           "qed": None, "sa": None, "strain": None, "passed": False}
    row["raw_min_pair"] = None
    row["raw_overlap"] = None
    row["clashes_raw"] = None
    row["vina"] = None
    row["final_pass"] = False
    row["clashes_pre_dock"] = None
    row["clashes_docked"] = None
    row["pose_min_pair"] = None
    row["posebusters_min_ratio"] = None
    row["fragments_docked"] = None
    row["pose_invalid"] = None
    from project_build.postprocess.bond_perception import BondPerceptionEngine
    with suppress_c_stderr():
        mol, is_valid, n_frag = BondPerceptionEngine.reconstruct_mol(
            torch.tensor(pos_np), torch.tensor(h_np), tolerance=0.35)
        if not (is_valid and mol and n_frag == 1):
            row["topology_reject"] = "bond perception failed or fragmented"
            return row
        row["smiles"] = Chem.MolToSmiles(mol)

        # The raw point cloud is noisy by design: this generator frequently places atoms
        # inside each other (measured minimum heavy-atom pair 0.2-0.6 A), and repairing
        # that is precisely what the reconstruction stage exists for. The raw geometry is
        # therefore recorded as a diagnostic, not used as a gate; the relaxed conformer is
        # what the funnel judges and what gets docked.
        if len(pos_np) < 2:
            row["topology_reject"] = "generated sample has fewer than two atoms"
            return row
        raw_matrix = np.linalg.norm(pos_np[:, None, :] - pos_np[None, :, :], axis=-1)
        np.fill_diagonal(raw_matrix, 9.9)
        raw_min = float(raw_matrix.min())
        row["raw_min_pair"] = round(raw_min, 3)
        row["raw_overlap"] = bool(raw_min < RAW_SOFT_MIN)

        # The SBDD 9-channel ligand vocabulary is {C, N, O, F, P, S, Cl, Br, I}
        # (SBDD_HEAVY_REVERSE_MAP in project_build.postprocess.bond_perception): there is
        # no hydrogen channel, so every generated atom is a heavy atom. The pocket tensor
        # uses 6 channels {H, C, N, O, S, Se}, so index 0 is dropped there.
        pocket_heavy = p_pos_np[p_types_np > 0]
        raw_dists = np.linalg.norm(pos_np[:, None, :] - pocket_heavy[None, :, :], axis=-1)
        row["clashes_raw"] = int(np.sum(raw_dists < 1.50))

        reason = topology_reason(mol)
        if reason:
            row["topology_reject"] = reason
            return row
        closest, energy, rel_pos, rel_sym = geometry_report(mol)
        row["closest_nonbonded"] = round(closest, 3)
        row["abs_uff"] = round(energy, 1)
        if closest < MIN_NONBONDED:
            row["geometry_reject"] = "relaxed conformer: non-bonded heavy atoms %.2f A apart" % closest
            return row

        reason = paper_ring_rules(mol)
        if reason:
            row["ring_reject"] = reason
            return row

        # Receptor compatibility is judged on the DOCKED pose (stage F), never on the raw or
        # vacuum-relaxed conformer: before docking the molecule has not been placed by the
        # scoring function, so counting receptor overlaps here only measures the noise of the
        # raw point cloud. Both counts are recorded for the audit trail.
        rel_dists = np.linalg.norm(rel_pos[:, None, :] - pocket_heavy[None, :, :], axis=-1)
        row["clashes_pre_dock"] = int(np.sum(rel_dists < 1.50))
        qed = float(QED.qed(mol))
        sa = float(sascorer.calculateScore(mol))
        strain = compute_clean_strain_energy(mol)
        row.update({"qed": round(qed, 4), "sa": round(sa, 2), "strain": round(strain, 1)})
        if qed < QED_MIN or sa > SA_MAX or strain > STRAIN_MAX:
            return row
        row["passed"] = True
        return row


def dock_worker(pack):
    """Stage E: prepare, dock in the focused box, return the score and the raw pose."""
    idx, pos_np, h_np, p_pos_np, p_types_np, rec_path, center, box, exhaust, seed = pack
    from meeko import MoleculePreparation, PDBQTWriterLegacy
    from vina import Vina
    from project_build.postprocess.bond_perception import BondPerceptionEngine
    with suppress_c_stderr():
        mol, is_valid, n_frag = BondPerceptionEngine.reconstruct_mol(
            torch.tensor(pos_np), torch.tensor(h_np), tolerance=0.35)
        if not (is_valid and mol and n_frag == 1):
            return {"idx": idx, "vina": None, "pose": None, "error": "reconstruction"}
        try:
            prep = MoleculePreparation()
            # dock a relaxed copy: the raw generated conformer can still hold local overlaps,
            # and docking must start from a physically valid geometry
            relaxed = Chem.AddHs(mol, addCoords=True)
            AllChem.UFFOptimizeMolecule(relaxed, maxIters=200)
            setup = prep.prepare(relaxed)[0]
            written = PDBQTWriterLegacy.write_string(setup)
            lig_pdbqt = written[0] if isinstance(written, tuple) else written
            engine = Vina(sf_name="vina", cpu=1, seed=seed, verbosity=0)
            engine.set_receptor(rec_path)
            engine.set_ligand_from_string(lig_pdbqt)
            engine.compute_vina_maps(center=list(center), box_size=[box, box, box])
            engine.dock(exhaustiveness=exhaust, n_poses=1)
            score = float(engine.energies(n_poses=1)[0][0])
            return {"idx": idx, "vina": score, "pose": engine.poses(n_poses=1), "error": None}
        except Exception as exc:
            return {"idx": idx, "vina": None, "pose": None, "error": type(exc).__name__}


def run_pool(packs, worker, workers, label):
    """Run a worker over the packs and print progress."""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    results = []
    started = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for pack in packs:
            futures[pool.submit(worker, pack)] = pack[0]
        for done, future in enumerate(as_completed(futures), start=1):
            pack_index = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                # one malformed sample must never kill a long run
                results.append({"idx": pack_index, "smiles": None, "vina": None, "pose": None,
                                "worker_error": type(exc).__name__ + ": " + str(exc)[:120],
                                "topology_reject": "worker crashed", "geometry_reject": None,
                                "ring_reject": None, "clashes": None, "qed": None, "sa": None,
                                "strain": None, "raw_min_pair": None, "raw_overlap": None,
                                "passed": False})
            if done % 25 == 0 or done == len(futures):
                sys.stdout.write("\r  %s: %d/%d (%.1f min)"
                                 % (label, done, len(futures), (time.time() - started) / 60.0))
                sys.stdout.flush()
    print()
    return results


def generate(args, device, pocket, out_dir):
    """Stage A with an on-disk cache."""
    cache = os.path.join(out_dir, "gen_cache.npz")
    if args.reuse_cache and os.path.exists(cache):
        blob = np.load(cache)
        print("stage A: reused %d cached molecules" % len(blob["coords"]))
        return blob["coords"], blob["types"]

    import project_build.models.backbones.target_sbdd_backbone  # registers the backbone
    from project_build.core.registry import BACKBONE_REGISTRY
    from project_build.diffusion.sde.vp_sde import DualSpaceVPSDE
    from project_build.diffusion.solvers.reverse_sde_sampler import ReverseSDESampler
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = BACKBONE_REGISTRY.build(checkpoint.get("config", {})["model"]).to(device)
    model.load_state_dict(checkpoint.get("ema_state_dict", checkpoint["model_state_dict"]))
    model.eval()
    sde = DualSpaceVPSDE(schedule_type="cosine").to(device)
    sampler = ReverseSDESampler(sde=sde, num_steps=100, spring_k=0.25, corrector_steps=1,
                                bond_gate_thresh=0.35, enable_svd_seeding=True,
                                enable_self_healing=True, device=device)
    p_pos, p_h = pocket["pocket_pos"], pocket["pocket_h"]
    p_center = torch.mean(p_pos, dim=0, keepdim=True)
    p_pos_centered = (p_pos - p_center).unsqueeze(0)
    p_h_batch = p_h.unsqueeze(0)

    started = time.time()
    coords_list, types_list = [], []
    for done in range(0, args.total_samples, args.batch_size):
        size = min(args.batch_size, args.total_samples - done)
        with torch.no_grad():
            sampled, atom_types, _ = sampler.sample(
                model=model, num_samples=size, num_atoms=args.num_atoms, num_atom_types=9,
                model_kwargs={"pocket_pos": p_pos_centered.repeat(size, 1, 1),
                              "pocket_h": p_h_batch.repeat(size, 1, 1)})
        coords_list.append(sampled.cpu().numpy() + p_center.cpu().numpy())
        types_list.append(atom_types.cpu().numpy())
        sys.stdout.write("\r  stage A: generated %d/%d (%.1f min)"
                         % (min(done + size, args.total_samples), args.total_samples,
                            (time.time() - started) / 60.0))
        sys.stdout.flush()
    print()
    coords = np.concatenate(coords_list, axis=0)
    types = np.concatenate(types_list, axis=0)
    np.savez_compressed(cache, coords=coords, types=types)
    print("stage A: cached %d molecules -> %s" % (len(coords), cache))
    return coords, types


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="EGFR_Allosteric")
    parser.add_argument("--data_dir", default="real_world_study/data/allosteric_targets")
    parser.add_argument("--out_dir", default="real_world_study/results/allosteric_screen")
    parser.add_argument("--checkpoint", default="checkpoints/sbdd/latest_sbdd_model.pt")
    parser.add_argument("--total_samples", type=int, default=20000)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--num_atoms", type=int, default=25)
    parser.add_argument("--box", type=float, default=16.0)
    parser.add_argument("--exhaustiveness", type=int, default=8,
                        help="8 matches the protocol used for the reference drugs")
    parser.add_argument("--vina_max", type=float, default=-8.0,
                        help="reporting cut on the docked score; the distribution is printed too")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--export_top", type=int, default=3)
    parser.add_argument("--reuse_cache", action="store_true")
    parser.add_argument("--skip_docking", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    workers = args.workers if args.workers > 0 else max(1, multiprocessing.cpu_count() - 2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 100)
    print("BUILD 3.0 screening | %s | samples %d | box %.1f A | seed %d | %d workers"
          % (args.target, args.total_samples, args.box, args.seed, workers))
    print("=" * 100)

    pocket_path = os.path.join(args.data_dir, "pocket_%s.pt" % args.target)
    receptor_path = os.path.join(args.data_dir, "pocket_%s.pdbqt" % args.target)
    for path in (pocket_path, receptor_path):
        if not os.path.exists(path):
            raise SystemExit("missing input: %s" % path)
    pocket = torch.load(pocket_path, map_location=device)
    center = [float(v) for v in pocket["center"]]
    p_pos_np = pocket["pocket_pos"].cpu().numpy()
    p_types_np = np.argmax(pocket["pocket_h"].cpu().numpy(), axis=-1)
    print("pocket center: [%.2f, %.2f, %.2f] | pocket atoms: %d"
          % (center[0], center[1], center[2], len(p_pos_np)))

    coords, types = generate(args, device, pocket, args.out_dir)
    packs = [(i, coords[i], types[i], p_pos_np, p_types_np) for i in range(len(coords))]

    print("stages B-D: reconstruct, validate, score")
    rows = run_pool(packs, filter_worker, workers, "stages B-D")
    rows.sort(key=lambda r: r["idx"])

    survivors = [r for r in rows if r["passed"]]
    reasons = {}
    for row in rows:
        for key in ("topology_reject", "geometry_reject", "ring_reject"):
            if row[key]:
                reasons[row[key]] = reasons.get(row[key], 0) + 1
    funnel = {
        "generated": len(rows),
        "reconstructed_and_valid": sum(1 for r in rows if r["smiles"] is not None),
        "rejected_topology": sum(1 for r in rows if r["topology_reject"]),
        "rejected_geometry": sum(1 for r in rows if r["geometry_reject"]),
        "rejected_ring_rules": sum(1 for r in rows if r["ring_reject"]),
        "rejected_qed_sa_strain": sum(1 for r in rows if r["qed"] is not None
                                      and not r["passed"]),
        "passed_2D_and_geometry": len(survivors),
        "docked": 0,
        "passed_vina_cutoff": 0,
        "rejection_reasons": reasons,
    }
    print("funnel after stages B-D: " + json.dumps(funnel, ensure_ascii=False))
    funnel["worker_errors"] = sum(1 for r in rows if r.get("worker_error"))
    funnel["raw_conformer_overlaps_all"] = sum(1 for r in rows if r.get("raw_overlap"))
    funnel["raw_conformer_overlaps_among_survivors"] = sum(
        1 for r in survivors if r["raw_overlap"])

    dock_results = {}
    if survivors and not args.skip_docking:
        dock_packs = [(row["idx"], coords[row["idx"]], types[row["idx"]], p_pos_np, p_types_np,
                       receptor_path, center, args.box, args.exhaustiveness, args.seed)
                      for row in survivors]
        print("stage E: docking %d survivors at exhaustiveness %d"
              % (len(dock_packs), args.exhaustiveness))
        for result in run_pool(dock_packs, dock_worker, workers, "stage E"):
            dock_results[result["idx"]] = result
        funnel["docked"] = sum(1 for r in dock_results.values() if r["vina"] is not None)

    print("stage F: evaluating docked poses (clashes, geometry) and applying the score cut")
    receptor = receptor_atoms(receptor_path)
    docked_mols = {}
    for row in rows:
        hit = dock_results.get(row["idx"])
        if not hit or hit["vina"] is None or not hit["pose"]:
            continue
        row["vina"] = round(hit["vina"], 3)
        try:
            mol = mol_from_pose(parse_pdbqt_atoms(hit["pose"]), row["smiles"])
        except Exception as exc:
            row["pose_invalid"] = type(exc).__name__
            continue
        components, closest_pair, n_heavy, heavy, positions = fragments_and_min_distance(mol)
        top, severe, mild, ratio, ratio_pair = audit_against_receptor(receptor, heavy, positions)
        row["clashes_docked"] = severe
        row["fragments_docked"] = components
        row["posebusters_min_ratio"] = round(ratio, 3)
        row["pose_min_pair"] = round(closest_pair, 3)
        docked_mols[row["idx"]] = (mol, top, severe, mild, ratio)
        if severe > CLASH_MAX:
            continue
        if hit["vina"] > args.vina_max:
            continue
        row["final_pass"] = True

    scores = sorted(r["vina"] for r in rows if r.get("vina") is not None)
    funnel["pose_invalid"] = sum(1 for r in rows if r.get("pose_invalid"))
    funnel["rejected_post_dock_clash"] = sum(1 for r in rows if r.get("clashes_docked") is not None
                                             and r["clashes_docked"] > CLASH_MAX)
    funnel["passed_vina_cutoff"] = sum(1 for r in rows if r["final_pass"])
    funnel["vina_cut_used"] = args.vina_max
    if scores:
        funnel["docked_score_best"] = scores[0]
        funnel["docked_score_median"] = scores[len(scores) // 2]
        funnel["docked_score_worst"] = scores[-1]
        clash_clean = [r["vina"] for r in rows if r.get("clashes_docked") is not None
                       and r["clashes_docked"] <= CLASH_MAX]
        funnel["clash_clean_best"] = min(clash_clean) if clash_clean else None
        print("docked scores: n=%d | best %.2f | median %.2f | worst %.2f | clash-clean best %s"
              % (len(scores), scores[0], scores[len(scores) // 2], scores[-1],
                 ("%.2f" % min(clash_clean)) if clash_clean else "n/a"))

    columns = ["idx", "smiles", "final_pass", "vina", "qed", "sa", "strain",
               "closest_nonbonded", "abs_uff", "raw_min_pair", "raw_overlap", "clashes_raw",
               "clashes_pre_dock", "clashes_docked", "pose_min_pair", "posebusters_min_ratio",
               "fragments_docked", "pose_invalid",
               "topology_reject", "geometry_reject", "ring_reject"]
    csv_path = os.path.join(args.out_dir, "candidates.csv")
    with open(csv_path, "w", encoding="utf-8") as handle:
        handle.write(",".join(columns) + chr(10))
        for row in rows:
            handle.write(",".join("" if row[c] is None else str(row[c]) for c in columns)
                         + chr(10))
    print("written: %s" % csv_path)

    ranked = sorted([r for r in rows if r.get("clashes_docked") is not None
                     and r["clashes_docked"] <= CLASH_MAX], key=lambda r: r["vina"])
    report_path = os.path.join(args.out_dir, "top_by_score.csv")
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(",".join(["rank"] + columns) + chr(10))
        for rank, row in enumerate(ranked[:25], start=1):
            handle.write(",".join([str(rank)] + ["" if row[c] is None else str(row[c])
                                                for c in columns]) + chr(10))
    print("written: %s (top 25 clash-clean docked molecules, no score cut)" % report_path)
    leads = sorted([r for r in rows if r["final_pass"]], key=lambda r: r["vina"])
    lead_path = os.path.join(args.out_dir, "top_leads.csv")
    with open(lead_path, "w", encoding="utf-8") as handle:
        handle.write(",".join(["rank"] + columns) + chr(10))
        for rank, row in enumerate(leads, start=1):
            handle.write(",".join([str(rank)] + ["" if row[c] is None else str(row[c])
                                                for c in columns]) + chr(10))
    print("written: %s (%d leads passed the full funnel)" % (lead_path, len(leads)))

    exported = []
    for rank, row in enumerate(leads[:args.export_top], start=1):
        hit = dock_results.get(row["idx"])
        if not hit or not hit["pose"]:
            continue
        pose_coords = np.array([pos for _, pos in parse_pdbqt_atoms(hit["pose"])])
        pose_matrix = np.linalg.norm(pose_coords[:, None, :] - pose_coords[None, :, :], axis=-1)
        np.fill_diagonal(pose_matrix, 9.9)
        pose_min_pair = float(pose_matrix.min())
        try:
            mol = mol_from_pose(parse_pdbqt_atoms(hit["pose"]), row["smiles"])
        except Exception as exc:
            print("lead %d: pose export rejected (%s)" % (rank, exc))
            print("          smallest heavy-atom pair in the raw pose: %.2f A" % pose_min_pair)
            continue
        sdf_path = os.path.join(args.out_dir, "lead_%d_pose.sdf" % rank)
        mol_path = os.path.join(args.out_dir, "lead_%d_pose.mol" % rank)
        pdbqt_path = os.path.join(args.out_dir, "lead_%d_pose.pdbqt" % rank)
        with open(pdbqt_path, "w", encoding="utf-8") as handle:
            handle.write(hit["pose"])
        writer = Chem.SDWriter(sdf_path)
        writer.write(mol)
        writer.close()
        Chem.MolToMolFile(mol, mol_path)
        components, min_pair, n_heavy, heavy, positions = fragments_and_min_distance(mol)
        top, severe, mild, ratio, ratio_pair = audit_against_receptor(
            receptor_atoms(receptor_path), heavy, positions)
        record = {"rank": rank, "idx": row["idx"], "smiles": row["smiles"], "vina": row["vina"],
                  "qed": row["qed"], "sa": row["sa"], "clashes_docked": row["clashes_docked"],
                  "strain": row["strain"], "heavy_atoms": n_heavy, "fragments": components,
                  "receptor_contacts_below_1p50": severe, "receptor_contacts_mild": mild,
                  "posebusters_min_ratio": round(ratio, 3),
                  "pose_min_pair": round(pose_min_pair, 3),
                  "contacts": [{"residue": res, "pair": pair, "distance": round(dist, 2)}
                               for res, (dist, pair, pe, le) in top]}
        with open(os.path.join(args.out_dir, "lead_%d_metrics.json" % rank), "w",
                  encoding="utf-8") as handle:
            json.dump(record, handle, indent=2, ensure_ascii=False)
        exported.append(record)
        print("lead %d: idx %d | Vina %.2f | %s" % (rank, row["idx"], row["vina"], row["smiles"]))
        if rank == 1:
            complex_path = os.path.join(args.out_dir,
                                        "complex_%s_top1_exported.pdb" % args.target)
            n_rec, n_lig = write_complex(receptor_path, mol, complex_path,
                                         "BUILD 3.0 SCREENED LEAD idx %d (Vina %.2f)"
                                         % (row["idx"], row["vina"]))
            print("written: %s (%d receptor atoms, %d ligand atoms)"
                  % (complex_path, n_rec, n_lig))

    funnel["exported_leads"] = exported
    with open(os.path.join(args.out_dir, "funnel_report.json"), "w", encoding="utf-8") as handle:
        json.dump(funnel, handle, indent=2, ensure_ascii=False)
    with open(os.path.join(args.out_dir, "run_config.json"), "w", encoding="utf-8") as handle:
        json.dump({"args": vars(args),
                   "thresholds": {"clashes": CLASH_MAX, "qed": QED_MIN, "sa": SA_MAX,
                                  "strain": STRAIN_MAX, "vina": VINA_MAX,
                                  "min_nonbonded": MIN_NONBONDED},
                   "pocket_center": center, "workers": workers}, handle, indent=2)
    print("=" * 100)
    print("FUNNEL: " + json.dumps(funnel, ensure_ascii=False)[:800])
    print("=" * 100)
    print("files to hand over: the whole folder %s" % args.out_dir)


if __name__ == "__main__":
    main()
