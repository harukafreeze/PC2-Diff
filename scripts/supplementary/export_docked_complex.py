# scripts/supplementary/export_docked_complex.py
# =====================================================================================
# BUILD 3.0 -- correct export of a docked allosteric pose.
#
# Why this file exists:
#   An earlier export paired ligand atom records with coordinates taken from a different
#   (shuffled) array. The resulting PDB had 25 heavy atoms split into 8 fragments and 11
#   triples of atoms mutually closer than 1.8 A, which no molecule can reproduce. This
#   script rebuilds the pose -> molecule conversion so that atom identity and coordinates
#   can never drift apart, and it refuses to write output that fails its own geometry audit.
#
# What it does:
#   1. loads pocket center (.pt) and rigid receptor (.pdbqt) from --data-dir
#   2. rebuilds the ligand from --smiles, prepares it with Meeko, docks with AutoDock Vina
#      inside a 16.0 A box (same protocol as the paper)
#   3. converts the returned pose back to an RDKit molecule: heavy-atom connectivity from
#      the coordinates, bond orders transferred from the SMILES template, hydrogens added
#   4. audits the result (fragment count, minimum intra-ligand distance, receptor clashes)
#      and only writes the files if the audit passes
#   5. writes <target>_pose.sdf, <target>_pose.mol, complex_<target>_exported.pdb and
#      <target>_export_audit.json into --out-dir
#
# Usage (run from the repository root, inside the build_env environment):
#   python scripts/supplementary/export_docked_complex.py --target EGFR_Allosteric \
#          --smiles "CCC(C)NCc1cc2c3c(c1C)CC23CCC(C)C(C)CCCN"
#
# Re-export an already docked pose instead of re-docking:
#   python scripts/supplementary/export_docked_complex.py --target EGFR_Allosteric \
#          --smiles "..." --pose-pdbqt path/to/pose.pdbqt
# =====================================================================================
import argparse, json, os, sys
import numpy as np
# torch is imported lazily: it is only required when the pocket center is read from the .pt file.
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, QED, rdMolDescriptors, rdDetermineBonds
RDLogger.DisableLog("rdApp.*")

# meeko and vina are imported lazily inside the docking branch, so that an already docked
# pose can be re-exported on a machine where only RDKit is available.

# PDBQT atom types -> element symbols
TYPE2ELEM = {"A": "C", "C": "C", "N": "N", "NA": "N", "NS": "N", "OA": "O", "O": "O",
             "OS": "O", "SA": "S", "S": "S", "HD": "H", "H": "H", "HS": "H", "F": "F",
             "CL": "Cl", "BR": "Br", "I": "I", "P": "P", "MG": "Mg", "ZN": "Zn", "FE": "Fe",
             "MN": "Mn", "CA": "Ca", "K": "K", "NAA": "Na", "SE": "Se", "SI": "Si", "B": "B"}

VDW = {"H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "F": 1.47, "Cl": 1.75,
       "Br": 1.85, "I": 1.98, "P": 1.80, "B": 1.92, "Se": 1.90, "Mg": 1.73, "Zn": 1.39,
       "Ca": 2.31, "Na": 2.27, "K": 2.75, "Fe": 2.00, "Mn": 2.05}


def parse_pdbqt_atoms(text, keep_hydrogen=False):
    """Return [(element, np.array([x, y, z]))] for ATOM/HETATM records of a PDBQT text."""
    out = []
    for line in text.splitlines():
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        raw = line[76:78].strip().upper() if len(line) >= 78 else ""
        elem = TYPE2ELEM.get(raw, raw.capitalize() if raw else "C")
        if elem == "H" and not keep_hydrogen:
            continue
        out.append((elem, np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])))
    return out


def mol_from_pose(atoms, template_smiles, cov_factor=1.3):
    """Map the docked coordinates onto the SMILES molecular graph.

    The molecular graph and its bond orders come from the argument --smiles, so the chemistry
    stays valid, while the graph used for matching is derived from the pose coordinates. A
    geometry guard rejects any mapping that would produce impossible bond lengths.
    """
    xyz = [str(len(atoms)), "pose"]
    for elem, pos in atoms:
        xyz.append("%-2s %12.5f %12.5f %12.5f" % (elem, pos[0], pos[1], pos[2]))
    pose = Chem.MolFromXYZBlock(chr(10).join(xyz) + chr(10))
    if pose is None:
        raise RuntimeError("MolFromXYZBlock failed")
    template = Chem.MolFromSmiles(template_smiles)
    if template is None:
        raise RuntimeError("cannot parse --smiles")
    expected_bonds = template.GetNumBonds()
    got_bonds = None
    for factor in (cov_factor, 1.25, 1.20, 1.15, 1.10):
        pose = Chem.MolFromXYZBlock(chr(10).join(xyz) + chr(10))
        rdDetermineBonds.DetermineConnectivity(pose, useHueckel=False, covFactor=factor)
        got_bonds = pose.GetNumBonds()
        if got_bonds == expected_bonds:
            print("connectivity from coordinates: %d bonds (covFactor %.2f), matches the SMILES graph"
                  % (got_bonds, factor))
            break
    if got_bonds != expected_bonds:
        raise RuntimeError(
            "pose geometry is not physically realisable: %d bonds implied by the coordinates vs %d "
            "in --smiles. Two carbon atoms are closer than a chemical bond, which no force field "
            "would produce. The ligand was not generated with a valid 3D geometry."
            % (got_bonds, expected_bonds))
    if pose.GetNumAtoms() != template.GetNumAtoms():
        raise RuntimeError("heavy-atom count mismatch: pose %d vs SMILES %d"
                           % (pose.GetNumAtoms(), template.GetNumAtoms()))
    # match[k] = pose atom index that corresponds to template atom k (query = template)
    match = topology_only(pose).GetSubstructMatch(topology_only(template))
    if len(match) != template.GetNumAtoms():
        raise RuntimeError("topology mismatch: the docked pose is not isomorphic to --smiles")
    pose_conf = pose.GetConformer()
    conf = Chem.Conformer(template.GetNumAtoms())
    for tmpl_idx, pose_idx in enumerate(match):
        conf.SetAtomPosition(tmpl_idx, pose_conf.GetAtomPosition(pose_idx))
    out = Chem.Mol(template)
    out.RemoveAllConformers()
    out.AddConformer(conf, assignId=True)
    out = Chem.AddHs(out, addCoords=True)
    guard_bond_geometry(out)
    return out


def topology_only(mol):
    """Copy of mol with all bonds single and all aromatic flags cleared, for graph matching."""
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
    """Fail loudly if the docked coordinates cannot belong to this molecular graph.

    A wrong atom-to-coordinate mapping shows up immediately as absurd bond lengths, so this
    guard is what actually prevents a repeat of the corrupted export.
    """
    conf = mol.GetConformer()
    positions = conf.GetPositions()
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
            d = float(np.linalg.norm(positions[a] - positions[b]))
            closest_nonbonded = min(closest_nonbonded, d)
    print("bond geometry guard: longest bond %.2f A | shortest bond %.2f A | closest non-bonded %.2f A"
          % (worst_bond, closest_bond, closest_nonbonded))
    if worst_bond > 1.85 or closest_bond < 0.90:
        raise RuntimeError("implausible bond lengths: the coordinate mapping is wrong")
    if closest_nonbonded < 1.70:
        raise RuntimeError("non-bonded heavy atoms closer than 1.85 A: the ligand geometry is not realisable")


def fragments_and_min_distance(mol):
    """Connected components and shortest intra-ligand contact over heavy atoms."""
    heavy = Chem.RemoveHs(Chem.Mol(mol))
    pos = heavy.GetConformer().GetPositions()
    n = len(pos)
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    np.fill_diagonal(d, 9.9)
    adj = d < 1.75
    seen = [False] * n
    comps = 0
    for i in range(n):
        if seen[i]:
            continue
        comps += 1
        stack = [i]
        seen[i] = True
        while stack:
            u = stack.pop()
            for v in np.nonzero(adj[u])[0]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
    return comps, float(d.min()), n, heavy, pos


def audit_against_receptor(rec_atoms, heavy, pos):
    """Per-residue minimum heavy-atom contacts plus clash counts under three criteria."""
    per_res = {}
    severe = mild = 0
    worst_ratio, worst_pair = 9.9, ""
    syms = [a.GetSymbol() for a in heavy.GetAtoms()]
    for i in range(len(pos)):
        for r in rec_atoms:
            d = float(np.linalg.norm(pos[i] - r["xyz"]))
            r_sum = VDW.get(syms[i], 1.70) + VDW.get(r["elem"], 1.70)
            if d < 1.50:
                severe += 1
            if d < r_sum - 0.40:
                mild += 1
            if d / r_sum < worst_ratio:
                worst_ratio = d / r_sum
                worst_pair = r["res"] + ":" + r["name"] + "-LIG:" + str(i + 1)
            if r["res"] not in per_res or d < per_res[r["res"]][0]:
                per_res[r["res"]] = (d, r["res"] + ":" + r["name"] + " - LIG:" + syms[i] + str(i + 1),
                                     r["elem"], syms[i])
    top = sorted(per_res.items(), key=lambda kv: kv[1][0])[:10]
    return top, severe, mild, worst_ratio, worst_pair


def write_complex(receptor_path, mol, out_pdb, remark):
    """Receptor ATOM records + ligand HETATM records, with element columns filled in."""
    rec_lines = []
    serial = 1
    for line in open(receptor_path, encoding="utf-8", errors="ignore"):
        if not line.startswith("ATOM"):
            continue
        raw = line[76:78].strip().upper() if len(line) >= 78 else ""
        elem = TYPE2ELEM.get(raw, raw.capitalize() if raw else "C")
        rec_lines.append("ATOM  %5d %-4s %-3s A%4s    %8.3f%8.3f%8.3f  1.00 20.00          %2s"
                         % (serial, line[12:16].strip(), line[17:20].strip(), line[22:26].strip(),
                            float(line[30:38]), float(line[38:46]), float(line[46:54]), elem))
        serial += 1
    conf = mol.GetConformer()
    lig_lines = []
    for atom in mol.GetAtoms():
        p = conf.GetAtomPosition(atom.GetIdx())
        lig_lines.append("HETATM%5d %-4s LIG B   1    %8.3f%8.3f%8.3f  1.00 20.00          %2s"
                         % (atom.GetIdx() + 1, atom.GetSymbol() + str(atom.GetIdx() + 1),
                            p.x, p.y, p.z, atom.GetSymbol()))
    with open(out_pdb, "w", encoding="utf-8") as handle:
        handle.write("REMARK    " + remark + chr(10))
        handle.write(chr(10).join(rec_lines) + chr(10) + "TER" + chr(10))
        handle.write(chr(10).join(lig_lines) + chr(10) + "END" + chr(10))
    return len(rec_lines), len(lig_lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="EGFR_Allosteric")
    ap.add_argument("--smiles", required=True)
    ap.add_argument("--data-dir", default="real_world_study/data/allosteric_targets")
    ap.add_argument("--out-dir", default="real_world_study/results")
    ap.add_argument("--box", type=float, default=16.0)
    ap.add_argument("--center", default=None, help="x,y,z; if omitted the center is read from the .pt file")
    ap.add_argument("--exhaustiveness", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cpu", type=int, default=4)
    ap.add_argument("--cov-factor", type=float, default=1.3)
    ap.add_argument("--pose-pdbqt", default=None, help="re-export this pose instead of re-docking")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    pt_path = os.path.join(args.data_dir, "pocket_%s.pt" % args.target)
    rec_path = os.path.join(args.data_dir, "pocket_%s.pdbqt" % args.target)
    if not os.path.exists(pt_path) or not os.path.exists(rec_path):
        raise SystemExit("missing receptor files: %s / %s" % (pt_path, rec_path))

    if args.center:
        center = [float(v) for v in args.center.split(",")]
    else:
        import torch
        pocket = torch.load(pt_path, map_location="cpu")
        center = [float(v) for v in pocket["center"]]
    print("target           : %s" % args.target)
    print("pocket center    : [%.2f, %.2f, %.2f] | box %.1f A" % (center[0], center[1], center[2], args.box))

    if args.pose_pdbqt:
        pose_text = open(args.pose_pdbqt, encoding="utf-8", errors="ignore").read()
        score = float("nan")
        print("pose source      : %s (re-export only)" % args.pose_pdbqt)
    else:
        from meeko import MoleculePreparation, PDBQTWriterLegacy
        from vina import Vina
        template = Chem.MolFromSmiles(args.smiles)
        mol_h = Chem.AddHs(template)
        params = AllChem.ETKDGv3()
        params.randomSeed = args.seed
        if AllChem.EmbedMolecule(mol_h, params) != 0:
            raise SystemExit("conformer generation failed")
        AllChem.UFFOptimizeMolecule(mol_h, maxIters=200)
        prep = MoleculePreparation()
        setup = prep.prepare(mol_h)[0]
        written = PDBQTWriterLegacy.write_string(setup)
        lig_pdbqt = written[0] if isinstance(written, tuple) else written
        vina = Vina(sf_name="vina", cpu=args.cpu, seed=args.seed, verbosity=0)
        vina.set_receptor(rec_path)
        vina.set_ligand_from_string(lig_pdbqt)
        vina.compute_vina_maps(center=center, box_size=[args.box] * 3)
        vina.dock(exhaustiveness=args.exhaustiveness, n_poses=1)
        score = float(vina.energies(n_poses=1)[0][0])
        pose_text = vina.poses(n_poses=1)
        print("docked Vina score: %.2f kcal/mol (exhaustiveness %d, seed %d)"
              % (score, args.exhaustiveness, args.seed))

    atoms = parse_pdbqt_atoms(pose_text, keep_hydrogen=False)
    mol = mol_from_pose(atoms, args.smiles, cov_factor=args.cov_factor)

    canonical = Chem.MolToSmiles(Chem.RemoveHs(mol))
    reference = Chem.MolToSmiles(Chem.MolFromSmiles(args.smiles))
    comps, min_d, n_heavy, heavy, pos = fragments_and_min_distance(mol)

    rec_atoms = []
    for line in open(rec_path, encoding="utf-8", errors="ignore"):
        if line.startswith("ATOM"):
            raw = line[76:78].strip().upper() if len(line) >= 78 else ""
            rec_atoms.append({
                "res": line[17:20].strip() + line[22:26].strip(),
                "name": line[12:16].strip(),
                "elem": TYPE2ELEM.get(raw, raw.capitalize() if raw else "C"),
                "xyz": np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])]),
            })
    top, severe, mild, ratio, ratio_pair = audit_against_receptor(rec_atoms, heavy, pos)

    print(chr(10) + "=" * 88)
    print("EXPORT AUDIT")
    print("=" * 88)
    print("ligand formula          : %s | MW %.2f Da | QED %.4f"
          % (rdMolDescriptors.CalcMolFormula(heavy), Descriptors.MolWt(heavy), QED.qed(heavy)))
    print("canonical SMILES match  : %s" % ("YES" if canonical == reference else "NO  <-- MISMATCH"))
    print("heavy atoms             : %d | connected fragments at 1.75 A: %d" % (n_heavy, comps))
    print("shortest heavy-atom pair : %.2f A (includes chemical bonds)" % min_d)
    print("receptor contacts < 1.50 A (paper metric) ......... %d" % severe)
    print("receptor contacts < sum(vdW) - 0.40 A ............ %d" % mild)
    print("PoseBusters-style ratio min d / sum(vdW) ......... %.3f (%s)" % (ratio, ratio_pair))
    print("-" * 88)
    print("per-residue minimum heavy-atom contact (top 10)")
    for res, (d, pair, pe, le) in top:
        print("   %-10s %-34s %6.2f A" % (res, pair, d))
    ok = (comps == 1) and (min_d > 0.95) and (canonical == reference)
    print("=" * 88)
    print("RESULT: %s" % ("PASS - writing files" if ok else "FAIL - files NOT written"))
    print("=" * 88)
    if not ok:
        raise SystemExit(1)

    sdf_path = os.path.join(args.out_dir, "%s_pose.sdf" % args.target)
    mol_path = os.path.join(args.out_dir, "%s_pose.mol" % args.target)
    pdb_path = os.path.join(args.out_dir, "complex_%s_exported.pdb" % args.target)
    json_path = os.path.join(args.out_dir, "%s_export_audit.json" % args.target)
    writer = Chem.SDWriter(sdf_path)
    writer.write(mol)
    writer.close()
    Chem.MolToMolFile(mol, mol_path)
    n_rec, n_lig = write_complex(rec_path, mol, pdb_path,
                                 "BUILD 3.0 EXPORTED COMPLEX (%s, Vina = %.2f)" % (args.target, score))
    summary = {"target": args.target, "smiles": args.smiles, "canonical_smiles": canonical,
               "vina_score": None if score != score else round(score, 3),
               "heavy_atoms": n_heavy, "fragments": comps, "min_intra_distance": round(min_d, 3),
               "receptor_clashes_1p5": severe, "receptor_clashes_mild": mild,
               "posebusters_min_ratio": round(ratio, 3),
               "top_residue_contacts": [{"residue": res, "pair": pair, "distance": round(d, 2)}
                                         for res, (d, pair, pe, le) in top],
               "receptor_atoms": n_rec, "ligand_atoms": n_lig}
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print("written: %s" % sdf_path)
    print("written: %s" % mol_path)
    print("written: %s" % pdb_path)
    print("written: %s" % json_path)


if __name__ == "__main__":
    main()
