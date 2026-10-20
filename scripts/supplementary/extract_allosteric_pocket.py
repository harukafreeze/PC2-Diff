# scripts/supplementary/extract_allosteric_pocket.py
# =====================================================================================
# Generalised allosteric pocket extraction (any PDB entry + any co-crystal ligand).
#
# Reproduces, for an arbitrary target, exactly what extract_true_allosteric_5d41.py did for
# 5D41: take the co-crystallised allosteric ligand, keep the protein atoms within a focused
# shell around it, cap the shell to a fixed atom count, and write the two files the rest of
# the pipeline consumes:
#     pocket_<name>.pt      pocket_pos [N,3], pocket_h [N,6] one-hot {H,C,N,O,S,Se},
#                           pocket_elem, center
#     pocket_<name>.pdbqt   rigid receptor, real residue names and numbers preserved
#
# Real residue labels matter: the earlier 5D41 extractor wrote PRO1..PROn, which is why the
# exported complexes lost their residue identities.
#
# Usage (from the repository root, inside build_env):
#   python scripts/supplementary/extract_allosteric_pocket.py --pdb 5EHR --ligand 5OD --name SHP2_Allosteric
#   python scripts/supplementary/extract_allosteric_pocket.py --pdb 5MO4 --ligand AY7 --name ABL1_Allosteric
#
# The PDB file is downloaded from RCSB if it is not already present in --data_dir.
# =====================================================================================
import argparse, json, os, sys, urllib.request
from collections import Counter

import numpy as np

PROTEIN_ATOM_MAP = {1: 0, 6: 1, 7: 2, 8: 3, 16: 4, 34: 5}   # H, C, N, O, S, Se
PDBQT_TYPE = {1: "HD", 6: "C", 7: "N", 8: "OA", 16: "SA", 34: "SE"}
WATER = {"HOH", "DOD", "WAT"}


def download_pdb(pdb_id, path):
    if os.path.exists(path):
        return path
    url = "https://files.rcsb.org/download/%s.pdb" % pdb_id.upper()
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request) as response, open(path, "wb") as handle:
        handle.write(response.read())
    print("downloaded %s -> %s" % (url, path))
    return path


def parse(path, ligand_code, ligand_chain="auto"):
    """Return protein atoms and the coordinates of ONE copy of the co-crystal ligand.

    Crystal structures frequently contain several copies of the ligand (5EHR holds two). All
    copies would put the pocket centre between them, so the chain is resolved explicitly: by
    default the chain of the first matching HETATM record.
    """
    protein, copies, ligand_resname = [], {}, None
    for line in open(path, encoding="utf-8", errors="ignore"):
        if line.startswith("ATOM"):
            name = line[12:16].strip()
            protein.append({
                "chain": line[21], "resn": line[17:20].strip(), "resid": line[22:26].strip(),
                "name": name, "elem": (line[76:78].strip() or name[0]).capitalize(),
                "xyz": np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])]),
            })
        elif line.startswith("HETATM"):
            code = line[17:20].strip()
            if code == ligand_code:
                copies.setdefault(line[21], []).append(
                    [float(line[30:38]), float(line[38:46]), float(line[46:54])])
                ligand_resname = code
            elif code in WATER:
                continue
    if not copies:
        return protein, np.array([]), None, None
    if ligand_chain == "auto":
        ligand_chain = sorted(copies)[0]
    if ligand_chain not in copies:
        raise SystemExit("ligand %s is not present in chain %r (available: %s)"
                         % (ligand_code, ligand_chain, sorted(copies)))
    if len(copies) > 1:
        print("ligand %s present in %d chains %s -> using chain %s only"
              % (ligand_code, len(copies), sorted(copies), ligand_chain))
    return protein, np.array(copies[ligand_chain]), ligand_resname, ligand_chain


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb", required=True, help="four-character PDB entry id, e.g. 5EHR")
    parser.add_argument("--ligand", required=True, help="co-crystal allosteric ligand code, e.g. 5OD")
    parser.add_argument("--name", required=True, help="pocket name used by the pipeline, e.g. SHP2_Allosteric")
    parser.add_argument("--data_dir", default="real_world_study/data/allosteric_targets")
    parser.add_argument("--radius", type=float, default=10.0, help="shell radius around the ligand")
    parser.add_argument("--max_atoms", type=int, default=256)
    parser.add_argument("--ligand_chain", default="auto", help="chain holding the ligand copy to use")
    parser.add_argument("--dry_run", action="store_true", help="analyse only, write nothing")
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)
    pdb_path = os.path.join(args.data_dir, "%s.pdb" % args.pdb.upper())
    download_pdb(args.pdb, pdb_path)
    protein, ligand, ligand_resname, ligand_chain = parse(pdb_path, args.ligand, args.ligand_chain)
    if len(ligand) == 0:
        raise SystemExit("ligand %s not found in %s" % (args.ligand, pdb_path))
    print("PDB %s | ligand %s (%d atoms, chain %s) | protein atoms %d"
          % (args.pdb.upper(), ligand_resname, len(ligand), ligand_chain, len(protein)))

    centre = ligand.mean(axis=0)
    shell = []
    for atom in protein:
        # distance to the NEAREST ligand atom, which is the criterion the 5D41 pocket used
        distance = float(np.linalg.norm(atom["xyz"] - ligand, axis=1).min())
        if distance <= args.radius:
            shell.append((distance, atom))
    if not shell:
        raise SystemExit("no protein atom within %.1f A of the ligand" % args.radius)
    shell.sort(key=lambda item: item[0])
    if len(shell) > args.max_atoms:
        print("shell contained %d atoms, keeping the %d closest" % (len(shell), args.max_atoms))
        shell = shell[:args.max_atoms]
    atoms = [atom for _, atom in shell]
    print("pocket atoms: %d | member residues: %d | chain(s): %s"
          % (len(atoms), len({a["resn"] + a["resid"] for a in atoms}),
             sorted({a["chain"] for a in atoms})))
    print("ligand centroid: [%.2f, %.2f, %.2f] | shell radius %.1f A"
          % (centre[0], centre[1], centre[2], args.radius))

    if args.dry_run:
        print("dry run: pocket built but not written")
        print("pocket tensor centre would be: [%.2f, %.2f, %.2f]"
              % tuple(np.mean([a["xyz"] for a in atoms], axis=0)))
        return

    import torch
    positions = torch.tensor(np.array([a["xyz"] for a in atoms]), dtype=torch.float32)
    elements = torch.tensor([{"H": 1, "C": 6, "N": 7, "O": 8, "S": 16, "SE": 34}.get(
        a["elem"].upper(), 6) for a in atoms], dtype=torch.long)
    one_hot = torch.zeros(len(atoms), 6, dtype=torch.float32)
    for i, z in enumerate(elements.tolist()):
        one_hot[i, PROTEIN_ATOM_MAP.get(z, 1)] = 1.0
    centre_out = positions.mean(dim=0).numpy().tolist()
    pt_path = os.path.join(args.data_dir, "pocket_%s.pt" % args.name)
    torch.save({"pocket_pos": positions, "pocket_h": one_hot, "pocket_elem": elements,
                "center": centre_out}, pt_path)

    pdbqt_path = os.path.join(args.data_dir, "pocket_%s.pdbqt" % args.name)
    with open(pdbqt_path, "w", encoding="utf-8") as handle:
        handle.write("REMARK    BUILD 3.0 allosteric receptor | PDB %s | ligand %s | %.1f A shell\n"
                     % (args.pdb.upper(), ligand_resname, args.radius))
        serial = 0
        for atom, z in zip(atoms, elements.tolist()):
            atom_type = PDBQT_TYPE.get(z, "C")
            if atom_type == "HD":
                continue
            serial += 1
            handle.write("ATOM  %5d %-4s %-3s %s%4s    %8.3f%8.3f%8.3f  1.00 20.00     0.000 %-2s\n"
                         % (serial, atom["name"][:4], atom["resn"][:3], atom["chain"],
                            atom["resid"], atom["xyz"][0], atom["xyz"][1], atom["xyz"][2], atom_type))
        handle.write("TER\nEND\n")

    summary = {"pdb": args.pdb.upper(), "ligand": ligand_resname, "ligand_chain": ligand_chain,
               "pocket_name": args.name, "pocket_atoms": len(atoms),
               "residues": sorted({a["resn"] + a["resid"] for a in atoms}),
               "centre": [round(v, 3) for v in centre_out], "radius": args.radius,
               "elements": dict(Counter(a["elem"] for a in atoms))}
    with open(os.path.join(args.data_dir, "pocket_%s.json" % args.name), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print("written: %s" % pt_path)
    print("written: %s" % pdbqt_path)
    print("centre of the pocket tensor: [%.2f, %.2f, %.2f]" % tuple(centre_out))
    print("element census: %s" % summary["elements"])


if __name__ == "__main__":
    main()
