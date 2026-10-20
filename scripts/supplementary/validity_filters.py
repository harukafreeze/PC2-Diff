# scripts/supplementary/validity_filters.py
# =====================================================================================
# Hard validity filters for generated allosteric leads.
#
# Why: rdMolDescriptors.CalcNumBridgeheadAtoms returns 0 for fused small-ring cages
# (benzene fused simultaneously to a three- and a four-membered ring). Such a structure
# passed the previous filter and was reported as the allosteric lead, yet its relaxed
# geometry keeps two carbons 1.68 A apart, which no force field can produce.
#
# Use:
#   from validity_filters import is_acceptable, explain
#   ok, reason, info = is_acceptable(mol)          # mol must carry 3D coordinates
#   python scripts/supplementary/validity_filters.py "SMILES1" "SMILES2" ...
# =====================================================================================
import sys
from collections import Counter

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

MIN_NONBONDED = 1.90   # A, closest allowed distance between non-bonded heavy atoms
MAX_RING_SIZE = 8


def ring_report(mol):
    """Ring sizes from the symmetrized SSSR. Never use FastFindRings for this: it reports
    a single 8- or 9-membered ring for fused systems such as indane or benzocyclobutene,
    which then trips the ring-size rule with a false positive."""
    return [list(r) for r in Chem.GetSymmSSSR(mol)]


def topology_reason(mol):
    """Deterministic graph-level rejection of cages and other unusable polycyclics."""
    rings = ring_report(mol)
    if not rings:
        return "no ring"
    membership = Counter()
    for ring in rings:
        for idx in ring:
            membership[idx] += 1
    if any(count >= 3 for count in membership.values()):
        return "an atom belongs to 3 or more rings (cage)"
    bond_in_rings = Counter()
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
    if any(len(ring) > MAX_RING_SIZE for ring in rings):
        return "ring larger than %d" % MAX_RING_SIZE
    return None


def geometry_report(mol, max_iters=500):
    """Relax the conformer and measure the closest non-bonded heavy-atom contact."""
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
    return closest, energy


def is_acceptable(mol):
    """Return (ok, reason, info). mol must have 3D coordinates."""
    reason = topology_reason(mol)
    if reason:
        return False, reason, {}
    try:
        closest, energy = geometry_report(mol)
    except Exception as exc:
        return False, "geometry check failed (%s)" % type(exc).__name__, {}
    info = {"closest_nonbonded": round(closest, 3), "abs_uff": round(energy, 1)}
    if closest < MIN_NONBONDED:
        return False, "non-bonded heavy atoms %.2f A apart" % closest, info
    return True, "ok", info


def main():
    for smiles in sys.argv[1:]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print("unparsable: " + smiles)
            continue
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        AllChem.UFFOptimizeMolecule(mol, maxIters=1000)
        ok, reason, info = is_acceptable(Chem.RemoveHs(mol))
        print("%-9s %-34s %s" % ("ACCEPT" if ok else "REJECT", reason, info))
        print("          " + smiles)


if __name__ == "__main__":
    main()
