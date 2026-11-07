"""Conservative, bounded graph repair for failed reconstructions only.

Experimental: never imported by the production engine automatically.
"""
from collections import deque
from rdkit import Chem

LIMITS = {1: 1, 6: 4, 7: 4, 8: 2, 9: 1, 15: 5, 16: 6, 17: 1, 35: 1, 53: 1}
FORBIDDEN = {(8,8),(8,9),(9,9),(8,17),(8,35),(8,53),(7,9),(9,16),
             (9,17),(9,35),(9,53),(17,17),(35,35),(53,53)}

def diagnostics(mol):
    if mol is None:
        return ['no_molecule']
    try:
        return [f'{x.GetType()}: {x.Message()}' for x in Chem.DetectChemistryProblems(Chem.Mol(mol))]
    except Exception as exc:
        return [str(exc)]

def strict_accept(mol):
    m = Chem.Mol(mol)
    try:
        Chem.SanitizeMol(m)
        if len(Chem.GetMolFrags(m)) != 1:
            return None
        for a in m.GetAtoms():
            if sum(b.GetBondTypeAsDouble() for b in a.GetBonds()) > LIMITS.get(a.GetAtomicNum(), 4):
                return None
        for b in m.GetBonds():
            pair = tuple(sorted((b.GetBeginAtom().GetAtomicNum(), b.GetEndAtom().GetAtomicNum())))
            if pair in FORBIDDEN:
                return None
        return m
    except Exception:
        return None

def key(m):
    return (tuple(a.GetFormalCharge() for a in m.GetAtoms()),
            tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx(), str(b.GetBondType())) for b in m.GetBonds())))

def repair_failed(mol, max_states=64, max_edits=2):
    """Keep atoms/coordinates/core edges. Repair only local overvalence.

    Try quaternary-N charge correction, noncore bond-order reduction, or
    noncore edge deletion or chemically constrained bridge-endpoint replacement.
    No resampling,
    filtering, atom replacement, or changes to an already accepted graph.
    """
    if mol is None:
        return None, {'reason': 'no_molecule', 'states': 0}
    base = Chem.Mol(mol)
    queue = deque([(base, [])]); seen = {key(base)}; count = 0
    seed = base.GetNumAtoms() >= 6 and all(base.GetAtomWithIdx(i).GetAtomicNum() == 6 for i in range(6))
    while queue and count < max_states:
        m, edits = queue.popleft(); count += 1
        accepted = strict_accept(m)
        if accepted is not None:
            return accepted, {'reason': 'strictly_accepted', 'states': count, 'edits': edits}
        if len(edits) >= max_edits:
            continue
        candidates = []
        rewires = []
        for a in m.GetAtoms():
            i = a.GetIdx(); val = sum(b.GetBondTypeAsDouble() for b in a.GetBonds())
            if a.GetAtomicNum() == 7 and val == 4 and a.GetFormalCharge() == 0 and not a.GetIsAromatic():
                rw = Chem.RWMol(m); rw.GetAtomWithIdx(i).SetFormalCharge(1)
                candidates.append((rw.GetMol(), f'N{i}:charge+1'))
            limit = 3 if a.GetAtomicNum() == 7 and a.GetFormalCharge() == 0 else LIMITS.get(a.GetAtomicNum(), 4)
            if val <= limit:
                continue
            # Deterministic index ordering, never ranked using docking results.
            for b in sorted(a.GetBonds(), key=lambda b: b.GetIdx()):
                u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
                if b.GetIsAromatic() or (seed and u < 6 and v < 6):
                    continue
                order = b.GetBondTypeAsDouble()
                if order in (2, 3):
                    rw = Chem.RWMol(m)
                    rw.GetBondBetweenAtoms(u,v).SetBondType(Chem.BondType.SINGLE if order == 2 else Chem.BondType.DOUBLE)
                    candidates.append((rw.GetMol(), f'bond{u}-{v}:order-1'))
                rw = Chem.RWMol(m); rw.RemoveBond(u,v); candidate = rw.GetMol()
                if len(Chem.GetMolFrags(candidate)) == 1:
                    candidates.append((candidate, f'bond{u}-{v}:remove_nonbridge'))
                else:
                    # Removing an overvalent bridge is allowed only as an atomic
                    # remove-and-reconnect operation that retains every atom.
                    parts = Chem.GetMolFrags(candidate)
                    if len(parts) != 2:
                        continue
                    options = []
                    for left in parts[0]:
                        for right in parts[1]:
                            if tuple(sorted((left,right))) == tuple(sorted((u,v))):
                                continue
                            if seed and left < 6 and right < 6:
                                continue
                            atoms = [candidate.GetAtomWithIdx(k) for k in (left,right)]
                            if tuple(sorted(a.GetAtomicNum() for a in atoms)) in FORBIDDEN:
                                continue
                            allowed = True
                            for atom in atoms:
                                z = atom.GetAtomicNum()
                                cap = 3 if z == 7 and atom.GetFormalCharge() == 0 else LIMITS.get(z,4)
                                if sum(e.GetBondTypeAsDouble() for e in atom.GetBonds()) + 1 > cap:
                                    allowed = False
                                if seed and atom.GetIdx() < 6:
                                    outside = sum(e.GetOtherAtomIdx(atom.GetIdx()) >= 6 for e in atom.GetBonds())
                                    if outside >= 1: allowed = False
                            if not allowed: continue
                            conf = candidate.GetConformer() if candidate.GetNumConformers() else None
                            distance = (conf.GetAtomPosition(left)-conf.GetAtomPosition(right)).Length() if conf else float('inf')
                            radii = Chem.GetPeriodicTable()
                            cutoff = sum(radii.GetRcovalent(a.GetAtomicNum()) for a in atoms) + 0.35
                            if distance > cutoff:
                                continue
                            options.append((distance,left,right))
                    for _,left,right in sorted(options)[:8]:
                        joined = Chem.RWMol(candidate); joined.AddBond(left,right,Chem.BondType.SINGLE)
                        rewires.append((joined.GetMol(),f'bridge{u}-{v}:replace_with{left}-{right}'))
        candidates.extend(rewires)
        for candidate, edit in candidates:
            k = key(candidate)
            if k not in seen:
                seen.add(k); queue.append((candidate, edits + [edit]))
    return None, {'reason': 'no_valid_local_repair', 'states': count}
