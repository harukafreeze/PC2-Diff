"""At most two graph-preserving local torsion candidates, deterministic."""
import numpy as np
from rdkit import Chem
from rdkit.Chem import Lipinski, AllChem

def candidate_strain(mol):
    """Same 50+300 UFF definition; errors propagate instead of a fixed score."""
    work=Chem.AddHs(Chem.Mol(mol),addCoords=True)
    Chem.FastFindRings(work)
    if not AllChem.UFFHasAllMoleculeParams(work):raise ValueError('UFF parameters unavailable')
    AllChem.UFFOptimizeMolecule(work,maxIters=50)
    ff=AllChem.UFFGetMoleculeForceField(work)
    if ff is None:raise ValueError('UFF construction failed')
    initial=ff.CalcEnergy();ff.Minimize(maxIts=300);final=ff.CalcEnergy()
    if not np.isfinite(initial) or not np.isfinite(final):raise ValueError('Nonfinite strain energy')
    return float(min(1000.,max(0.,initial-final)))

def local_candidates(mol):
    xyz=np.asarray(mol.GetConformer().GetPositions())
    core=set(range(6)) if mol.GetNumAtoms()>=6 and all(mol.GetAtomWithIdx(i).GetAtomicNum()==6 for i in range(6)) and all(mol.GetBondBetweenAtoms(i,(i+1)%6) is not None for i in range(6)) else set()
    choices=[]
    for i,j in sorted(set(tuple(sorted(p)) for p in mol.GetSubstructMatches(Lipinski.RotatableBondSmarts))):
        b=mol.GetBondBetweenAtoms(i,j)
        if b.IsInRing() or b.GetBondType()!=Chem.BondType.SINGLE:continue
        rw=Chem.RWMol(mol);rw.RemoveBond(i,j)
        parts=Chem.GetMolFrags(rw)
        if len(parts)!=2:continue
        for branch in parts:
            if core.intersection(branch):continue
            u,v=(i,j) if j in branch else (j,i)
            if len(branch)<2:continue
            choices.append((-min(len(branch),mol.GetNumAtoms()-len(branch)),len(branch),u,v,tuple(branch)))
    if not choices:return []
    _,_,u,v,branch=min(choices)
    axis=xyz[v]-xyz[u];length=np.linalg.norm(axis)
    if length<1e-8:return []
    axis/=length;relative=xyz[list(branch)]-xyz[u]
    radial=np.linalg.norm(np.cross(relative,axis),axis=1).max()
    if radial<1e-8:return []
    angle=min(np.deg2rad(15.),2*np.arcsin(min(1.,.75/(2*radial))))
    result=[]
    for sign in [-1,1]:
        theta=sign*angle
        rotated=relative*np.cos(theta)+np.cross(axis,relative)*np.sin(theta)+np.outer(relative@axis,axis)*(1-np.cos(theta))
        new=xyz.copy();new[list(branch)]=xyz[u]+rotated
        candidate=Chem.Mol(mol)
        for k,pos in enumerate(new):candidate.GetConformer().SetAtomPosition(k,tuple(float(a) for a in pos))
        assert Chem.MolToSmiles(candidate,isomericSmiles=True)==Chem.MolToSmiles(mol,isomericSmiles=True)
        assert np.max(np.linalg.norm(new-xyz,axis=1))<=.75+1e-8
        if core:assert np.array_equal(new[sorted(core)],xyz[sorted(core)])
        result.append((candidate,dict(bond=[u,v],angle_degrees=float(np.rad2deg(theta)),max_shift=float(np.linalg.norm(new-xyz,axis=1).max()))))
    return result
