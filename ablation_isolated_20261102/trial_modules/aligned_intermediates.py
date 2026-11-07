"""Two fixed intermediate poses with a fixed core and bounded contact motion."""
import numpy as np
from rdkit import Chem

def intermediate_candidates(original,relaxed_xyz,receptor):
    xyz=original.GetConformer().GetPositions();target=np.asarray(relaxed_xyz)
    if target.shape!=xyz.shape:raise ValueError('Endpoint atom counts differ')
    if np.linalg.norm(target-xyz,axis=1).max()<1e-3:return []
    core=np.zeros(len(xyz),dtype=bool)
    if len(xyz)>=6 and all(original.GetAtomWithIdx(i).GetAtomicNum()==6 for i in range(6)) and all(original.GetBondBetweenAtoms(i,(i+1)%6) is not None for i in range(6)):
        core[:6]=True
    anchors=np.linalg.norm(xyz[:,None]-np.asarray(receptor)[None],axis=-1).min(1)<3.5
    cap=np.where(core,0.,np.where(anchors,.25,.75))
    results=[]
    for fraction in [.5,.75]:
        delta=fraction*(target-xyz);length=np.linalg.norm(delta,axis=1)
        delta*=np.minimum(1.,cap/np.maximum(length,1e-12))[:,None]
        if np.linalg.norm(delta,axis=1).max()<1e-7:continue
        m=Chem.Mol(original)
        for i,pos in enumerate(xyz+delta):m.GetConformer().SetAtomPosition(i,tuple(float(v) for v in pos))
        assert np.array_equal(m.GetConformer().GetPositions()[core],xyz[core])
        results.append((m,dict(fraction=fraction,max_shift=float(np.linalg.norm(delta,axis=1).max()))))
    return results
