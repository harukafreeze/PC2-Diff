"""Fixed graph, receptor-aware coordinate optimization. Experimental only."""
import numpy as np
from scipy.optimize import minimize
from rdkit import Chem
from rdkit.Chem import AllChem

def contact_term(xyz, receptor, cutoff, strength=20.0):
    diff=xyz[:,None]-receptor[None,:]
    distance=np.sqrt(np.sum(diff*diff,axis=-1)+1e-12)
    overlap=np.maximum(cutoff-distance,0.)
    energy=.5*strength*np.sum(overlap**2)
    gradient=np.sum(-strength*(overlap/distance)[:,:,None]*diff,axis=1)
    return float(energy),gradient

def refine_in_pocket(mol,receptor,receptor_numbers):
    """One fixed setting; keeps atom identities, bonds, and stereo tags.

    MMFF94 (UFF only if unavailable), soft heavy/core positional restraints,
    receptor repulsion, max 500 constrained SLSQP steps. Returns original on rejected
    candidates. Iteration-capped candidates may pass if the explicit gates pass;
    optimizer convergence status is always recorded separately.
    """
    info=dict(accepted=False,max_iterations=500,max_heavy_shift=2.,max_rmsd=.75,
              contact_ratio=.75,receptor_strength=20.,heavy_strength=1.,core_strength=25.)
    try:
        original=Chem.Mol(mol)
        if any(a.GetAtomicNum()==1 for a in original.GetAtoms()):
            raise ValueError('Expected heavy-atom input')
        initial=np.asarray(original.GetConformer().GetPositions()).copy();n=len(initial)
        graph=Chem.MolToSmiles(original,isomericSmiles=True)
        receptor=np.asarray(receptor,dtype=float);receptor_numbers=np.asarray(receptor_numbers,dtype=int)
        if not len(receptor) or receptor.shape!=(len(receptor_numbers),3):raise ValueError('Invalid receptor')
        work=Chem.AddHs(Chem.Mol(original),addCoords=True)
        if AllChem.MMFFHasAllMoleculeParams(work):
            ff=AllChem.MMFFGetMoleculeForceField(work,AllChem.MMFFGetMoleculeProperties(work));info['force_field']='MMFF94'
        elif AllChem.UFFHasAllMoleculeParams(work):
            ff=AllChem.UFFGetMoleculeForceField(work);info['force_field']='UFF'
        else:raise ValueError('Force-field parameters unavailable')
        ff.Initialize();x0=np.asarray(work.GetConformer().GetPositions()).copy()
        pt=Chem.GetPeriodicTable();lr=np.array([pt.GetRvdw(a.GetAtomicNum()) for a in original.GetAtoms()])
        rr=np.array([pt.GetRvdw(int(z)) for z in receptor_numbers]);cutoff=.75*(lr[:,None]+rr[None,:])
        core=np.zeros(n,dtype=bool)
        if n>=6 and all(original.GetAtomWithIdx(i).GetAtomicNum()==6 for i in range(6)) and all(original.GetBondBetweenAtoms(i,(i+1)%6) is not None for i in range(6)):
            core[:6]=True
        force=np.where(core,25.,1.);free=np.where(core,.05,.15)
        def objective(flat):
            points=flat.reshape(-1,3);values=tuple(float(v) for v in flat)
            energy=float(ff.CalcEnergy(values));grad=np.asarray(ff.CalcGrad(values)).reshape(-1,3).copy()
            e,g=contact_term(points[:n],receptor,cutoff);energy+=e;grad[:n]+=g
            delta=points[:n]-initial;distance=np.sqrt(np.sum(delta*delta,axis=1)+1e-12)
            excess=np.maximum(distance-free,0.)
            energy+=.5*np.sum(force*excess**2)
            grad[:n]+=(force*excess/distance)[:,None]*delta
            return float(energy),grad.ravel()
        start,_=objective(x0.ravel())
        internal_before=float(ff.CalcEnergy(tuple(float(v) for v in x0.ravel())))
        # Same acceptance limits, enforced during optimization with a small
        # inward numerical margin. No post-hoc increase of allowed movement.
        shift_bound = 2.0 - 1e-5
        rmsd_bound = .75 - 1e-5
        initial_repulsion = contact_term(initial,receptor,cutoff)[0]
        repulsion_scale = max(1.0, initial_repulsion)
        def feasibility(flat):
            delta=flat.reshape(-1,3)[:n]-initial
            squared=np.sum(delta*delta,axis=1)
            repulsion,_=contact_term(flat.reshape(-1,3)[:n],receptor,cutoff)
            return np.r_[1-squared/shift_bound**2,
                         1-np.sum(squared)/(n*rmsd_bound**2),
                         (initial_repulsion-repulsion)/repulsion_scale]
        def feasibility_jac(flat):
            delta=flat.reshape(-1,3)[:n]-initial
            jac=np.zeros((n+2,len(flat)))
            for i in range(n):jac[i,3*i:3*i+3]=-2*delta[i]/shift_bound**2
            jac[n,:3*n]=(-2*delta/(n*rmsd_bound**2)).ravel()
            _,repulsion_grad=contact_term(flat.reshape(-1,3)[:n],receptor,cutoff)
            jac[n+1,:3*n]=-repulsion_grad.ravel()/repulsion_scale
            return jac
        scale=max(1.,np.sqrt(abs(start)))
        def scaled_objective(flat):
            energy,grad=objective(flat)
            return energy/scale,grad/scale
        # Coordinate boxes are looser than (and implied by) the spherical
        # 2 A constraints; they do not exclude any feasible spherical point.
        bounds=[(float(v-2),float(v+2)) for v in initial.ravel()]+[(None,None)]*(len(x0.ravel())-3*n)
        result=minimize(scaled_objective,x0.ravel(),jac=True,method='SLSQP',bounds=bounds,
            constraints=[dict(type='ineq',fun=feasibility,jac=feasibility_jac)],
            options=dict(maxiter=500,ftol=1e-9))
        result.fun=objective(result.x)[0]
        info.update(solver='SLSQP',explicit_constraints=True,
                    minimum_constraint_value=float(feasibility(result.x).min()),objective_scale=scale)
        final=result.x.reshape(-1,3)[:n];shifts=np.linalg.norm(final-initial,axis=1)
        before=np.linalg.norm(initial[:,None]-receptor[None,:],axis=-1)
        after=np.linalg.norm(final[:,None]-receptor[None,:],axis=-1)
        old_contacts=int((before<1.5).sum());new_contacts=int((after<1.5).sum())
        old_repulsion=contact_term(initial,receptor,cutoff)[0];new_repulsion=contact_term(final,receptor,cutoff)[0]
        internal_after=float(ff.CalcEnergy(tuple(float(v) for v in result.x)))
        info.update(converged=bool(result.success),optimizer_status=int(result.status),message=str(result.message),
            iterations=int(result.nit),objective_before=float(start),objective_after=float(result.fun),
            actual_max_shift=float(shifts.max()),rmsd=float(np.sqrt(np.mean(shifts**2))),
            contacts_before=old_contacts,contacts_after=new_contacts,
            repulsion_before=old_repulsion,repulsion_after=new_repulsion,
            internal_energy_before=internal_before,internal_energy_after=internal_after)
        failures=[]
        if not np.isfinite(final).all() or not np.isfinite(result.fun):failures.append('nonfinite')
        if result.fun>=start:failures.append('objective_not_lower')
        if internal_after>internal_before+1e-6:failures.append('internal_energy_increased')
        for atom in original.GetAtoms():
            if atom.GetChiralTag()==Chem.ChiralType.CHI_UNSPECIFIED:continue
            center=atom.GetIdx();neighbors=sorted(a.GetIdx() for a in work.GetAtomWithIdx(center).GetNeighbors())
            if len(neighbors)>=3:
                old=np.linalg.det(x0[neighbors[:3]]-x0[center])
                new=np.linalg.det(result.x.reshape(-1,3)[neighbors[:3]]-result.x.reshape(-1,3)[center])
                if old*new<=0:failures.append('specified_stereocenter_flip')
        if shifts.max()>2.:failures.append('shift_limit')
        if info['rmsd']>.75:failures.append('rmsd_limit')
        if new_contacts>old_contacts:failures.append('more_contacts')
        if new_repulsion>old_repulsion+1e-6:failures.append('repulsion_increased')
        if failures:
            info['reason']=','.join(failures);return original,info
        candidate=Chem.Mol(original)
        for i,pos in enumerate(final):candidate.GetConformer().SetAtomPosition(i,tuple(float(v) for v in pos))
        if Chem.MolToSmiles(candidate,isomericSmiles=True)!=graph:raise ValueError('Graph/stereo changed')
        info.update(accepted=True,reason='bounded_geometry_accepted')
        return candidate,info
    except Exception as exc:
        info['reason']='returned_original: '+str(exc)
        return Chem.Mol(mol),info
