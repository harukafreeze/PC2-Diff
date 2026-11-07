"""Frozen two-stage candidate selection used by the balanced workflow."""
def eligible(candidate,baseline):
    return (candidate['vina']<=baseline['vina']+1e-8 and
            candidate['strain']<=baseline['strain']+1e-8 and
            candidate['clashes']<=baseline['clashes'])

def choose_geometry(baseline,alternatives):
    pool=[baseline]+[r for r in alternatives if eligible(r,baseline)]
    return min(pool,key=lambda r:(r['strain'],r['vina']))

def choose_intermediate(baseline,alternatives):
    pool=[baseline]+[r for r in alternatives if eligible(r,baseline) and r['vina']<baseline['vina']-1e-6]
    return min(pool,key=lambda r:r['vina'])
