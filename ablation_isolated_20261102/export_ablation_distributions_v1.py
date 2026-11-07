"""Export existing five-group observations for plotting; no new experiment."""
import argparse,csv,hashlib,json,sys
from pathlib import Path
from statistics import mean
from rdkit import Chem
from rdkit.Chem import RDConfig
sys.dont_write_bytecode=True
sys.path.append(str(Path(RDConfig.RDContribDir)/'SA_Score'))
import sascorer

ROOT=Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite',required=True)
    p.add_argument('--out',required=True)
    a=p.parse_args()
    suite=Path(a.suite).resolve();out=Path(a.out).resolve()
    if not out.is_relative_to(ROOT) or out.exists() or out.with_suffix('.json').exists():
        p.error('Use a new output filename inside the isolated folder')
    records=[];metadata=dict(rdkit=Chem.rdBase.rdkitVersion,groups=[],script_sha=sha(Path(__file__)))
    for variant in range(1,6):
        folder=suite/f'variant_{variant}'
        path=folder/'samples.json';rows=json.loads(path.read_text())
        summary=json.loads((folder/'summary.json').read_text())
        if len(rows)!=1000 or len({(r['pocket'],r['sample']) for r in rows})!=1000:
            raise ValueError(f'Variant{variant}: expected1000 unique observations')
        values=[]
        for row in rows:
            qed,sa=None,None
            if row.get('smiles'):
                if row.get('qed') is None:raise ValueError('Graph has no recorded QED')
                qed=float(row['qed'])
                if row.get('sa') is not None:
                    sa=float(row['sa'])
                else:
                    mol=Chem.MolFromSmiles(row['smiles'])
                    if mol is None:raise ValueError('Cannot compute SA from stored graph')
                    # Identical operation to the completed suite's extended_summary.
                    sa=float((10-sascorer.calculateScore(mol))/9)
                values.append((qed,sa))
            records.append(dict(variant=variant,pocket=row['pocket'],sample=row['sample'],
                                valid=int(bool(row.get('valid'))),connected=int(bool(row.get('connected'))),
                                qed=qed,sa=sa))
        stats=dict(variant=variant,attempts=len(rows),descriptor_count=len(values),
                   valid_connected=sum(bool(r.get('valid')) and bool(r.get('connected')) for r in rows),
                   qed_mean=mean(v[0] for v in values),sa_mean=mean(v[1] for v in values),
                   samples_sha=sha(path),summary_sha=sha(folder/'summary.json'))
        for metric in ('qed','sa'):
            if len(values)!=summary[metric]['count'] or abs(stats[f'{metric}_mean']-summary[metric]['mean'])>1e-8:
                raise ValueError(f'Variant{variant}: {metric} distribution does not match completed summary')
        metadata['groups'].append(stats)
        print(f"Variant{variant}: {len(rows)} attempts, {len(values)} descriptor records; means match",flush=True)
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['variant','pocket','sample','valid','connected','qed','sa'])
        writer.writeheader();writer.writerows(records)
    out.with_suffix('.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print('Exported',len(records),'rows:',out)


if __name__=='__main__':main()
