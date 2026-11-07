"""Report the recorded main-benchmark and five-group ablation results."""
import json
from pathlib import Path
from statistics import mean

def main():
    root = Path(__file__).resolve().parents[1]
    directory = root / "results/manuscript"
    rows = json.loads((directory / "crossdocked_molecules.json").read_text(encoding="utf-8"))
    assert len(rows) == len({(r["pocket"], r["sample"]) for r in rows}) == 1000
    result = {
        "molecules": len(rows),
        "pockets": len({r["pocket"] for r in rows}),
        "valid_pct": 100 * mean(r["valid"] for r in rows),
        "connected_pct": 100 * mean(r["connected"] for r in rows),
        "mean_metrics": {key: mean(r[key] for r in rows)
                         for key in ("qed", "sa", "vina", "strain", "clashes")},
    }
    groups = json.loads((directory / "ablation_summary.json").read_text(encoding="utf-8"))
    result["ablation"] = [
        {"variant": group["variant"],
         **{key: value for key, value in group["summary"].items()
            if key not in {"errors", "failures", "timings", "timing"}}}
        for group in groups
    ]
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
