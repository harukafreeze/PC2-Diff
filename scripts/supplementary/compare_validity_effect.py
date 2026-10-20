# scripts/supplementary/compare_validity_effect.py
# =====================================================================================
# Control experiment for the allosteric section of the paper.
#
# Question: does the chemical-validity funnel merely trim a few outliers, or does it
# change which molecules a docking-only workflow would have promoted?
#
# Method: take the same cached samples used for the reported campaign, split them into
#   (a) molecules that survive the validity gates, and
#   (b) molecules rejected as bridged cages / fused cyclopropanes,
# dock an equal-sized random sample of each group with identical Vina settings, and
# compare the two score distributions plus the receptor-clash rate of the poses.
#
# Requires screen_allosteric_candidates.py in the same folder (it reuses its filters and
# its docking worker, so both groups are treated identically).
#
# Usage (from the repository root, inside build_env):
#   python scripts/supplementary/compare_validity_effect.py --per_group 300
#
# Outputs (in --out_dir, default real_world_study/results/allosteric_screen):
#   validity_effect.csv        one row per docked molecule with its group and metrics
#   validity_effect.json       per-group summary (n, best, median, worst, clash-free rate)
# =====================================================================================
import argparse, json, os, sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import screen_allosteric_candidates as S
except ImportError:
    raise SystemExit("copy screen_allosteric_candidates.py next to this file first")


CAGE_MARKERS = ("3 or more rings", "three-membered ring fused")


def group_of(row):
    """Return valid, cage, or other for a scored row from the screening filters."""
    if row.get("passed"):
        return "valid"
    reason = (row.get("topology_reject") or "")
    if any(marker in reason for marker in CAGE_MARKERS):
        return "cage"
    return "other"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="EGFR_Allosteric")
    parser.add_argument("--data_dir", default="real_world_study/data/allosteric_targets")
    parser.add_argument("--out_dir", default="real_world_study/results/allosteric_screen")
    parser.add_argument("--cache", default="real_world_study/results/allosteric_screen/gen_cache.npz")
    parser.add_argument("--per_group", type=int, default=300)
    parser.add_argument("--box", type=float, default=16.0)
    parser.add_argument("--exhaustiveness", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    if not os.path.exists(args.cache):
        raise SystemExit("cache not found: %s (run screen_allosteric_candidates.py first)" % args.cache)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    workers = args.workers if args.workers > 0 else max(1, 8)
    os.makedirs(args.out_dir, exist_ok=True)

    blob = np.load(args.cache)
    coords, types = blob["coords"], blob["types"]
    pocket_path = os.path.join(args.data_dir, "pocket_%s.pt" % args.target)
    receptor_path = os.path.join(args.data_dir, "pocket_%s.pdbqt" % args.target)
    pocket = torch.load(pocket_path, map_location="cpu")
    center = [float(v) for v in pocket["center"]]
    p_pos_np = pocket["pocket_pos"].cpu().numpy()
    p_types_np = np.argmax(pocket["pocket_h"].cpu().numpy(), axis=-1)
    print("cached samples: %d | pocket atoms: %d | %d workers" % (len(coords), len(p_pos_np), workers))

    packs = [(i, coords[i], types[i], p_pos_np, p_types_np) for i in range(len(coords))]
    print("classifying every cached sample with the validity filters ...")
    rows = S.run_pool(packs, S.filter_worker, workers, "classify")
    by_group = {"valid": [], "cage": [], "other": []}
    for row in rows:
        by_group[group_of(row)].append(row["idx"])
    print("group sizes: valid %d | cage %d | other %d"
          % (len(by_group["valid"]), len(by_group["cage"]), len(by_group["other"])))

    rng = np.random.RandomState(args.seed)
    chosen = {}
    for name in ("valid", "cage"):
        idxs = by_group[name]
        if len(idxs) > args.per_group:
            idxs = list(rng.choice(idxs, size=args.per_group, replace=False))
        chosen[name] = [int(i) for i in idxs]
        print("docking %s group: %d molecules" % (name, len(chosen[name])))

    dock_packs = []
    for name, idxs in chosen.items():
        for idx in idxs:
            dock_packs.append((idx, coords[idx], types[idx], p_pos_np, p_types_np,
                               receptor_path, center, args.box, args.exhaustiveness, args.seed))
    scored = {}
    for result in S.run_pool(dock_packs, S.dock_worker, workers, "docking"):
        scored[result["idx"]] = result

    receptor = S.receptor_atoms(receptor_path)
    table = []
    for name, idxs in chosen.items():
        for idx in idxs:
            hit = scored.get(idx)
            if not hit or hit["vina"] is None or not hit["pose"]:
                continue
            row = next((r for r in rows if r["idx"] == idx), None)
            record = {"idx": idx, "group": name, "vina": round(hit["vina"], 3),
                      "clashes_docked": None, "posebusters_min_ratio": None,
                      "pose_invalid": None, "smiles": None if row is None else row.get("smiles")}
            try:
                mol = S.mol_from_pose(S.parse_pdbqt_atoms(hit["pose"]), record["smiles"])
            except Exception as exc:
                record["pose_invalid"] = type(exc).__name__
                table.append(record)
                continue
            components, closest_pair, n_heavy, heavy, positions = S.fragments_and_min_distance(mol)
            top, severe, mild, ratio, ratio_pair = S.audit_against_receptor(receptor, heavy, positions)
            record["clashes_docked"] = severe
            record["posebusters_min_ratio"] = round(ratio, 3)
            table.append(record)

    summary = {}
    for name in ("valid", "cage"):
        scores = sorted(r["vina"] for r in table if r["group"] == name)
        clean = [r["vina"] for r in table if r["group"] == name
                 and r.get("clashes_docked") is not None and r["clashes_docked"] <= 3]
        summary[name] = {
            "docked": len(scores),
            "clash_free": len(clean),
            "best": scores[0] if scores else None,
            "median": scores[len(scores) // 2] if scores else None,
            "worst": scores[-1] if scores else None,
            "best_clash_free": min(clean) if clean else None,
        }
    print()
    print("%-8s %8s %10s %10s %10s %14s" % ("group", "docked", "best", "median", "worst", "best clash-free"))
    for name in ("valid", "cage"):
        s = summary[name]
        fmt = lambda v: "n/a" if v is None else "%.2f" % v
        print("%-8s %8d %10s %10s %10s %14s"
              % (name, s["docked"], fmt(s["best"]), fmt(s["median"]), fmt(s["worst"]),
                 fmt(s["best_clash_free"])))
    columns = ["idx", "group", "vina", "clashes_docked", "posebusters_min_ratio", "smiles"]
    csv_path = os.path.join(args.out_dir, "validity_effect.csv")
    with open(csv_path, "w", encoding="utf-8") as handle:
        handle.write(",".join(columns) + "\n")
        for record in sorted(table, key=lambda r: (r["group"], r["vina"])):
            handle.write(",".join("" if record.get(c) is None else str(record[c]) for c in columns) + "\n")
    with open(os.path.join(args.out_dir, "validity_effect.json"), "w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "per_group": args.per_group,
                   "cage_markers": list(CAGE_MARKERS),
                   "group_sizes": {k: len(v) for k, v in by_group.items()}}, handle, indent=2)
    print("written: %s" % csv_path)
    print("written: %s" % os.path.join(args.out_dir, "validity_effect.json"))


if __name__ == "__main__":
    main()
