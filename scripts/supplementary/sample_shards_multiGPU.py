# scripts/supplementary/sample_shards_multiGPU.py
# =====================================================================================
# Multi-GPU sharded sampling for the BUILD 3.0 allosteric campaigns.
#
# Why this exists: sampling is embarrassingly parallel (each batch is drawn independently,
# no gradients, no collective communication), so the single-GPU loop in
# screen_allosteric_candidates.py leaves three of four A800s idle. Sharding across GPUs
# gives an exact speedup, not an approximation: rank i simply draws its own share with its
# own seed stream.
#
# Run one process per GPU (example for 4 GPUs, 100k samples, batch 512):
#
#   for i in 0 1 2 3; do
#     CUDA_VISIBLE_DEVICES=$i python scripts/supplementary/sample_shards_multiGPU.py \
#         --rank $i --world_size 4 --total_samples 100000 --batch_size 512 \
#         --out_dir real_world_study/results/allosteric_screen_100k &
#   done
#   wait
#
# Then merge the shards into the file the screening pipeline expects:
#
#   python scripts/supplementary/sample_shards_multiGPU.py --merge \
#       --out_dir real_world_study/results/allosteric_screen_100k
#
# Outputs: <out_dir>/gen_cache_rank{i}.npz per rank, then <out_dir>/gen_cache.npz
# =====================================================================================
import argparse, glob, math, os, sys, time

import numpy as np


def build_sampler(args, device):
    """Load the checkpoint and build the reverse-SDE sampler (same settings as the paper)."""
    import torch
    import project_build.models.backbones.target_sbdd_backbone  # registers the backbone
    from project_build.core.registry import BACKBONE_REGISTRY
    from project_build.diffusion.sde.vp_sde import DualSpaceVPSDE
    from project_build.diffusion.solvers.reverse_sde_sampler import ReverseSDESampler

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = BACKBONE_REGISTRY.build(checkpoint.get("config", {})["model"]).to(device)
    model.load_state_dict(checkpoint.get("ema_state_dict", checkpoint["model_state_dict"]))
    model.eval()
    sde = DualSpaceVPSDE(schedule_type="cosine").to(device)
    sampler = ReverseSDESampler(sde=sde, num_steps=100, spring_k=0.25, corrector_steps=1,
                                bond_gate_thresh=0.35, enable_svd_seeding=True,
                                enable_self_healing=True, device=device)
    return model, sampler


def merge(args):
    shards = sorted(glob.glob(os.path.join(args.out_dir, "gen_cache_rank*.npz")))
    if not shards:
        raise SystemExit("no shards found in %s" % args.out_dir)
    coords, types = [], []
    for path in shards:
        blob = np.load(path)
        coords.append(blob["coords"])
        types.append(blob["types"])
        print("  %s -> %d samples" % (os.path.basename(path), len(blob["coords"])))
    merged = {"coords": np.concatenate(coords, axis=0), "types": np.concatenate(types, axis=0)}
    out = os.path.join(args.out_dir, "gen_cache.npz")
    np.savez_compressed(out, **merged)
    print("merged %d shards -> %s" % (len(shards), out))
    print("total samples: %d | coords %s | types %s"
          % (len(merged["coords"]), merged["coords"].shape, merged["types"].shape))


def sample_shard(args):
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    share = args.total_samples // args.world_size
    extra = args.total_samples - share * args.world_size
    count = share + (1 if args.rank < extra else 0)
    torch.manual_seed(args.seed + args.rank)
    np.random.seed(args.seed + args.rank)
    print("rank %d/%d | device %s | %d samples | batch %d | seed %d"
          % (args.rank, args.world_size, device, count, args.batch_size, args.seed + args.rank))

    pocket_path = os.path.join(args.data_dir, "pocket_%s.pt" % args.target)
    if not os.path.exists(pocket_path):
        raise SystemExit("missing pocket file: %s" % pocket_path)
    pocket = torch.load(pocket_path, map_location="cpu")
    p_pos, p_h = pocket["pocket_pos"].to(device), pocket["pocket_h"].to(device)
    center = torch.mean(p_pos, dim=0, keepdim=True)
    p_pos_centered = (p_pos - center).unsqueeze(0)
    p_h_batch = p_h.unsqueeze(0)
    model, sampler = build_sampler(args, device)

    started = time.time()
    coords_list, types_list = [], []
    done = 0
    while done < count:
        size = min(args.batch_size, count - done)
        with torch.no_grad():
            sampled, atom_types, _ = sampler.sample(
                model=model, num_samples=size, num_atoms=args.num_atoms,
                num_atom_types=9,
                model_kwargs={"pocket_pos": p_pos_centered.repeat(size, 1, 1),
                              "pocket_h": p_h_batch.repeat(size, 1, 1)})
        coords_list.append(sampled.cpu().numpy() + center.cpu().numpy())
        types_list.append(atom_types.cpu().numpy())
        done += size
        rate = done / max(1e-9, (time.time() - started) / 60.0)
        sys.stdout.write("\r  rank %d: %d/%d (%.1f min, %.0f samples/min)"
                         % (args.rank, done, count, (time.time() - started) / 60.0, rate))
        sys.stdout.flush()
    print()
    out = os.path.join(args.out_dir, "gen_cache_rank%d.npz" % args.rank)
    np.savez_compressed(out, coords=np.concatenate(coords_list, axis=0),
                        types=np.concatenate(types_list, axis=0))
    print("rank %d wrote %s (%d samples, %.1f min)"
          % (args.rank, out, count, (time.time() - started) / 60.0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="EGFR_Allosteric")
    parser.add_argument("--data_dir", default="real_world_study/data/allosteric_targets")
    parser.add_argument("--out_dir", default="real_world_study/results/allosteric_screen_100k")
    parser.add_argument("--checkpoint", default="checkpoints/sbdd/latest_sbdd_model.pt")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--total_samples", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_atoms", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    if args.merge:
        merge(args)
    else:
        sample_shard(args)


if __name__ == "__main__":
    main()
