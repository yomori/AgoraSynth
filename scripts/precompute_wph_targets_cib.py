"""Precompute the multi-channel WPH feature vector for each CIB training patch.

Multi-channel analog of precompute_wph_targets.py. Uses the WPH config stored
in the CIB prior so the per-sample targets share the prior's feature ordering
(per-channel auto blocks followed by cross-band blocks). Saves the
(n_patches, n_features) array plus the per-feature std used to normalize the
per-sample loss.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from agorasynth.flow_matching import channel_pairs, make_wph_features_multi_fn  # noqa: E402
from agorasynth.wph import WPHConfig, WPHOp  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/train_cib.npz"))
    parser.add_argument("--wph-prior", type=Path, default=Path("runs/wph_prior_cib.npz"))
    parser.add_argument("--out", type=Path, default=Path("data/wph_targets_cib.npz"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--wph-chunk-size", type=int, default=1)
    parser.add_argument("--std-floor-rel", type=float, default=1e-3)
    parser.add_argument("--start", type=int, default=0,
                        help="First patch index to process (for sharded builds).")
    parser.add_argument("--stop", type=int, default=-1,
                        help="One-past-last patch index (-1 = end).")
    parser.add_argument("--raw-out", type=Path, default=None,
                        help="If set, save raw F_targets[start:stop] to this .npy "
                             "and skip std/final-npz. Concatenate shards + compute "
                             "the global inv_std separately. Lets a long run be split "
                             "into fresh-process shards (avoids per-process buildup).")
    args = parser.parse_args(argv)

    import jax.numpy as jnp

    print(f"loading {args.data} ...")
    data = np.load(args.data)
    if "cib_patches" not in data.files:
        raise RuntimeError(
            f"{args.data} has no 'cib_patches'; rebuild with build_dataset_cib.py."
        )
    patches = np.asarray(data["cib_patches"], dtype=np.float32)   # (N, C, H, W)
    n_full = patches.shape[0]
    stop = n_full if args.stop < 0 else min(args.stop, n_full)
    patches = patches[args.start:stop]                            # shard slice (or full)
    n, c, h, w = patches.shape
    if h != w:
        raise ValueError(f"non-square patches not supported: {h}x{w}")
    print(f"  {n} patches [{args.start}:{stop}] of {n_full}, {c} bands, {h}x{w}")

    print(f"loading WPH config from {args.wph_prior} ...")
    prior = np.load(args.wph_prior)
    cfg = WPHConfig(
        M=int(prior["M"]), N=int(prior["N"]), J=int(prior["J"]),
        L=int(prior["L"]), dn=int(prior["dn"]), A=int(prior["A"]),
    )
    n_channels = int(prior["n_channels"])
    if n_channels != c:
        raise ValueError(
            f"prior n_channels={n_channels} != dataset channels={c}"
        )
    if cfg.M != h or cfg.N != w:
        raise ValueError(
            f"prior patch size {cfg.M}x{cfg.N} doesn't match dataset patch size {h}x{w}"
        )
    op = WPHOp.build(cfg)
    n_blocks = c + len(channel_pairs(c))
    print(f"  WPH: J={cfg.J} L={cfg.L} dn={cfg.dn} A={cfg.A}, "
          f"n_total/block={op.n_total}, n_blocks={n_blocks}")
    features_fn = make_wph_features_multi_fn(
        op, n_channels=c, chunk_size=args.wph_chunk_size, checkpoint=False,
    )

    chunks = []
    for start in range(0, n, args.batch_size):
        batch = patches[start : start + args.batch_size]
        batch_nhwc = np.transpose(batch, (0, 2, 3, 1))            # (b, H, W, C)
        feats = np.asarray(features_fn(jnp.asarray(batch_nhwc)))
        chunks.append(feats.astype(np.float32))
        done = min(start + args.batch_size, n)
        print(f"    {done}/{n} done", end="\r", flush=True)
    print()
    F_targets = np.concatenate(chunks, axis=0).astype(np.float32)
    print(f"  features: shape={F_targets.shape}")

    if "n_features" in prior.files and int(prior["n_features"]) != F_targets.shape[1]:
        raise ValueError(
            f"feature count {F_targets.shape[1]} != prior's "
            f"{int(prior['n_features'])} — config mismatch."
        )

    if args.raw_out is not None:
        args.raw_out.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.raw_out, F_targets)
        print(f"saved raw shard [{args.start}:{stop}] {F_targets.shape} -> {args.raw_out}")
        return 0

    std_per_feature = F_targets.std(axis=0).astype(np.float64)
    median_std = float(np.median(std_per_feature))
    floor = args.std_floor_rel * median_std if median_std > 0 else 1e-12
    std_per_feature = np.maximum(std_per_feature, floor)
    inv_std = (1.0 / std_per_feature).astype(np.float32)
    n_floored = int((F_targets.std(axis=0) < floor).sum())
    if n_floored:
        print(f"  floored {n_floored}/{F_targets.shape[1]} features with std < {floor:.3e}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        F_targets=F_targets,
        std_per_feature=std_per_feature.astype(np.float32),
        inv_std=inv_std,
        n_features=np.int64(F_targets.shape[1]),
        n_channels=np.int64(c),
    )
    print(f"saved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
