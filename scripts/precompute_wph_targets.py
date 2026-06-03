"""Precompute the WPH feature vector for each training patch.

For per-sample WPH loss training, we need WPH(y_target_i) for every training
patch i. Since these are constant across training (the targets don't change),
we compute them once and save alongside the dataset.

The WPH config is taken from the prior file so the precomputed features
are compatible with the prior's mean/cov metadata. We also save the
per-feature std across the dataset, which is used to normalize the
per-sample WPH loss so each feature contributes on the same scale.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from agorasynth.wph import (  # noqa: E402
    WPHOp,
    WPHPriorStats,
    compute_S_batch,
    to_real_features,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/train.npz"),
                        help="Dataset .npz from build_dataset.py (must have y_patches).")
    parser.add_argument("--wph-prior", type=Path, default=Path("runs/wph_prior.npz"),
                        help="WPH prior .npz; uses its config so features are "
                             "compatible with the existing prior.")
    parser.add_argument("--out", type=Path, default=Path("data/wph_targets.npz"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--std-floor-rel", type=float, default=1e-3,
                        help="Per-feature std floor relative to median std; prevents "
                             "1/std exploding for near-constant features.")
    args = parser.parse_args(argv)

    import jax.numpy as jnp

    print(f"loading {args.data} ...")
    data = np.load(args.data)
    if "y_patches" not in data.files:
        raise RuntimeError(
            f"{args.data} has no 'y_patches' field; rebuild with the AgoraSynth "
            "build_dataset.py."
        )
    y_patches = np.asarray(data["y_patches"], dtype=np.float32)
    n, h, w = y_patches.shape
    if h != w:
        raise ValueError(f"non-square patches not supported: {h}x{w}")
    print(f"  {n} patches at {h}x{w}")

    print(f"loading WPH config from {args.wph_prior} ...")
    prior = WPHPriorStats.load(args.wph_prior)
    cfg = prior.config
    if cfg.M != h or cfg.N != w:
        raise ValueError(
            f"prior cfg M=N={cfg.M} doesn't match patch size {h}x{w}"
        )
    op = WPHOp.build(cfg)
    print(
        f"  WPH config: M=N={cfg.M}, J={cfg.J}, L={cfg.L}, "
        f"dn={cfg.dn}, A={cfg.A}, n_total={op.n_total}"
    )

    chunks = []
    for start in range(0, n, args.batch_size):
        chunk = y_patches[start : start + args.batch_size]
        s_complex = compute_S_batch(op, jnp.asarray(chunk))
        s_real = to_real_features(s_complex)
        chunks.append(np.asarray(s_real))
        done = min(start + args.batch_size, n)
        print(f"    {done}/{n} done", end="\r", flush=True)
    print()
    F_targets = np.concatenate(chunks, axis=0).astype(np.float32)
    print(f"  features: shape={F_targets.shape}")

    std_per_feature = F_targets.std(axis=0).astype(np.float64)
    median_std = float(np.median(std_per_feature))
    floor = args.std_floor_rel * median_std if median_std > 0 else 1e-12
    std_per_feature = np.maximum(std_per_feature, floor)
    inv_std = (1.0 / std_per_feature).astype(np.float32)
    n_floored = int((F_targets.std(axis=0) < floor).sum())
    if n_floored:
        print(
            f"  floored {n_floored}/{F_targets.shape[1]} features "
            f"with std < {floor:.3e}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        F_targets=F_targets,
        std_per_feature=std_per_feature.astype(np.float32),
        inv_std=inv_std,
        n_features=np.int64(F_targets.shape[1]),
    )
    print(f"saved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
