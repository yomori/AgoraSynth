"""Fit the WPH prior on the physical-y patches saved by build_dataset.py.

The prior is a multivariate Gaussian (mu, Sigma) on the real-valued WPH
feature vector. Used by train_fm_wph.py as the target distribution that
the predicted clean sample should match in feature space.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from agorasynth.wph import (  # noqa: E402
    WPHConfig,
    WPHOp,
    WPHPriorStats,
    compute_S_batch,
    d4_orbit,
    to_real_features,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/train.npz"),
                        help="Dataset .npz from build_dataset.py.")
    parser.add_argument("--out", type=Path, default=Path("runs/wph_prior.npz"))
    parser.add_argument("--n-patches", type=int, default=2000,
                        help="Number of patches sampled from the dataset for the fit.")
    parser.add_argument("--J", type=int, default=6,
                        help="Number of wavelet scales.")
    parser.add_argument("--L", type=int, default=4,
                        help="Number of orientations.")
    parser.add_argument("--dn", type=int, default=0,
                        help="WPH translation radii (0 means tau=(0,0) only).")
    parser.add_argument("--A", type=int, default=4,
                        help="WPH translation angles.")
    parser.add_argument("--augment-d4", action="store_true",
                        help="Augment with the 8 D4-group transforms (8x patches).")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--cov-ridge-rel", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    import jax.numpy as jnp

    print(f"loading {args.data} ...")
    data = np.load(args.data)
    if "y_patches" not in data.files:
        raise RuntimeError(
            f"{args.data} has no 'y_patches' field; rebuild with the AgoraSynth "
            "build_dataset.py (it saves physical-y alongside gaussianized x)."
        )
    y_patches = np.asarray(data["y_patches"], dtype=np.float32)  # (N, H, W)
    N, H, W = y_patches.shape
    if H != W:
        raise ValueError(f"non-square patches not supported: {H}x{W}")
    M = H
    print(f"  loaded {N} patches at {M}x{M}")

    rng = np.random.default_rng(args.seed)
    take = min(args.n_patches, N)
    idx = rng.choice(N, size=take, replace=False)
    selected = y_patches[idx]
    print(f"  selected {take} patches for prior fit")

    if args.augment_d4:
        aug = np.stack([d4_orbit(p) for p in selected], axis=0)        # (take, 8, M, M)
        all_patches = aug.reshape(-1, M, M)
        print(f"  D4 augmentation: {all_patches.shape[0]} patches total")
    else:
        all_patches = selected

    cfg = WPHConfig(M=M, N=M, J=args.J, L=args.L, dn=args.dn, A=args.A)
    print(f"  WPH config: M=N={M}, J={cfg.J}, L={cfg.L}, dn={cfg.dn}, A={cfg.A}")
    op = WPHOp.build(cfg)
    print(f"  n_total WPH coefficients = {op.n_total}")

    chunks = []
    for start in range(0, all_patches.shape[0], args.batch_size):
        chunk = all_patches[start : start + args.batch_size]
        s_complex = compute_S_batch(op, jnp.asarray(chunk))
        s_real = to_real_features(s_complex)
        chunks.append(np.asarray(s_real))
        done = min(start + args.batch_size, all_patches.shape[0])
        print(f"    {done}/{all_patches.shape[0]} done", end="\r", flush=True)
    print()
    s_all = np.concatenate(chunks, axis=0)
    print(f"  computed features: shape={s_all.shape}")

    mean = s_all.mean(axis=0)
    centered = s_all - mean
    n = centered.shape[0]
    cov = (centered.T @ centered) / max(n - 1, 1)
    if args.cov_ridge_rel:
        diag = float(np.trace(cov) / cov.shape[0])
        cov = cov + args.cov_ridge_rel * diag * np.eye(cov.shape[0], dtype=np.float64)
    print(f"  mean: shape={mean.shape}, cov: shape={cov.shape}")

    prior = WPHPriorStats(
        mean=mean.astype(np.float64),
        cov=cov.astype(np.float64),
        n_samples=int(n),
        config=cfg,
        coeff_metadata={"d4_augmented": bool(args.augment_d4), "n_unique_patches": int(take)},
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    prior.save(args.out)
    print(f"saved WPH prior -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
