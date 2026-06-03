"""Fit the multi-channel WPH prior on the physical CIB patches.

Multi-channel analog of build_wph_prior.py. Features are assembled by the
SAME routine the training graph uses (make_wph_features_multi_fn), so the
prior, the per-sample targets, and the training-time predictions all share
one feature ordering:

    [ auto(95), auto(150), auto(220), cross(95,150), cross(95,220), cross(150,220) ]

D4 augmentation applies the SAME rotation/reflection to all bands, preserving
the inter-band relationship that the cross-WPH blocks measure.

Saves (mean, cov) of the real-valued multi-channel feature vector plus the
WPH config (J, L, dn, A, M, N) and channel bookkeeping needed downstream.
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


def _d4_orbit_multi(stack: np.ndarray) -> np.ndarray:
    """8 D4 transforms applied identically across channels. (C,H,W)->(8,C,H,W)."""
    out = []
    for flip in (False, True):
        base = stack[:, :, ::-1] if flip else stack
        for k in range(4):
            out.append(np.rot90(base, k=k, axes=(1, 2)))
    return np.stack(out, axis=0).astype(np.float32, copy=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/train_cib.npz"))
    parser.add_argument("--out", type=Path, default=Path("runs/wph_prior_cib.npz"))
    parser.add_argument("--n-patches", type=int, default=2000)
    parser.add_argument("--J", type=int, default=6)
    parser.add_argument("--L", type=int, default=4)
    parser.add_argument("--dn", type=int, default=0)
    parser.add_argument("--A", type=int, default=4)
    parser.add_argument("--augment-d4", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--wph-chunk-size", type=int, default=1)
    parser.add_argument("--cov-ridge-rel", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    import jax.numpy as jnp

    print(f"loading {args.data} ...")
    data = np.load(args.data)
    if "cib_patches" not in data.files:
        raise RuntimeError(
            f"{args.data} has no 'cib_patches'; rebuild with build_dataset_cib.py."
        )
    patches = np.asarray(data["cib_patches"], dtype=np.float32)   # (N, C, H, W)
    if patches.ndim != 4:
        raise ValueError(f"expected (N, C, H, W), got {patches.shape}")
    n, c, h, w = patches.shape
    if h != w:
        raise ValueError(f"non-square patches not supported: {h}x{w}")
    bands = [str(b) for b in data["bands"]] if "bands" in data.files else list(range(c))
    print(f"  {n} patches, {c} bands {bands}, {h}x{w}")

    rng = np.random.default_rng(args.seed)
    take = min(args.n_patches, n)
    idx = rng.choice(n, size=take, replace=False)
    selected = patches[idx]                                       # (take, C, H, W)
    print(f"  selected {take} patches for prior fit")

    if args.augment_d4:
        aug = np.stack([_d4_orbit_multi(p) for p in selected], axis=0)   # (take,8,C,H,W)
        all_patches = aug.reshape(-1, c, h, w)
        print(f"  D4 augmentation: {all_patches.shape[0]} patches total")
    else:
        all_patches = selected

    cfg = WPHConfig(M=h, N=w, J=args.J, L=args.L, dn=args.dn, A=args.A)
    op = WPHOp.build(cfg)
    n_total_per_block = op.n_total
    n_blocks = c + len(channel_pairs(c))
    print(f"  WPH: J={cfg.J} L={cfg.L} dn={cfg.dn} A={cfg.A}, "
          f"n_total/block={n_total_per_block}, n_blocks={n_blocks} "
          f"({c} auto + {len(channel_pairs(c))} cross)")
    features_fn = make_wph_features_multi_fn(
        op, n_channels=c, chunk_size=args.wph_chunk_size, checkpoint=False,
    )

    chunks = []
    for start in range(0, all_patches.shape[0], args.batch_size):
        batch = all_patches[start : start + args.batch_size]      # (b, C, H, W)
        batch_nhwc = np.transpose(batch, (0, 2, 3, 1))            # (b, H, W, C)
        feats = np.asarray(features_fn(jnp.asarray(batch_nhwc)))
        chunks.append(feats)
        done = min(start + args.batch_size, all_patches.shape[0])
        print(f"    {done}/{all_patches.shape[0]} done", end="\r", flush=True)
    print()
    s_all = np.concatenate(chunks, axis=0)
    print(f"  features: shape={s_all.shape} (expect n_features="
          f"{n_blocks * 2 * n_total_per_block})")

    mean = s_all.mean(axis=0)
    centered = s_all - mean
    nn = centered.shape[0]
    cov = (centered.T @ centered) / max(nn - 1, 1)
    if args.cov_ridge_rel:
        diag = float(np.trace(cov) / cov.shape[0])
        cov = cov + args.cov_ridge_rel * diag * np.eye(cov.shape[0], dtype=np.float64)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        mean=mean.astype(np.float64),
        cov=cov.astype(np.float64),
        n_samples=np.int64(nn),
        n_channels=np.int64(c),
        bands=np.asarray(bands),
        n_total_per_block=np.int64(n_total_per_block),
        n_features=np.int64(s_all.shape[1]),
        J=np.int64(args.J), L=np.int64(args.L),
        dn=np.int64(args.dn), A=np.int64(args.A),
        M=np.int64(h), N=np.int64(w),
        d4_augmented=np.bool_(args.augment_d4),
    )
    print(f"saved multi-channel WPH prior -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
