"""Generate paired (z, x) data from a trained FM model for rectified-flow reflow.

For each ``z ~ N(0, I)``, we ODE-integrate the trained velocity field from
t=0 to t=1 to get the deterministic sample ``x = sample_heun(z)``. The
resulting ``(z, x)`` pairs are *coupled* (z deterministically maps to x),
so retraining flow matching on these pairs as the new (x_0, x_1) endpoints
produces a velocity field whose ODE trajectories are nearly straight --
sample-able in 1-4 steps instead of 30.

Saves to a single .npz with the same metadata fields as data/train.npz so
train_reflow.py can drop in.
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from agorasynth.flow_matching import sample_heun  # noqa: E402
from agorasynth.unet import UNet  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True,
                        help="FM checkpoint to use as the source velocity field.")
    parser.add_argument("--n-pairs", type=int, default=10000,
                        help="Number of (z, x) pairs to generate.")
    parser.add_argument("--n-steps", type=int, default=30,
                        help="Heun ODE steps used to map z -> x.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("data/reflow_pairs.npz"))
    args = parser.parse_args(argv)

    print(f"jax devices: {jax.devices()}")
    print(f"loading {args.ckpt} ...")
    with open(args.ckpt, "rb") as f:
        ckpt = pickle.load(f)
    params = ckpt["params"]
    cfg = ckpt["config"]
    quantile_grid = np.asarray(ckpt["quantile_grid"], dtype=np.float64)
    z_grid = np.asarray(ckpt["z_grid"], dtype=np.float64)
    y0 = float(ckpt["y0"])
    pixel_size_arcmin = float(ckpt["pixel_size_arcmin"])
    patch_size_deg = float(ckpt["patch_size_deg"])
    h, w = cfg["patch_hw"]
    c = cfg["out_channels"]
    print(f"  patch={h}x{w}, channels={cfg['channels']}")

    model = UNet(
        channels=tuple(cfg["channels"]),
        t_dim=cfg["t_dim"],
        bottleneck_blocks=cfg["bottleneck_blocks"],
        out_channels=c,
    )
    apply_fn = model.apply

    rng = np.random.default_rng(args.seed)
    n_done = 0
    z_chunks = []
    x_chunks = []
    t0 = time.time()
    while n_done < args.n_pairs:
        b = min(args.batch_size, args.n_pairs - n_done)
        z_np = rng.standard_normal((b, h, w, c)).astype(np.float32)
        z_j = jnp.asarray(z_np)
        x_j = sample_heun(
            apply_fn, params,
            n_steps=args.n_steps,
            init_noise=z_j,
        )
        z_chunks.append(z_np)
        x_chunks.append(np.asarray(x_j))
        n_done += b
        elapsed = (time.time() - t0) / 60.0
        rate = n_done / max(elapsed, 1e-6)
        eta = (args.n_pairs - n_done) / max(rate, 1e-6)
        print(
            f"    {n_done}/{args.n_pairs} pairs done "
            f"({elapsed:.1f} min, ~{eta:.1f} min remaining)",
            flush=True,
        )

    z_all = np.concatenate(z_chunks, axis=0)        # (N, H, W, C)
    x_all = np.concatenate(x_chunks, axis=0)        # (N, H, W, C)
    print(
        f"  generated z: {z_all.shape}  range "
        f"[{z_all.min():.3f}, {z_all.max():.3f}]"
    )
    print(
        f"  generated x: {x_all.shape}  range "
        f"[{x_all.min():.3f}, {x_all.max():.3f}]"
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        z_pairs=z_all,
        x_pairs=x_all,
        quantile_grid=quantile_grid,
        z_grid=z_grid,
        y0=np.float64(y0),
        patch_size_deg=np.float64(patch_size_deg),
        pixel_size_arcmin=np.float64(pixel_size_arcmin),
        source_ckpt=str(args.ckpt),
        source_n_steps=np.int64(args.n_steps),
    )
    print(f"saved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
