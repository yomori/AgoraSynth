"""Extract ZEA patches from a HEALPix Compton-y map; gaussianize; save .npz.

Identical data product to AgoraScore's build_dataset.py: patches in
N(0,1)-marginal "gaussianized log-y" space, plus the (z_grid, quantile_grid)
needed to invert back to physical y.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from agorasynth.data import (  # noqa: E402
    extract_patches,
    gaussianize_patches,
    random_sphere_directions,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", dest="map_path", type=Path, required=True,
                        help="Path to a HEALPix .fits y-map.")
    parser.add_argument("--n-patches", type=int, default=10000)
    parser.add_argument("--patch-size-deg", type=float, default=5.0)
    parser.add_argument("--pixel-size-arcmin", type=float, default=1.6)
    parser.add_argument("--y0", type=float, default=1e-7,
                        help="Floor added before the log transform.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nest", action="store_true",
                        help="Map is in NESTED ordering (default: RING).")
    parser.add_argument("--out", type=Path, default=Path("data/train.npz"))
    args = parser.parse_args(argv)

    import healpy as hp

    print(f"loading {args.map_path} ...")
    hmap = hp.read_map(str(args.map_path))
    nside = hp.get_nside(hmap)
    print(f"  Nside={nside}, {hmap.size:,} pixels, dtype={hmap.dtype}")

    pixel_size_deg = args.pixel_size_arcmin / 60.0
    n_pix = int(round(args.patch_size_deg / pixel_size_deg))
    print(f"  patch grid: {n_pix}x{n_pix} px ({n_pix * pixel_size_deg:.3f} deg)")

    ra_centers, dec_centers = random_sphere_directions(args.n_patches, seed=args.seed)
    print(f"sampled {args.n_patches} random sky directions")

    patches = extract_patches(
        hmap, ra_centers, dec_centers,
        n_pix=n_pix, pixel_size_arcmin=args.pixel_size_arcmin,
        nest=args.nest, progress=True,
    )
    print(
        f"  patches: shape={patches.shape} dtype={patches.dtype} "
        f"min={patches.min():.3e} median={float(np.median(patches)):.3e} "
        f"max={patches.max():.3e}"
    )

    x_train, quantile_grid, z_grid = gaussianize_patches(patches, y0=args.y0)
    print(
        f"  gaussianized: x range [{x_train.min():.3f}, {x_train.max():.3f}], "
        f"log-y quantile range [{quantile_grid[0]:.3f}, {quantile_grid[-1]:.3f}]"
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        x_train=x_train[:, None, :, :],          # (N, 1, H, W) NCHW for convenience
        y_patches=patches,                        # (N, H, W) physical y, for WPH prior fit
        quantile_grid=quantile_grid,
        z_grid=z_grid,
        y0=np.float64(args.y0),
        ra_centers=ra_centers,
        dec_centers=dec_centers,
        patch_size_deg=np.float64(args.patch_size_deg),
        pixel_size_arcmin=np.float64(args.pixel_size_arcmin),
        seed=np.int64(args.seed),
    )
    print(f"saved {x_train.shape[0]} training patches -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
