"""Extract co-located ZEA patches from the 3 SPT-3G CIB maps; gaussianize; save.

Multi-channel analog of build_dataset.py. The 95/150/220 GHz Agora CIB maps
trace the same dusty galaxies, so patches are sampled at the SAME random sky
directions across the three bands and stacked into a (N, C, H, W) array.
Each band is gaussianized independently (its own log-intensity quantile grid).

Default inputs (lensed, no color correction — the only SPT-3G variant present):
  .../cib/nocc/len/spt3g/agora_len_mag_cibmap_spt3g_{095,150,220}ghz.fits

Saved keys: `x_train` (N, C, H, W) gaussianized, `cib_patches` (N, C, H, W)
physical intensity for the WPH prior fit, `quantile_grid` (C, n_quantiles),
`z_grid`, `y0` (C,), `bands`, plus patch metadata.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from agorasynth.data import (  # noqa: E402
    extract_patches_multi,
    gaussianize_patches_multi,
    random_sphere_directions,
)

SPT3G_DIR = Path(
    "/global/cfs/cdirs/mp107c/yomori/agora/products/components/cib/nocc/len/spt3g"
)
DEFAULT_MAPS = [
    SPT3G_DIR / "agora_len_mag_cibmap_spt3g_095ghz.fits",
    SPT3G_DIR / "agora_len_mag_cibmap_spt3g_150ghz.fits",
    SPT3G_DIR / "agora_len_mag_cibmap_spt3g_220ghz.fits",
]
DEFAULT_BANDS = ["095", "150", "220"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maps", type=Path, nargs="+", default=DEFAULT_MAPS,
                        help="HEALPix CIB maps, one per band (co-located).")
    parser.add_argument("--bands", type=str, nargs="+", default=DEFAULT_BANDS,
                        help="Band labels, matched 1:1 with --maps.")
    parser.add_argument("--n-patches", type=int, default=10000)
    parser.add_argument("--patch-size-deg", type=float, default=5.0)
    parser.add_argument("--pixel-size-arcmin", type=float, default=1.6)
    parser.add_argument("--y0", type=float, nargs="+", default=[0.0],
                        help="Per-band log floor; one value (shared) or one per band. "
                             "CIB intensity is strictly positive, so 0.0 is fine.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nest", action="store_true",
                        help="Maps are NESTED ordering (default: RING).")
    parser.add_argument("--out", type=Path, default=Path("data/train_cib.npz"))
    args = parser.parse_args(argv)

    if len(args.bands) != len(args.maps):
        raise SystemExit(
            f"--bands ({len(args.bands)}) must match --maps ({len(args.maps)})"
        )
    c = len(args.maps)
    y0 = np.asarray(args.y0, dtype=np.float64)
    if y0.size not in (1, c):
        raise SystemExit(f"--y0 must have 1 or {c} values, got {y0.size}")
    y0 = np.broadcast_to(y0, (c,)).astype(np.float64)

    import healpy as hp

    pixel_size_deg = args.pixel_size_arcmin / 60.0
    n_pix = int(round(args.patch_size_deg / pixel_size_deg))
    print(f"patch grid: {n_pix}x{n_pix} px ({n_pix * pixel_size_deg:.3f} deg), "
          f"{c} bands {args.bands}")

    hmaps = []
    for band, mp in zip(args.bands, args.maps):
        print(f"loading {band} GHz: {mp} ...")
        hm = hp.read_map(str(mp))
        print(f"  Nside={hp.get_nside(hm)}, {hm.size:,} pixels, "
              f"min={hm.min():.3e} median={float(np.median(hm)):.3e} max={hm.max():.3e}")
        hmaps.append(hm)

    nsides = {hp.get_nside(hm) for hm in hmaps}
    if len(nsides) != 1:
        raise SystemExit(f"maps have differing Nside: {nsides}")

    ra_centers, dec_centers = random_sphere_directions(args.n_patches, seed=args.seed)
    print(f"sampled {args.n_patches} random sky directions (shared across bands)")

    patches = extract_patches_multi(
        hmaps, ra_centers, dec_centers,
        n_pix=n_pix, pixel_size_arcmin=args.pixel_size_arcmin,
        nest=args.nest, progress=True,
    )
    print(f"  patches: shape={patches.shape} dtype={patches.dtype}")
    for ch, band in enumerate(args.bands):
        p = patches[:, ch]
        print(f"    {band} GHz: min={p.min():.3e} median={float(np.median(p)):.3e} "
              f"max={p.max():.3e}")

    # The Agora CIB maps have a few negative pixels (ringing near bright
    # sources), so log(I + y0) needs a per-band offset guaranteeing I + y0 > 0.
    # Bump y0 only where needed; a positive map keeps the user-supplied y0.
    band_min = patches.min(axis=(0, 2, 3))                       # (C,)
    band_med = np.median(patches, axis=(0, 2, 3))               # (C,)
    floor_needed = -band_min + 1e-3 * np.abs(band_med)
    y0 = np.maximum(y0, floor_needed).astype(np.float64)
    print(f"  per-band log offset y0 = {np.array2string(y0, precision=3)}")

    x_train, quantile_grid, z_grid = gaussianize_patches_multi(patches, y0=y0)
    print(f"  gaussianized x: shape={x_train.shape} "
          f"range [{x_train.min():.3f}, {x_train.max():.3f}]")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        x_train=x_train,                          # (N, C, H, W) gaussianized
        cib_patches=patches,                      # (N, C, H, W) physical intensity
        quantile_grid=quantile_grid,              # (C, n_quantiles)
        z_grid=z_grid,                            # (n_quantiles,)
        y0=y0,                                    # (C,)
        bands=np.asarray(args.bands),
        map_paths=np.asarray([str(p) for p in args.maps]),
        ra_centers=ra_centers,
        dec_centers=dec_centers,
        patch_size_deg=np.float64(args.patch_size_deg),
        pixel_size_arcmin=np.float64(args.pixel_size_arcmin),
        seed=np.int64(args.seed),
    )
    print(f"saved {x_train.shape[0]} training patches ({c} bands) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
