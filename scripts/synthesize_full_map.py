"""Generate a full-footprint Compton-y map by tiling FM-sampled patches.

Lays the model's 188x188 patches on a single global ZEA grid covering the
specified RA/Dec footprint. Patches overlap by ``--overlap`` fraction
(default 25%); their contributions are blended with a separable cosine
window into a global accumulator. Saved as FITS with proper WCS.

Limitations
-----------
This is *tile-and-blend*, not coherent generation. Each patch is drawn
independently from the FM prior, so:
- Small-scale (<5 deg) structure -- including cluster cores, halo
  morphology, and the high-ell power spectrum -- is correct.
- Large-scale modes crossing patch boundaries are NOT constrained. Cosine
  blending hides the seams smoothly but doesn't enforce coherent
  large-scale power. For analyses sensitive to >5 deg modes (large-scale
  cross-correlation with kSZ/CMB-lensing), use a coherent-stitching
  variant (Repaint-style boundary conditioning during ODE integration --
  not implemented here).

Geometry caveat: every patch is placed at a pixel offset in the *single*
global ZEA, rather than each generating a local-tangent ZEA at its own
center. For SPT-1500 (extent ~50 deg from the tangent at the footprint
center) the ZEA projection distortion is ~10-15% in radial pixel scale
at the edges, so edge patches are subtly stretched relative to training.
Acceptable for a v1; if the edge regions look wrong you can either
shrink the footprint or implement per-patch reprojection.
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

from agorasynth.data import gaussianized_to_physical  # noqa: E402
from agorasynth.flow_matching import sample_euler_one_step, sample_heun  # noqa: E402
from agorasynth.unet import UNet  # noqa: E402


def _cosine_window_2d(h: int, w: int, edge_frac: float = 0.25) -> np.ndarray:
    """Separable raised-cosine taper. ``edge_frac`` of pixels on each side
    are tapered from 0 -> 1; interior is 1.
    """

    def axis(L: int) -> np.ndarray:
        n_edge = max(1, int(L * edge_frac))
        w = np.ones(L, dtype=np.float64)
        ramp = 0.5 * (1 - np.cos(np.pi * np.arange(n_edge) / n_edge))
        w[:n_edge] = ramp
        w[-n_edge:] = ramp[::-1]
        return w

    return np.outer(axis(h), axis(w))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True,
                        help="FM checkpoint .pkl from train_fm_wph.py.")
    parser.add_argument("--ra-min", type=float, default=-50.0)
    parser.add_argument("--ra-max", type=float, default=50.0)
    parser.add_argument("--dec-min", type=float, default=-70.0)
    parser.add_argument("--dec-max", type=float, default=-40.0)
    parser.add_argument("--ra-center", type=float, default=None,
                        help="RA of the global ZEA tangent point (default: footprint mid).")
    parser.add_argument("--dec-center", type=float, default=None,
                        help="Dec of the global ZEA tangent point (default: footprint mid).")
    parser.add_argument("--pixel-arcmin", type=float, default=None,
                        help="Output pixel size. Default: read from checkpoint.")
    parser.add_argument("--overlap", type=float, default=0.25,
                        help="Patch overlap fraction (0=no overlap, 0.5=half).")
    parser.add_argument("--edge-frac", type=float, default=0.25,
                        help="Cosine-window taper width as fraction of patch.")
    parser.add_argument("--n-steps", type=int, default=30,
                        help="Heun ODE steps per patch.")
    parser.add_argument("--sampler", choices=("heun", "euler"), default="heun",
                        help="heun integrates the ODE; euler is one network pass.")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Patches sampled per JIT'd batch.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("samples/full_map.fits"))
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
    pixel_arcmin = float(ckpt["pixel_size_arcmin"])
    if args.pixel_arcmin is not None and abs(args.pixel_arcmin - pixel_arcmin) > 1e-6:
        print(
            f"  WARNING: --pixel-arcmin {args.pixel_arcmin} differs from checkpoint's "
            f"{pixel_arcmin}; using checkpoint's (resampling not implemented)."
        )
    patch_h, patch_w = cfg["patch_hw"]
    print(f"  patch shape: {patch_h}x{patch_w}, pixel size {pixel_arcmin} arcmin")

    model = UNet(
        channels=tuple(cfg["channels"]),
        t_dim=cfg["t_dim"],
        bottleneck_blocks=cfg["bottleneck_blocks"],
        out_channels=cfg["out_channels"],
    )
    apply_fn = model.apply

    # ------------------------------------------------------------------
    # Build the global ZEA grid
    # ------------------------------------------------------------------
    from astropy.io import fits
    from astropy.wcs import WCS

    ra_c = args.ra_center if args.ra_center is not None else 0.5 * (args.ra_min + args.ra_max)
    dec_c = (
        args.dec_center if args.dec_center is not None
        else 0.5 * (args.dec_min + args.dec_max)
    )
    pix_deg = pixel_arcmin / 60.0

    # Size the global array by projecting the four footprint corners onto
    # ZEA at (ra_c, dec_c) and taking the bounding box. We pad by patch/2
    # on each side so the corners are fully covered.
    corners_ra = np.array([args.ra_min, args.ra_min, args.ra_max, args.ra_max])
    corners_dec = np.array([args.dec_min, args.dec_max, args.dec_min, args.dec_max])

    probe_wcs = WCS(naxis=2)
    probe_wcs.wcs.crpix = [1.0, 1.0]
    probe_wcs.wcs.cdelt = [-pix_deg, pix_deg]
    probe_wcs.wcs.crval = [ra_c, dec_c]
    probe_wcs.wcs.ctype = ["RA---ZEA", "DEC--ZEA"]
    px_corners, py_corners = probe_wcs.wcs_world2pix(corners_ra, corners_dec, 0)
    margin = max(patch_h, patch_w) // 2
    n_pix_x = int(np.ceil(px_corners.max() - px_corners.min())) + 2 * margin
    n_pix_y = int(np.ceil(py_corners.max() - py_corners.min())) + 2 * margin

    # Re-set crpix so the tangent point sits in the middle of the global array.
    crpix_x = -px_corners.min() + margin + 1
    crpix_y = -py_corners.min() + margin + 1
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [crpix_x, crpix_y]
    wcs.wcs.cdelt = [-pix_deg, pix_deg]
    wcs.wcs.crval = [ra_c, dec_c]
    wcs.wcs.ctype = ["RA---ZEA", "DEC--ZEA"]
    print(
        f"  global ZEA tangent at (RA, Dec) = ({ra_c:.2f}, {dec_c:.2f})"
    )
    print(
        f"  global grid: {n_pix_x} x {n_pix_y} pixels "
        f"(~{n_pix_x * pix_deg:.1f} deg x {n_pix_y * pix_deg:.1f} deg)"
    )

    # ------------------------------------------------------------------
    # Patch placement
    # ------------------------------------------------------------------
    stride_x = max(1, int(round(patch_w * (1.0 - args.overlap))))
    stride_y = max(1, int(round(patch_h * (1.0 - args.overlap))))
    xs = list(range(0, n_pix_x - patch_w + 1, stride_x))
    if not xs or xs[-1] + patch_w < n_pix_x:
        xs.append(max(0, n_pix_x - patch_w))
    ys = list(range(0, n_pix_y - patch_h + 1, stride_y))
    if not ys or ys[-1] + patch_h < n_pix_y:
        ys.append(max(0, n_pix_y - patch_h))
    positions = [(x, y) for y in ys for x in xs]
    print(
        f"  patches: {len(xs)} x {len(ys)} = {len(positions)} total "
        f"(stride {stride_x}x{stride_y}, overlap {args.overlap:.2f})"
    )

    accumulator = np.zeros((n_pix_y, n_pix_x), dtype=np.float64)
    weights = np.zeros((n_pix_y, n_pix_x), dtype=np.float64)
    win = _cosine_window_2d(patch_h, patch_w, edge_frac=args.edge_frac)

    # ------------------------------------------------------------------
    # Sample patches in batches and blend
    # ------------------------------------------------------------------
    seed = args.seed
    t0 = time.time()
    n_done = 0
    for start in range(0, len(positions), args.batch_size):
        chunk = positions[start : start + args.batch_size]
        if args.sampler == "euler":
            x_batch = sample_euler_one_step(
                apply_fn, params,
                n_samples=len(chunk),
                spatial_shape=(patch_h, patch_w),
                n_channels=cfg["out_channels"],
                seed=seed,
            )
        else:
            x_batch = sample_heun(
                apply_fn, params,
                n_samples=len(chunk),
                spatial_shape=(patch_h, patch_w),
                n_channels=cfg["out_channels"],
                n_steps=args.n_steps,
                seed=seed,
            )
        seed += 10_000  # advance seed across batches
        x_np = np.asarray(x_batch)[..., 0]                     # (B, H, W)
        y_np = np.asarray(gaussianized_to_physical(
            jnp.asarray(x_np), quantile_grid, z_grid, y0=y0
        ))
        for k, (px, py) in enumerate(chunk):
            patch = y_np[k]
            accumulator[py : py + patch_h, px : px + patch_w] += win * patch
            weights[py : py + patch_h, px : px + patch_w] += win
        n_done += len(chunk)
        elapsed = (time.time() - t0) / 60.0
        rate = n_done / max(elapsed, 1e-6)
        eta = (len(positions) - n_done) / max(rate, 1e-6)
        print(
            f"    {n_done}/{len(positions)} done  "
            f"({elapsed:.1f} min, ~{eta:.1f} min remaining)",
            flush=True,
        )

    final = accumulator / np.maximum(weights, 1e-30)
    valid = weights > 0
    print(
        f"  final map: shape={final.shape}, "
        f"covered fraction={float(valid.mean()):.3f}, "
        f"y range [{float(final[valid].min()):.3e}, "
        f"{float(final[valid].max()):.3e}]"
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    primary = fits.PrimaryHDU(data=final.astype(np.float32), header=wcs.to_header())
    primary.header["BUNIT"] = "compton_y"
    primary.header["EXTNAME"] = "Y_MAP"
    primary.header["RA_MIN"] = args.ra_min
    primary.header["RA_MAX"] = args.ra_max
    primary.header["DEC_MIN"] = args.dec_min
    primary.header["DEC_MAX"] = args.dec_max
    primary.header["OVERLAP"] = args.overlap
    primary.header["NSTEPS"] = args.n_steps
    primary.header["SAMPLER"] = args.sampler
    primary.header["SEED"] = args.seed
    primary.header["CKPT"] = str(args.ckpt)

    weight_hdu = fits.ImageHDU(
        data=weights.astype(np.float32), header=wcs.to_header(), name="WEIGHT"
    )
    fits.HDUList([primary, weight_hdu]).writeto(args.out, overwrite=True)
    print(f"saved -> {args.out}")
    print("  HDU 0: Y_MAP (the synthesized Compton-y, in physical units)")
    print("  HDU 1: WEIGHT (cosine-window accumulated weight; 0 = uncovered)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
