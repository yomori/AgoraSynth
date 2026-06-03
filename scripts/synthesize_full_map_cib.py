"""Synthesize full-footprint CIB maps (95/150/220 GHz) by tiling FM patches.

Multi-channel analog of synthesize_full_map.py. One joint sample yields all C
bands per patch, so the bands stay mutually consistent within each patch.
Patches overlap and are cosine-blended into C global accumulators; each band
is written to its own FITS file (``<out_stem>_<band>ghz.fits``) sharing the
same ZEA WCS, with a common WEIGHT map.

Same tile-and-blend caveat as the y version: <patch-size structure is correct,
large-scale modes across patch seams are not constrained.
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

from agorasynth.data import gaussianized_to_physical_multi  # noqa: E402
from agorasynth.flow_matching import sample_euler_one_step, sample_heun  # noqa: E402
from agorasynth.unet import UNet  # noqa: E402


def _cosine_window_2d(h, w, edge_frac=0.25):
    def axis(L):
        n_edge = max(1, int(L * edge_frac))
        win = np.ones(L, dtype=np.float64)
        ramp = 0.5 * (1 - np.cos(np.pi * np.arange(n_edge) / n_edge))
        win[:n_edge] = ramp
        win[-n_edge:] = ramp[::-1]
        return win
    return np.outer(axis(h), axis(w))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--ra-min", type=float, default=-50.0)
    parser.add_argument("--ra-max", type=float, default=50.0)
    parser.add_argument("--dec-min", type=float, default=-70.0)
    parser.add_argument("--dec-max", type=float, default=-40.0)
    parser.add_argument("--ra-center", type=float, default=None)
    parser.add_argument("--dec-center", type=float, default=None)
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--edge-frac", type=float, default=0.25)
    parser.add_argument("--n-steps", type=int, default=30)
    parser.add_argument("--sampler", choices=("heun", "euler"), default="heun")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("samples/full_map_cib.fits"),
                        help="Output stem; per-band files are <stem>_<band>ghz.fits.")
    args = parser.parse_args(argv)

    from astropy.io import fits
    from astropy.wcs import WCS

    print(f"jax devices: {jax.devices()}")
    with open(args.ckpt, "rb") as f:
        ckpt = pickle.load(f)
    params = ckpt["params"]
    cfg = ckpt["config"]
    quantile_grid = np.asarray(ckpt["quantile_grid"], dtype=np.float64)
    z_grid = np.asarray(ckpt["z_grid"], dtype=np.float64)
    y0 = np.asarray(ckpt["y0"], dtype=np.float64)
    bands = [str(b) for b in ckpt.get("bands", list(range(cfg["out_channels"])))]
    pixel_arcmin = float(ckpt["pixel_size_arcmin"])
    patch_h, patch_w = cfg["patch_hw"]
    c = cfg["out_channels"]
    pix_deg = pixel_arcmin / 60.0
    print(f"  patch {patch_h}x{patch_w}, {c} bands {bands}, pixel {pixel_arcmin}'")

    model = UNet(channels=tuple(cfg["channels"]), t_dim=cfg["t_dim"],
                 bottleneck_blocks=cfg["bottleneck_blocks"], out_channels=c)

    ra_c = args.ra_center if args.ra_center is not None else 0.5 * (args.ra_min + args.ra_max)
    dec_c = args.dec_center if args.dec_center is not None else 0.5 * (args.dec_min + args.dec_max)

    corners_ra = np.array([args.ra_min, args.ra_min, args.ra_max, args.ra_max])
    corners_dec = np.array([args.dec_min, args.dec_max, args.dec_min, args.dec_max])
    probe = WCS(naxis=2)
    probe.wcs.crpix = [1.0, 1.0]
    probe.wcs.cdelt = [-pix_deg, pix_deg]
    probe.wcs.crval = [ra_c, dec_c]
    probe.wcs.ctype = ["RA---ZEA", "DEC--ZEA"]
    px, py = probe.wcs_world2pix(corners_ra, corners_dec, 0)
    margin = max(patch_h, patch_w) // 2
    n_pix_x = int(np.ceil(px.max() - px.min())) + 2 * margin
    n_pix_y = int(np.ceil(py.max() - py.min())) + 2 * margin

    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [-px.min() + margin + 1, -py.min() + margin + 1]
    wcs.wcs.cdelt = [-pix_deg, pix_deg]
    wcs.wcs.crval = [ra_c, dec_c]
    wcs.wcs.ctype = ["RA---ZEA", "DEC--ZEA"]
    print(f"  global ZEA tangent ({ra_c:.2f}, {dec_c:.2f}); grid {n_pix_x}x{n_pix_y} "
          f"(~{n_pix_x * pix_deg:.1f} x {n_pix_y * pix_deg:.1f} deg)")

    stride_x = max(1, int(round(patch_w * (1.0 - args.overlap))))
    stride_y = max(1, int(round(patch_h * (1.0 - args.overlap))))
    xs = list(range(0, n_pix_x - patch_w + 1, stride_x))
    if not xs or xs[-1] + patch_w < n_pix_x:
        xs.append(max(0, n_pix_x - patch_w))
    ys = list(range(0, n_pix_y - patch_h + 1, stride_y))
    if not ys or ys[-1] + patch_h < n_pix_y:
        ys.append(max(0, n_pix_y - patch_h))
    positions = [(x, y) for y in ys for x in xs]
    print(f"  {len(xs)}x{len(ys)} = {len(positions)} patches "
          f"(stride {stride_x}x{stride_y}, overlap {args.overlap:.2f})")

    accum = np.zeros((c, n_pix_y, n_pix_x), dtype=np.float64)
    weights = np.zeros((n_pix_y, n_pix_x), dtype=np.float64)
    win = _cosine_window_2d(patch_h, patch_w, edge_frac=args.edge_frac)

    seed = args.seed
    t0 = time.time()
    n_done = 0
    for start in range(0, len(positions), args.batch_size):
        chunk = positions[start : start + args.batch_size]
        if args.sampler == "euler":
            x_batch = sample_euler_one_step(
                model.apply, params, n_samples=len(chunk),
                spatial_shape=(patch_h, patch_w), n_channels=c, seed=seed,
            )
        else:
            x_batch = sample_heun(
                model.apply, params, n_samples=len(chunk),
                spatial_shape=(patch_h, patch_w), n_channels=c,
                n_steps=args.n_steps, seed=seed,
            )
        seed += 10_000
        y_batch = np.asarray(gaussianized_to_physical_multi(
            jnp.asarray(np.asarray(x_batch)), quantile_grid, z_grid, y0=y0
        ))                                                         # (B, H, W, C)
        for k, (px0, py0) in enumerate(chunk):
            for ch in range(c):
                accum[ch, py0 : py0 + patch_h, px0 : px0 + patch_w] += win * y_batch[k, ..., ch]
            weights[py0 : py0 + patch_h, px0 : px0 + patch_w] += win
        n_done += len(chunk)
        elapsed = (time.time() - t0) / 60.0
        print(f"    {n_done}/{len(positions)} done ({elapsed:.1f} min)", flush=True)

    wsafe = np.maximum(weights, 1e-30)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    header = wcs.to_header()
    for ch, band in enumerate(bands):
        final = (accum[ch] / wsafe).astype(np.float32)
        primary = fits.PrimaryHDU(data=final, header=header)
        primary.header["BUNIT"] = "cib_intensity"
        primary.header["BAND"] = f"{band}ghz"
        primary.header["EXTNAME"] = f"CIB_{band}"
        for key, val in (("RA_MIN", args.ra_min), ("RA_MAX", args.ra_max),
                         ("DEC_MIN", args.dec_min), ("DEC_MAX", args.dec_max),
                         ("OVERLAP", args.overlap), ("NSTEPS", args.n_steps),
                         ("SAMPLER", args.sampler), ("SEED", args.seed)):
            primary.header[key] = val
        primary.header["CKPT"] = str(args.ckpt)
        weight_hdu = fits.ImageHDU(data=weights.astype(np.float32),
                                   header=header, name="WEIGHT")
        out_band = args.out.with_name(f"{args.out.stem}_{band}ghz.fits")
        fits.HDUList([primary, weight_hdu]).writeto(out_band, overwrite=True)
        print(f"  saved {band} GHz -> {out_band}")
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
