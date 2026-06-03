"""Coherent full-footprint CIB maps (95/150/220 GHz) via Repaint-style stitching.

Multi-channel analog of ``synthesize_full_map_coherent.py``. Unlike
``synthesize_full_map_cib.py`` (independent tiles blended with a cosine window),
this enforces consistency across patch boundaries: when generating a patch that
overlaps an already-generated region, the overlap pixels are constrained to
track the linear-interpolation path during ODE integration, and the model
infills the rest consistently with that boundary -- so adjacent patches agree at
the seam by construction, not by blending.

Because one joint sample yields all C bands per patch, the constraint couples
*all three bands at once* at every seam: the stitch preserves both the spatial
coherence AND the inter-band CIB correlation across the whole footprint.

Patch scheduling
----------------
Patches on a regular grid with stride ``s = (1 - overlap) * patch`` are
4-colored ``(i % 2, j % 2)`` -> phase 0..3. For overlap <= 0.5 no two patches in
the same phase touch, so each phase is one batched ODE call. Phase 0 is sampled
unconditionally; phases 1-3 are conditioned on whatever is already generated.

The global state is held in *gaussianized* per-band log-intensity space (the
model's domain); we de-gaussianize per band once at the end and write one
physical-intensity FITS per band (shared ZEA WCS).

Caveats
-------
- The model only knows statistics up to its patch size; coherent stitching makes
  the field seamless and propagates correlations patch-to-patch, but cannot add
  true power on scales >> a patch.
- Overlap > 0.5 needs a finer coloring (not implemented); default 0.25 is fine.
- Edge patches at large angles from the global tangent are mildly distorted vs.
  training (~10-15% pixel-scale change at 50 deg from tangent for SPT-1500).
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
from agorasynth.flow_matching import (  # noqa: E402
    sample_heun,
    sample_heun_conditional,
)
from agorasynth.unet import UNet  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--ra-min", type=float, default=-50.0)
    parser.add_argument("--ra-max", type=float, default=50.0)
    parser.add_argument("--dec-min", type=float, default=-70.0)
    parser.add_argument("--dec-max", type=float, default=-40.0)
    parser.add_argument("--ra-center", type=float, default=None)
    parser.add_argument("--dec-center", type=float, default=None)
    parser.add_argument("--overlap", type=float, default=0.25,
                        help="Patch overlap fraction (must be <= 0.5).")
    parser.add_argument("--n-steps", type=int, default=30,
                        help="Heun ODE steps per patch.")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Max patches per JIT'd batch within a phase.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path,
                        default=Path("samples/full_map_cib_coherent.fits"),
                        help="Output stem; per-band files are <stem>_<band>ghz.fits.")
    args = parser.parse_args(argv)

    if args.overlap > 0.5:
        raise ValueError(
            f"--overlap {args.overlap} > 0.5 not supported by 4-color phasing; "
            "lower it or implement a finer schedule."
        )

    from astropy.io import fits
    from astropy.wcs import WCS

    print(f"jax devices: {jax.devices()}")
    print(f"loading {args.ckpt} ...")
    with open(args.ckpt, "rb") as f:
        ckpt = pickle.load(f)
    params = ckpt["params"]
    cfg = ckpt["config"]
    quantile_grid = np.asarray(ckpt["quantile_grid"], dtype=np.float64)   # (C, nq)
    z_grid = np.asarray(ckpt["z_grid"], dtype=np.float64)
    y0 = np.atleast_1d(np.asarray(ckpt["y0"], dtype=np.float64))          # (C,)
    pixel_arcmin = float(ckpt["pixel_size_arcmin"])
    patch_h, patch_w = cfg["patch_hw"]
    n_channels = int(cfg["out_channels"])
    bands = [str(b) for b in ckpt.get("bands", list(range(n_channels)))]
    if quantile_grid.shape[0] != n_channels or y0.shape[0] != n_channels:
        raise ValueError(
            f"checkpoint channel mismatch: out_channels={n_channels}, "
            f"quantile_grid={quantile_grid.shape}, y0={y0.shape}"
        )
    print(f"  patch {patch_h}x{patch_w}, {n_channels} bands {bands}, "
          f"pixel {pixel_arcmin}'")

    model = UNet(
        channels=tuple(cfg["channels"]),
        t_dim=cfg["t_dim"],
        bottleneck_blocks=cfg["bottleneck_blocks"],
        out_channels=n_channels,
    )
    apply_fn = model.apply

    # ------------------------------------------------------------------
    # Build global ZEA grid
    # ------------------------------------------------------------------
    ra_c = args.ra_center if args.ra_center is not None else 0.5 * (args.ra_min + args.ra_max)
    dec_c = (
        args.dec_center if args.dec_center is not None
        else 0.5 * (args.dec_min + args.dec_max)
    )
    pix_deg = pixel_arcmin / 60.0

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
    crpix_x = -px_corners.min() + margin + 1
    crpix_y = -py_corners.min() + margin + 1
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [crpix_x, crpix_y]
    wcs.wcs.cdelt = [-pix_deg, pix_deg]
    wcs.wcs.crval = [ra_c, dec_c]
    wcs.wcs.ctype = ["RA---ZEA", "DEC--ZEA"]
    print(f"  global ZEA tangent ({ra_c:.2f}, {dec_c:.2f}); grid "
          f"{n_pix_x} x {n_pix_y} (~{n_pix_x * pix_deg:.1f} x "
          f"{n_pix_y * pix_deg:.1f} deg)")

    # ------------------------------------------------------------------
    # Patch placement and 4-color phase scheduling
    # ------------------------------------------------------------------
    stride_x = max(1, int(round(patch_w * (1.0 - args.overlap))))
    stride_y = max(1, int(round(patch_h * (1.0 - args.overlap))))
    xs = list(range(0, n_pix_x - patch_w + 1, stride_x))
    if not xs or xs[-1] + patch_w < n_pix_x:
        xs.append(max(0, n_pix_x - patch_w))
    ys = list(range(0, n_pix_y - patch_h + 1, stride_y))
    if not ys or ys[-1] + patch_h < n_pix_y:
        ys.append(max(0, n_pix_y - patch_h))

    phases: list[list[tuple[int, int]]] = [[] for _ in range(4)]
    for j, py in enumerate(ys):
        for i, px in enumerate(xs):
            color = (i % 2) + 2 * (j % 2)
            phases[color].append((px, py))
    n_total = sum(len(p) for p in phases)
    print(f"  patches: {len(xs)} x {len(ys)} = {n_total} total | "
          f"phases: {[len(p) for p in phases]}")

    global_x = np.zeros((n_pix_y, n_pix_x, n_channels), dtype=np.float32)
    filled = np.zeros((n_pix_y, n_pix_x, n_channels), dtype=bool)

    # ------------------------------------------------------------------
    # Process phases sequentially; within a phase, batch by --batch-size.
    # ------------------------------------------------------------------
    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    for phase_idx, phase_positions in enumerate(phases):
        if not phase_positions:
            continue
        for batch_start in range(0, len(phase_positions), args.batch_size):
            chunk = phase_positions[batch_start : batch_start + args.batch_size]
            B = len(chunk)
            init_noise = rng.standard_normal(
                (B, patch_h, patch_w, n_channels)
            ).astype(np.float32)
            init_noise_j = jnp.asarray(init_noise)

            if phase_idx == 0:
                x_batch = sample_heun(
                    apply_fn, params, n_steps=args.n_steps, init_noise=init_noise_j,
                )
            else:
                known_mask = np.zeros((B, patch_h, patch_w, n_channels), dtype=bool)
                known_value = np.zeros((B, patch_h, patch_w, n_channels), dtype=np.float32)
                for k, (px, py) in enumerate(chunk):
                    known_mask[k] = filled[py : py + patch_h, px : px + patch_w]
                    known_value[k] = global_x[py : py + patch_h, px : px + patch_w]
                x_batch = sample_heun_conditional(
                    apply_fn, params,
                    init_noise=init_noise_j,
                    known_mask=jnp.asarray(known_mask),
                    known_value=jnp.asarray(known_value),
                    n_steps=args.n_steps,
                )

            x_batch_np = np.asarray(x_batch)
            for k, (px, py) in enumerate(chunk):
                local_filled = filled[py : py + patch_h, px : px + patch_w]
                # Keep already-filled pixels exactly (the constraint reproduces
                # them anyway; enforce to kill any tiny ODE drift).
                global_x[py : py + patch_h, px : px + patch_w] = np.where(
                    local_filled,
                    global_x[py : py + patch_h, px : px + patch_w],
                    x_batch_np[k],
                )
                filled[py : py + patch_h, px : px + patch_w] = True

            elapsed = (time.time() - t0) / 60.0
            n_done = sum(len(p) for p in phases[:phase_idx]) + batch_start + B
            print(f"    phase {phase_idx}/3  {n_done}/{n_total} patches  "
                  f"({elapsed:.1f} min)", flush=True)

    if not filled.all():
        n_uncov = int((~filled).sum())
        print(f"  WARNING: {n_uncov} pixel-channels left uncovered (zeros).")

    # ------------------------------------------------------------------
    # De-gaussianize per band and save one FITS per band
    # ------------------------------------------------------------------
    print("de-gaussianizing to physical intensity ...")
    global_y = np.asarray(gaussianized_to_physical_multi(
        jnp.asarray(global_x), quantile_grid, z_grid, y0=y0
    ))                                                          # (ny, nx, C)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    header = wcs.to_header()
    for ch, band in enumerate(bands):
        final = global_y[..., ch].astype(np.float32)
        print(f"  {band} GHz: range [{final.min():.3e}, {final.max():.3e}], "
              f"median {float(np.median(final)):.3e}")
        primary = fits.PrimaryHDU(data=final, header=header)
        primary.header["BUNIT"] = "cib_intensity"
        primary.header["BAND"] = f"{band}ghz"
        primary.header["EXTNAME"] = f"CIB_{band}"
        primary.header["METHOD"] = "coherent_repaint"
        for key, val in (("RA_MIN", args.ra_min), ("RA_MAX", args.ra_max),
                         ("DEC_MIN", args.dec_min), ("DEC_MAX", args.dec_max),
                         ("OVERLAP", args.overlap), ("NSTEPS", args.n_steps),
                         ("SEED", args.seed)):
            primary.header[key] = val
        primary.header["CKPT"] = str(args.ckpt)
        out_band = args.out.with_name(f"{args.out.stem}_{band}ghz.fits")
        fits.HDUList([primary]).writeto(out_band, overwrite=True)
        print(f"  saved {band} GHz -> {out_band}")
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
