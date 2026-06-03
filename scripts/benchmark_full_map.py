"""Benchmark full-footprint coherent synthesis: time N rounds in one process.

Builds the model + global ZEA grid once, runs the 4-color Repaint stitcher
N times with different seeds, and stacks the output maps into a single PNG
(N rows). JIT compile happens once during a warmup pass; the timed rounds
exclude it.
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
from agorasynth.flow_matching import sample_heun, sample_heun_conditional  # noqa: E402
from agorasynth.unet import UNet  # noqa: E402
from agorasynth.wph import (  # noqa: E402
    WPHOp,
    WPHPriorStats,
    compute_S_batch,
    to_real_features,
)


def _radial_power_spectrum(image, pixel_size_arcmin, n_bins=30):
    """Single-image radial power spectrum. ``image``: (H, W)."""
    arr = np.asarray(image, dtype=np.float64)
    H, W = arr.shape
    pix_rad = (pixel_size_arcmin / 60.0) * np.pi / 180.0
    F = np.fft.fft2(arr - arr.mean())
    P2d = (np.abs(F) ** 2) * pix_rad ** 2 / (H * W)
    fy = np.fft.fftfreq(H, d=pix_rad) * 2 * np.pi
    fx = np.fft.fftfreq(W, d=pix_rad) * 2 * np.pi
    fxg, fyg = np.meshgrid(fx, fy, indexing="xy")
    ell = np.sqrt(fxg ** 2 + fyg ** 2).ravel()
    P_flat = P2d.ravel()
    ell_max = ell.max()
    bins = np.logspace(
        np.log10(max(ell_max / 1000.0, 1.0)), np.log10(ell_max), n_bins + 1
    )
    digit = np.digitize(ell, bins) - 1
    centers = 0.5 * (bins[:-1] + bins[1:])
    P_binned = np.zeros(n_bins)
    for k in range(n_bins):
        mask = digit == k
        if mask.any():
            P_binned[k] = P_flat[mask].mean()
    return centers, P_binned


def _one_point_pdf(image, bins, floor=1e-12):
    """Histogram of log10(y) with externally-supplied bins."""
    log_y = np.log10(np.maximum(np.asarray(image), floor)).ravel()
    log_y = log_y[np.isfinite(log_y)]
    counts, edges = np.histogram(log_y, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, counts


def _project_healpix_to_zea_sht(
    hmap_data: np.ndarray,
    wcs,
    n_pix_x: int,
    n_pix_y: int,
    lmax: int | None = None,
    nest: bool = False,
    cache_path: Path | None = None,
) -> np.ndarray:
    """Reproject a HEALPix map onto a 2D ZEA grid via spherical-harmonic synthesis.

    Preserves power up to ``lmax`` rather than smoothing it (as bilinear
    interpolation does near the output Nyquist). Cost is dominated by
    ``healpy.map2alm`` -- one-time per HEALPix map; the result is cached
    to disk if ``cache_path`` is given.
    """
    if cache_path is not None and cache_path.exists():
        try:
            cached = np.load(cache_path)
            if cached.shape == (n_pix_y, n_pix_x):
                print(f"  loaded SHT-projected truth from {cache_path}")
                return cached.astype(np.float32)
            print(
                f"  cache shape mismatch ({cached.shape} vs ({n_pix_y}, {n_pix_x})); "
                "recomputing"
            )
        except Exception as exc:
            print(f"  failed to read cache {cache_path}: {exc}; recomputing")

    import healpy as hp

    try:
        from pixell import curvedsky, enmap
    except ImportError as exc:
        raise RuntimeError(
            "--truth-projection sht requires pixell. Install with "
            "`pip install pixell` or pass --truth-projection bilinear."
        ) from exc

    nside = hp.get_nside(hmap_data)
    if lmax is None:
        lmax = 3 * nside - 1

    if nest:
        print("  reordering NESTED -> RING for map2alm ...")
        hmap_data = hp.reorder(hmap_data, n2r=True)

    print(f"  map2alm (nside={nside}, lmax={lmax}) -- the slow step (~minutes) ...")
    t0 = time.time()
    alm = hp.map2alm(hmap_data, lmax=lmax, iter=0, use_pixel_weights=True)
    print(f"    map2alm done in {time.time() - t0:.1f} s")

    print(f"  alm2map onto ZEA {n_pix_y}x{n_pix_x} ...")
    t0 = time.time()
    omap = enmap.zeros((n_pix_y, n_pix_x), wcs)
    # pixell.curvedsky.alm2map derives lmax from the alm array itself.
    omap = curvedsky.alm2map(alm, omap)
    print(f"    alm2map done in {time.time() - t0:.1f} s")
    arr = np.asarray(omap, dtype=np.float32)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, arr)
        print(f"  cached SHT-projected truth -> {cache_path}")

    return arr


def _wph_features_from_map(image, wph_op, n_patches=64, patch_size=188, seed=0):
    """Random sub-patches from ``image`` -> mean WPH feature vector."""
    arr = np.asarray(image, dtype=np.float32)
    H, W = arr.shape
    rng = np.random.default_rng(seed)
    patches = np.empty((n_patches, patch_size, patch_size), dtype=np.float32)
    for k in range(n_patches):
        y0i = int(rng.integers(0, H - patch_size + 1))
        x0i = int(rng.integers(0, W - patch_size + 1))
        patches[k] = arr[y0i : y0i + patch_size, x0i : x0i + patch_size]
    s_complex = compute_S_batch(wph_op, jnp.asarray(patches))
    s_real = np.asarray(to_real_features(s_complex))
    return s_real.mean(axis=0)


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
    parser.add_argument("--n-steps", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-rounds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("samples/benchmark_fullmap.png"))
    parser.add_argument("--truth-map", type=Path,
                        default=Path("/global/cfs/cdirs/mp107c/yomori/agora/products/"
                                     "components/tsz/len/"
                                     "agora_ltszNG_bahamas80_bnd_unb_1.0e+12_1.0e+18_lensed.fits"),
                        help="HEALPix Compton-y FITS to project onto the global grid as "
                             "the truth row at the top of the plot. Skipped if missing.")
    parser.add_argument("--truth-nest", action="store_true",
                        help="Truth HEALPix map is in NESTED ordering (default: RING).")
    parser.add_argument("--no-truth", action="store_true",
                        help="Disable the truth row even if --truth-map exists.")
    parser.add_argument("--truth-projection", choices=("sht", "bilinear"),
                        default="sht",
                        help="HEALPix -> ZEA projection method. 'sht' (default) uses "
                             "spherical-harmonic synthesis (preserves power up to lmax). "
                             "'bilinear' is fast but smooths near the output Nyquist.")
    parser.add_argument("--truth-lmax", type=int, default=16000,
                        help="lmax for SHT projection. Default 16000 (well above the "
                             "1.6 arcmin Nyquist ell~6750, much faster than 3*Nside-1).")
    parser.add_argument("--truth-cache", type=Path, default=None,
                        help="Path to cache the projected truth (.npy). Default: "
                             "<truth-map>.sht_<shape>_<crpix>.npy next to the FITS.")
    parser.add_argument("--wph-prior", type=Path,
                        default=Path("runs/wph_prior.npz"),
                        help="WPH prior .npz; only used for the diagnostics plot's "
                             "WPH-features panel. Skipped if missing.")
    parser.add_argument("--n-wph-patches", type=int, default=64,
                        help="Random sub-patches per map for the WPH-features panel.")
    parser.add_argument("--diagnostics-out", type=Path, default=None,
                        help="Output path for the 3-panel diagnostics PNG. Default: "
                             "same stem as --out with '_diagnostics' suffix.")
    args = parser.parse_args(argv)

    if args.overlap > 0.5:
        raise ValueError("--overlap must be <= 0.5 for 4-color phasing")

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
    patch_h, patch_w = cfg["patch_hw"]
    n_channels = cfg["out_channels"]
    print(f"  patch={patch_h}x{patch_w}, pixel size {pixel_arcmin} arcmin")

    model = UNet(
        channels=tuple(cfg["channels"]),
        t_dim=cfg["t_dim"],
        bottleneck_blocks=cfg["bottleneck_blocks"],
        out_channels=n_channels,
    )
    apply_fn = model.apply

    # ------------------------------------------------------------------
    # Build global ZEA grid (once, shared across rounds)
    # ------------------------------------------------------------------
    from astropy.wcs import WCS

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
    px_c, py_c = probe_wcs.wcs_world2pix(corners_ra, corners_dec, 0)
    margin = max(patch_h, patch_w) // 2
    n_pix_x = int(np.ceil(px_c.max() - px_c.min())) + 2 * margin
    n_pix_y = int(np.ceil(py_c.max() - py_c.min())) + 2 * margin

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
    print(
        f"  global grid: {n_pix_x} x {n_pix_y} | patches {len(xs)}x{len(ys)} = {n_total} | "
        f"phases {[len(p) for p in phases]}"
    )

    # Re-anchor crpix so the tangent point sits in the middle of the global array.
    crpix_x = -px_c.min() + margin + 1
    crpix_y = -py_c.min() + margin + 1
    grid_wcs = WCS(naxis=2)
    grid_wcs.wcs.crpix = [crpix_x, crpix_y]
    grid_wcs.wcs.cdelt = [-pix_deg, pix_deg]
    grid_wcs.wcs.crval = [ra_c, dec_c]
    grid_wcs.wcs.ctype = ["RA---ZEA", "DEC--ZEA"]

    # Project the Agora HEALPix truth onto the same grid (optional).
    truth_map = None
    if not args.no_truth and args.truth_map is not None and args.truth_map.exists():
        print(f"projecting truth map {args.truth_map} ({args.truth_projection}) ...")
        import healpy as hp

        hmap = hp.read_map(str(args.truth_map))

        if args.truth_projection == "sht":
            cache_path = args.truth_cache
            if cache_path is None:
                stem = args.truth_map.stem
                tag = (
                    f"{stem}.sht_{n_pix_y}x{n_pix_x}"
                    f"_crpix{int(crpix_x)}_{int(crpix_y)}.npy"
                )
                cache_path = args.truth_map.with_name(tag)
            truth_map = _project_healpix_to_zea_sht(
                hmap, grid_wcs, n_pix_x, n_pix_y,
                lmax=args.truth_lmax,
                nest=args.truth_nest,
                cache_path=cache_path,
            )
        else:
            ii, jj = np.meshgrid(
                np.arange(n_pix_x), np.arange(n_pix_y), indexing="xy"
            )
            ra_grid, dec_grid = grid_wcs.wcs_pix2world(ii, jj, 0)
            theta = np.deg2rad(90.0 - dec_grid)
            phi = np.deg2rad(ra_grid)
            truth_map = hp.get_interp_val(
                hmap, theta, phi, nest=args.truth_nest
            ).astype(np.float32)

        print(
            f"  truth map: {truth_map.shape}, y range "
            f"[{truth_map.min():.3e}, {truth_map.max():.3e}]"
        )
    elif not args.no_truth and args.truth_map is not None:
        print(f"  truth map {args.truth_map} not found; skipping truth row")

    # ------------------------------------------------------------------
    # Synthesize one map closure
    # ------------------------------------------------------------------
    def synthesize_one(seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        global_x = np.zeros((n_pix_y, n_pix_x, n_channels), dtype=np.float32)
        filled = np.zeros((n_pix_y, n_pix_x, n_channels), dtype=bool)
        for phase_idx, phase_positions in enumerate(phases):
            if not phase_positions:
                continue
            for bstart in range(0, len(phase_positions), args.batch_size):
                chunk = phase_positions[bstart : bstart + args.batch_size]
                B = len(chunk)
                init_noise = rng.standard_normal(
                    (B, patch_h, patch_w, n_channels)
                ).astype(np.float32)
                init_noise_j = jnp.asarray(init_noise)
                if phase_idx == 0:
                    x_batch = sample_heun(
                        apply_fn, params, n_steps=args.n_steps,
                        init_noise=init_noise_j,
                    )
                else:
                    known_mask = np.zeros(
                        (B, patch_h, patch_w, n_channels), dtype=bool
                    )
                    known_value = np.zeros(
                        (B, patch_h, patch_w, n_channels), dtype=np.float32
                    )
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
                x_batch.block_until_ready()
                x_batch_np = np.asarray(x_batch)
                for k, (px, py) in enumerate(chunk):
                    local_filled = filled[py : py + patch_h, px : px + patch_w]
                    global_x[py : py + patch_h, px : px + patch_w] = np.where(
                        local_filled,
                        global_x[py : py + patch_h, px : px + patch_w],
                        x_batch_np[k],
                    )
                    filled[py : py + patch_h, px : px + patch_w] = True
        return np.asarray(gaussianized_to_physical(
            jnp.asarray(global_x[..., 0]), quantile_grid, z_grid, y0=y0
        ))

    # ------------------------------------------------------------------
    # Warmup + timed rounds
    # ------------------------------------------------------------------
    print("warmup pass (triggers JIT compile for both shape variants) ...")
    t0 = time.time()
    _warm = synthesize_one(args.seed)
    t_warmup = time.time() - t0
    print(f"  warmup: {t_warmup:.2f} s")

    print(f"running {args.n_rounds} timed rounds ...")
    maps = np.zeros((args.n_rounds, n_pix_y, n_pix_x), dtype=np.float32)
    times = []
    t_total_0 = time.time()
    for r in range(args.n_rounds):
        t0 = time.time()
        m = synthesize_one(args.seed + 1 + r)
        dt = time.time() - t0
        times.append(dt)
        maps[r] = m.astype(np.float32)
        print(f"  round {r + 1:>2d}/{args.n_rounds}: {dt:.2f} s")
    t_total = time.time() - t_total_0

    times_arr = np.asarray(times)
    print()
    print("timing summary")
    print(f"  warmup (incl. JIT compile): {t_warmup:.2f} s")
    print(f"  total of {args.n_rounds} timed rounds: {t_total:.2f} s")
    print(
        f"  per-round: mean {times_arr.mean():.2f} s, "
        f"min {times_arr.min():.2f} s, max {times_arr.max():.2f} s"
    )
    deg2 = (n_pix_x * pix_deg) * (n_pix_y * pix_deg)
    print(
        f"  approx footprint area (pixel-grid): {deg2:.0f} sq deg; "
        f"throughput: {deg2 / times_arr.mean():.0f} sq deg / s"
    )

    # ------------------------------------------------------------------
    # Plot N rows stacked
    # ------------------------------------------------------------------
    print(f"plotting {args.n_rounds} stacked maps ...")
    import matplotlib.pyplot as plt

    panels = list(maps)
    if truth_map is not None:
        panels = [truth_map] + panels
    panels_log = [np.log10(np.maximum(p, 1e-12)) for p in panels]
    vmin = float(np.percentile(np.concatenate(
        [p.ravel() for p in panels_log]
    ), 1))
    vmax = float(np.percentile(np.concatenate(
        [p.ravel() for p in panels_log]
    ), 99))
    aspect_per_row = n_pix_y / n_pix_x
    fig_w = 14.0
    nrows = len(panels)
    fig_h = max(2.0, fig_w * aspect_per_row) * nrows + 1.0
    fig, axes = plt.subplots(
        nrows, 1,
        figsize=(fig_w, fig_h),
        squeeze=False,
    )
    im = None
    truth_offset = 1 if truth_map is not None else 0
    for r in range(nrows):
        ax = axes[r, 0]
        im = ax.imshow(
            panels_log[r], cmap="inferno", origin="lower",
            vmin=vmin, vmax=vmax, aspect="equal",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        if r == 0 and truth_map is not None:
            ax.set_ylabel("truth (Agora)", fontsize=10, color="C0")
        else:
            ax.set_ylabel(f"round {r + 1 - truth_offset}", fontsize=10)
    fig.suptitle(
        f"Coherent SPT-1500 synthesis: {args.n_rounds} rounds, "
        f"per-round mean {times_arr.mean():.1f} s "
        f"(warmup {t_warmup:.1f} s)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0.0, 0.0, 0.92, 0.97))
    cax = fig.add_axes((0.94, 0.05, 0.012, 0.9))
    fig.colorbar(im, cax=cax, label=r"$\log_{10}\,y$")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {args.out}")

    # ------------------------------------------------------------------
    # Summary-statistics diagnostics: power spectrum, 1-pt PDF, WPH features
    # ------------------------------------------------------------------
    diag_out = args.diagnostics_out
    if diag_out is None:
        diag_out = args.out.with_name(f"{args.out.stem}_diagnostics{args.out.suffix}")

    print("computing summary statistics ...")
    map_list = list(maps)
    label_list = [f"round {r + 1}" for r in range(args.n_rounds)]
    has_truth = truth_map is not None
    if has_truth:
        map_list = [truth_map] + map_list
        label_list = ["truth"] + label_list

    # Power spectrum (same ell bins automatically from same shape).
    P_per_map = []
    ells_ref = None
    for m in map_list:
        ell, P = _radial_power_spectrum(m, pixel_arcmin)
        if ells_ref is None:
            ells_ref = ell
        P_per_map.append(P)
    P_per_map = np.asarray(P_per_map)

    # 1-pt PDF with common bins from all maps.
    log_concat = np.concatenate([
        np.log10(np.maximum(m, 1e-12)).ravel() for m in map_list
    ])
    log_concat = log_concat[np.isfinite(log_concat)]
    pdf_bins = np.linspace(
        float(np.percentile(log_concat, 0.1)),
        float(np.percentile(log_concat, 99.9)),
        80 + 1,
    )
    pdf_per_map = []
    pdf_centers = None
    for m in map_list:
        c, p = _one_point_pdf(m, pdf_bins)
        if pdf_centers is None:
            pdf_centers = c
        pdf_per_map.append(p)
    pdf_per_map = np.asarray(pdf_per_map)

    # WPH features (optional).
    wph_per_map = None
    if args.wph_prior is not None and args.wph_prior.exists():
        print(f"  computing WPH features (prior {args.wph_prior}) ...")
        prior = WPHPriorStats.load(args.wph_prior)
        wph_op = WPHOp.build(prior.config)
        wph_per_map = []
        for k, m in enumerate(map_list):
            f = _wph_features_from_map(
                m, wph_op,
                n_patches=args.n_wph_patches,
                patch_size=prior.config.M,
                seed=args.seed + k,
            )
            wph_per_map.append(f)
        wph_per_map = np.asarray(wph_per_map)
    else:
        print(f"  WPH prior {args.wph_prior} not found; skipping WPH panel")

    print(f"plotting diagnostics ...")
    n_panels = 3 if wph_per_map is not None else 2
    # Two rows: top = stat itself, bottom = (real - truth) / sigma_real residual
    fig, axes = plt.subplots(
        2, n_panels,
        figsize=(5.5 * n_panels, 7.5),
        gridspec_kw={"height_ratios": [3, 1]},
        squeeze=False,
    )

    def _bands(y):
        return {
            "p2p5": np.percentile(y, 2.5, axis=0),
            "p16": np.percentile(y, 16.0, axis=0),
            "median": np.percentile(y, 50.0, axis=0),
            "p84": np.percentile(y, 84.0, axis=0),
            "p97p5": np.percentile(y, 97.5, axis=0),
            "std": y.std(axis=0),
        }

    def _plot_main(ax, x, y_all, has_truth):
        if has_truth:
            truth = y_all[0]
            real = y_all[1:]
        else:
            truth = None
            real = y_all
        b = _bands(real)
        ax.fill_between(x, b["p2p5"], b["p97p5"], color="C1", alpha=0.2,
                        label="realizations 95% (2.5/97.5)")
        ax.fill_between(x, b["p16"], b["p84"], color="C1", alpha=0.4,
                        label="realizations 68% (16/84)")
        ax.plot(x, b["median"], color="C1", lw=1.5, label="realizations median")
        for r in real:
            ax.plot(x, r, color="C1", alpha=0.12, lw=0.5)
        if truth is not None:
            ax.plot(x, truth, color="C0", lw=2.2, label="truth (Agora)")
        ax.legend(loc="best", fontsize=8)

    def _plot_residual(ax, x, y_all, has_truth):
        if not has_truth:
            ax.text(0.5, 0.5, "(no truth)", ha="center", va="center",
                    transform=ax.transAxes, color="grey")
            ax.set_xticks([])
            ax.set_yticks([])
            return
        truth = y_all[0]
        real = y_all[1:]
        b = _bands(real)
        sigma = np.where(b["std"] > 0, b["std"], np.nan)
        z = (truth - b["median"]) / sigma
        ax.axhline(0, color="grey", lw=0.6)
        ax.axhspan(-1, 1, color="C1", alpha=0.2)
        ax.axhspan(-2, 2, color="C1", alpha=0.1)
        ax.plot(x, z, color="C0", lw=1.4)
        ax.set_ylabel(
            r"(truth - median) / $\sigma_{\rm real}$", fontsize=9
        )
        # Pin to a sane range; clip for display only.
        ax.set_ylim(-6, 6)

    # ---- Panel 1: power spectrum
    mask = (P_per_map.min(axis=0) > 0) & (ells_ref > 0)
    x1 = ells_ref[mask]
    y1 = P_per_map[:, mask]
    ax = axes[0, 0]
    _plot_main(ax, x1, y1, has_truth)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\ell$ [rad$^{-1}$]")
    ax.set_ylabel(r"$P(\ell)$")
    ax.set_title("Radial power spectrum")
    ax.grid(True, which="both", alpha=0.3)
    ax2 = axes[1, 0]
    _plot_residual(ax2, x1, y1, has_truth)
    ax2.set_xscale("log")
    ax2.set_xlabel(r"$\ell$ [rad$^{-1}$]")

    # ---- Panel 2: 1-pt PDF
    ax = axes[0, 1]
    _plot_main(ax, pdf_centers, pdf_per_map, has_truth)
    ax.set_yscale("log")
    ax.set_xlabel(r"$\log_{10}\,y$")
    ax.set_ylabel("density")
    ax.set_title("1-point PDF")
    ax.grid(True, which="both", alpha=0.3)
    ax2 = axes[1, 1]
    _plot_residual(ax2, pdf_centers, pdf_per_map, has_truth)
    ax2.set_xlabel(r"$\log_{10}\,y$")

    # ---- Panel 3: WPH features
    if wph_per_map is not None:
        ax = axes[0, 2]
        # Sort features by descending |truth| (or |median| if no truth) so
        # large-amplitude features sit on the left and the symlog y axis is
        # readable across orders of magnitude.
        ref = np.abs(wph_per_map[0]) if has_truth else np.median(
            np.abs(wph_per_map), axis=0
        )
        order = np.argsort(-ref)
        wph_sorted = wph_per_map[:, order]
        idx = np.arange(wph_sorted.shape[1])
        _plot_main(ax, idx, wph_sorted, has_truth)
        # symlog with linthresh ~ small fraction of the typical magnitude
        # gives readable spacing across positive/negative features at all scales.
        med_mag = float(np.median(np.abs(wph_sorted))) if wph_sorted.size else 1.0
        linthresh = max(med_mag * 0.01, 1e-30)
        ax.set_yscale("symlog", linthresh=linthresh)
        ax.set_xlabel("WPH feature (sorted by |truth| descending)")
        ax.set_ylabel("feature value (mean over sub-patches)")
        ax.set_title(
            f"WPH features (avg over {args.n_wph_patches} sub-patches/map)"
        )
        ax.grid(True, which="both", alpha=0.3)
        ax2 = axes[1, 2]
        _plot_residual(ax2, idx, wph_sorted, has_truth)
        ax2.set_xlabel("WPH feature (sorted by |truth| descending)")

    fig.suptitle(
        f"Summary statistics: truth (Agora) vs. {args.n_rounds} synthesized realizations\n"
        "top: stat with realization 68% / 95% bands. "
        r"bottom: (truth - median) / $\sigma_{\rm real}$, shaded $\pm 1, 2\sigma$.",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(diag_out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {diag_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
