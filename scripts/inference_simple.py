"""Simple NumPyro toy: noisy SPT-1500 footprint, recover y under a Gaussian prior.

Data flow:
  1. Project the Agora HEALPix y-map onto an SPT-1500 ZEA grid via SHT
     (cached after first call), then gaussianize per-pixel using the
     ``quantile_grid`` from data/train.npz so the marginal is ~N(0, 1).
  2. Generate mock observation: d = y + n,  n ~ N(0, sigma_noise^2 I).
  3. NumPyro model: y ~ N(0, 1) i.i.d.; d | y ~ N(y, sigma_noise^2).
  4. NUTS samples the posterior. Pixels are independent so the posterior
     is conjugate Gaussian -- analytic posterior is computed alongside
     for verification.

This is the linear-Gaussian inverse-problem template. The only line that
changes when swapping in the AgoraSynth FM prior is the ``y ~ Normal(0,1)``
sample site -- the data + likelihood half stays put.

Memory note: a 2840 x 1059 grid is ~3M pixels. ``--num-samples 100``
keeps GPU memory ~5 GB; bump only if you need tighter posterior stats.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpyro  # noqa: E402
import numpyro.distributions as dist  # noqa: E402
from numpyro.infer import MCMC, NUTS  # noqa: E402


def _build_zea_wcs(ra_min, ra_max, dec_min, dec_max, pix_arcmin, patch_pad):
    """Return (wcs, n_pix_x, n_pix_y) for an SPT-1500-like ZEA tangent grid."""
    from astropy.wcs import WCS

    ra_c = 0.5 * (ra_min + ra_max)
    dec_c = 0.5 * (dec_min + dec_max)
    pix_deg = pix_arcmin / 60.0
    corners_ra = np.array([ra_min, ra_min, ra_max, ra_max])
    corners_dec = np.array([dec_min, dec_max, dec_min, dec_max])
    probe = WCS(naxis=2)
    probe.wcs.crpix = [1.0, 1.0]
    probe.wcs.cdelt = [-pix_deg, pix_deg]
    probe.wcs.crval = [ra_c, dec_c]
    probe.wcs.ctype = ["RA---ZEA", "DEC--ZEA"]
    px, py = probe.wcs_world2pix(corners_ra, corners_dec, 0)
    margin = patch_pad
    n_pix_x = int(np.ceil(px.max() - px.min())) + 2 * margin
    n_pix_y = int(np.ceil(py.max() - py.min())) + 2 * margin
    crpix_x = -px.min() + margin + 1
    crpix_y = -py.min() + margin + 1
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [crpix_x, crpix_y]
    wcs.wcs.cdelt = [-pix_deg, pix_deg]
    wcs.wcs.crval = [ra_c, dec_c]
    wcs.wcs.ctype = ["RA---ZEA", "DEC--ZEA"]
    return wcs, n_pix_x, n_pix_y, crpix_x, crpix_y


def _project_healpix_sht(hmap_data, wcs, n_pix_x, n_pix_y, lmax, nest, cache_path):
    """SHT-based reprojection (or load from cache). Returns (n_pix_y, n_pix_x) float32."""
    if cache_path.exists():
        cached = np.load(cache_path)
        if cached.shape == (n_pix_y, n_pix_x):
            print(f"  loaded cached SHT-projected truth from {cache_path}")
            return cached.astype(np.float32)
        print(
            f"  cache shape mismatch ({cached.shape} vs ({n_pix_y}, {n_pix_x})); recomputing"
        )

    import healpy as hp
    from pixell import curvedsky, enmap

    if nest:
        print("  reordering NESTED -> RING for map2alm ...")
        hmap_data = hp.reorder(hmap_data, n2r=True)

    nside = hp.get_nside(hmap_data)
    print(f"  map2alm (nside={nside}, lmax={lmax}) -- ~minutes ...")
    t0 = time.time()
    alm = hp.map2alm(hmap_data, lmax=lmax, iter=0, use_pixel_weights=True)
    print(f"    map2alm done in {time.time() - t0:.1f} s")

    print(f"  alm2map onto ZEA {n_pix_y}x{n_pix_x} ...")
    t0 = time.time()
    omap = enmap.zeros((n_pix_y, n_pix_x), wcs)
    omap = curvedsky.alm2map(alm, omap)
    print(f"    alm2map done in {time.time() - t0:.1f} s")
    arr = np.asarray(omap, dtype=np.float32)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, arr)
    print(f"  cached -> {cache_path}")
    return arr


def model(data: jnp.ndarray, sigma_noise: float):
    """Per-pixel N(0, 1) prior + Gaussian noise likelihood."""
    y = numpyro.sample(
        "y", dist.Normal(0.0, 1.0).expand(data.shape).to_event(2)
    )
    numpyro.sample(
        "d", dist.Normal(y, sigma_noise).to_event(2), obs=data
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/train.npz"),
                        help="Dataset .npz with quantile_grid + z_grid (for gaussianize).")
    parser.add_argument("--healpix-truth", type=Path,
                        default=Path("/global/cfs/cdirs/mp107c/yomori/agora/products/"
                                     "components/tsz/len/"
                                     "agora_ltszNG_bahamas80_bnd_unb_1.0e+12_1.0e+18_lensed.fits"),
                        help="HEALPix Compton-y FITS to use as truth.")
    parser.add_argument("--truth-nest", action="store_true")
    parser.add_argument("--truth-lmax", type=int, default=16000)
    parser.add_argument("--ra-min", type=float, default=-50.0)
    parser.add_argument("--ra-max", type=float, default=50.0)
    parser.add_argument("--dec-min", type=float, default=-70.0)
    parser.add_argument("--dec-max", type=float, default=-40.0)
    parser.add_argument("--pixel-arcmin", type=float, default=1.6)
    parser.add_argument("--patch-pad", type=int, default=94,
                        help="Padding (px) around the footprint corners.")
    parser.add_argument("--sigma-noise", type=float, default=0.5,
                        help="Std of additive noise (in gaussianized log-y units).")
    parser.add_argument("--num-warmup", type=int, default=200)
    parser.add_argument("--num-samples", type=int, default=100,
                        help="NUTS posterior samples to draw. Each ~12 MB at full SPT-1500.")
    parser.add_argument("--num-chains", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("samples/inference_simple.png"))
    parser.add_argument("--samples-dir", type=Path, default=None,
                        help="Directory to save individual posterior-sample PNGs into. "
                             "Default: <out-stem>_samples/ next to --out.")
    parser.add_argument("--max-samples-saved", type=int, default=None,
                        help="Cap on how many individual samples to write as PNG. "
                             "Default: all of them.")
    parser.add_argument("--vmin", type=float, default=-4.0,
                        help="Color-scale lower bound for value panels (truth/data/samples).")
    parser.add_argument("--vmax", type=float, default=4.0,
                        help="Color-scale upper bound for value panels (truth/data/samples).")
    args = parser.parse_args(argv)

    print(f"jax devices: {jax.devices()}")
    numpyro.set_host_device_count(args.num_chains)

    # Load gaussianize transform from the training dataset.
    print(f"loading gaussianize transform from {args.data} ...")
    npz = np.load(args.data)
    quantile_grid = np.asarray(npz["quantile_grid"], dtype=np.float64)
    z_grid = np.asarray(npz["z_grid"], dtype=np.float64)
    y0 = float(npz["y0"])
    print(
        f"  quantile_grid log-y in [{quantile_grid[0]:.2f}, "
        f"{quantile_grid[-1]:.2f}]; y0={y0:.1e}"
    )

    # Build SPT-1500 ZEA grid.
    wcs, n_pix_x, n_pix_y, crpix_x, crpix_y = _build_zea_wcs(
        args.ra_min, args.ra_max, args.dec_min, args.dec_max,
        args.pixel_arcmin, args.patch_pad,
    )
    print(f"  ZEA grid: {n_pix_y} x {n_pix_x} pixels")

    # SHT-project Agora truth.
    if not args.healpix_truth.exists():
        raise FileNotFoundError(f"truth HEALPix not found: {args.healpix_truth}")
    print(f"projecting truth {args.healpix_truth} onto ZEA grid ...")
    import healpy as hp

    hmap = hp.read_map(str(args.healpix_truth))
    cache_tag = (
        f"{args.healpix_truth.stem}.sht_{n_pix_y}x{n_pix_x}"
        f"_crpix{int(crpix_x)}_{int(crpix_y)}.npy"
    )
    cache_path = args.healpix_truth.with_name(cache_tag)
    truth_y = _project_healpix_sht(
        hmap, wcs, n_pix_x, n_pix_y,
        lmax=args.truth_lmax, nest=args.truth_nest, cache_path=cache_path,
    )
    print(
        f"  truth y: shape={truth_y.shape}, range "
        f"[{truth_y.min():.3e}, {truth_y.max():.3e}]"
    )

    # Gaussianize: y -> x via log + interp on quantile_grid.
    print("gaussianizing truth ...")
    log_y = np.log(np.maximum(truth_y, 1e-30) + y0)
    truth_x = (
        np.interp(log_y.ravel(), quantile_grid, z_grid)
        .reshape(truth_y.shape)
        .astype(np.float32)
    )
    print(
        f"  truth x: range [{truth_x.min():.3f}, {truth_x.max():.3f}], "
        f"mean {truth_x.mean():.3f}, std {truth_x.std():.3f}"
    )

    # Mock observation: d = x + n.
    rng = np.random.default_rng(args.seed)
    noise = rng.standard_normal(truth_x.shape) * args.sigma_noise
    obs = truth_x + noise
    print(
        f"  mock data: sigma_noise={args.sigma_noise}, range "
        f"[{obs.min():.3f}, {obs.max():.3f}]"
    )

    # NUTS.
    n_dim = truth_x.size
    print(f"running NUTS ({n_dim:,} latent dims; {args.num_samples} samples) ...")
    t0 = time.time()
    mcmc = MCMC(
        NUTS(model),
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_chains=args.num_chains,
        progress_bar=True,
    )
    mcmc.run(jax.random.PRNGKey(args.seed), jnp.asarray(obs), args.sigma_noise)
    print(f"NUTS done in {time.time() - t0:.1f} s")

    samples = mcmc.get_samples()
    y_post = np.asarray(samples["y"])               # (n_samples, n_pix_y, n_pix_x)
    post_mean = y_post.mean(axis=0)
    post_std = y_post.std(axis=0)
    print(
        f"  posterior mean range [{post_mean.min():.3f}, {post_mean.max():.3f}], "
        f"std mean {post_std.mean():.3f}"
    )

    # Analytic conjugate posterior under prior N(0, 1):
    #   var_post = 1 / (1 + 1/sigma^2)
    #   mean_post = var_post * d / sigma^2
    var_a = 1.0 / (1.0 + 1.0 / args.sigma_noise ** 2)
    a_mean = var_a * obs / args.sigma_noise ** 2
    a_std = float(np.sqrt(var_a))
    print(
        f"  analytic posterior std (uniform per pixel) = {a_std:.3f}; "
        f"NUTS mean std = {post_std.mean():.3f}"
    )
    rel_err = float(np.sqrt(((post_mean - a_mean) ** 2).mean()) / a_std)
    print(f"  NUTS-vs-analytic mean RMS error = {rel_err:.3f} sigma")

    # Plot.
    print("plotting ...")
    import matplotlib.pyplot as plt

    panels = [
        ("truth x", truth_x, "viridis"),
        (f"data = x + N(0, {args.sigma_noise})", obs, "viridis"),
        ("NUTS posterior mean", post_mean, "viridis"),
        ("analytic posterior mean", a_mean, "viridis"),
        ("NUTS posterior std", post_std, "magma"),
    ]
    vmin = args.vmin
    vmax = args.vmax

    fig, axes = plt.subplots(
        len(panels), 1,
        figsize=(14.0, 3.0 * len(panels) + 1),
        squeeze=False,
    )
    for ax, (title, img, cmap) in zip(axes[:, 0], panels, strict=False):
        kwargs = dict(cmap=cmap, origin="lower", aspect="equal")
        if "std" not in title:
            kwargs["vmin"] = vmin
            kwargs["vmax"] = vmax
        im = ax.imshow(img, **kwargs)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.012, pad=0.005)

    fig.suptitle(
        f"SPT-1500 simple inference, {n_pix_y}x{n_pix_x}, "
        f"sigma_noise={args.sigma_noise}; "
        f"NUTS-vs-analytic mean RMS = {rel_err:.3f}sigma",
        fontsize=11,
    )
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {args.out}")

    # ------------------------------------------------------------------
    # Save individual posterior samples as separate PNGs
    # ------------------------------------------------------------------
    samples_dir = (
        args.samples_dir
        if args.samples_dir is not None
        else args.out.parent / f"{args.out.stem}_samples"
    )
    samples_dir.mkdir(parents=True, exist_ok=True)
    n_to_save = (
        y_post.shape[0]
        if args.max_samples_saved is None
        else min(args.max_samples_saved, y_post.shape[0])
    )
    print(f"saving {n_to_save} individual posterior samples to {samples_dir}/ ...")

    def _save_panel(arr, path, title, cmap, lo=None, hi=None):
        fig_, ax = plt.subplots(figsize=(14, 5.0))
        kwargs = dict(cmap=cmap, origin="lower", aspect="equal")
        if lo is not None:
            kwargs["vmin"] = lo
            kwargs["vmax"] = hi
        im_ = ax.imshow(arr, **kwargs)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        fig_.colorbar(im_, ax=ax, fraction=0.012, pad=0.005)
        fig_.tight_layout()
        fig_.savefig(path, dpi=100, bbox_inches="tight")
        plt.close(fig_)

    # Reference panels with the shared color scale.
    _save_panel(truth_x, samples_dir / "truth.png", "truth x", "viridis", vmin, vmax)
    _save_panel(obs, samples_dir / "data.png",
                f"data = x + N(0, {args.sigma_noise})", "viridis", vmin, vmax)
    _save_panel(post_mean, samples_dir / "posterior_mean.png",
                "NUTS posterior mean", "viridis", vmin, vmax)
    _save_panel(a_mean, samples_dir / "posterior_mean_analytic.png",
                "analytic posterior mean", "viridis", vmin, vmax)
    _save_panel(post_std, samples_dir / "posterior_std.png",
                "NUTS posterior std", "magma")

    # Per-sample PNGs.
    for i in range(n_to_save):
        _save_panel(
            y_post[i],
            samples_dir / f"sample_{i:04d}.png",
            f"NUTS posterior sample #{i}",
            "viridis", vmin, vmax,
        )
    print(f"saved {n_to_save} sample PNGs + 5 reference panels in {samples_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
