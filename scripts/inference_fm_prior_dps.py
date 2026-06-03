"""DPS (Diffusion Posterior Sampling) on flow matching: FM model as prior.

Replaces the trivial N(0,1) per-pixel prior in inference_simple.py with the
trained FM model. At each ODE step from t=0 -> t=1 we modify the velocity
by adding the gradient of -log p(d | x_hat_1) w.r.t. x_t, where
x_hat_1 = x_t + (1 - t) * v_theta(x_t, t) is the predicted clean sample.

Algorithm (per step):

    v        = v_theta(x_t, t)
    x_hat_1  = x_t + (1 - t) * v
    L        = 0.5 * || d - x_hat_1 ||^2 / sigma_noise^2          # neg log p(d | x_hat_1)
    grad     = d/dx_t L                                            # via jax autograd
    drift    = v - lambda_data * grad
    x_{t+dt} = x_t + dt * drift

The first term is the FM prior's drift (push toward the prior manifold);
the second is the data attraction (push toward y consistent with d). Their
balance is set by lambda_data. At high sigma_noise, the data gradient is
weak -> samples come essentially from the FM prior (realistic Compton-y).
At low sigma_noise, the data gradient anchors x near d -> samples
reproduce truth tightly.

One ODE integration per posterior sample. Cost ~= unconditional sampling
+ a backward pass through the FM model per step (so ~2-3x slower than
unconditional). For SPT-1500 at full resolution that's roughly 20-40 s
per sample after JIT; 100 samples ~ 30-60 minutes.
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from agorasynth.unet import UNet  # noqa: E402


# ---------------------------------------------------------------------------
# SHT-project the Agora HEALPix truth onto the global ZEA grid
# (mirror of inference_simple.py / benchmark_full_map.py)
# ---------------------------------------------------------------------------


def _build_zea_wcs(ra_min, ra_max, dec_min, dec_max, pix_arcmin, patch_pad):
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


def _sht_project(hmap_data, wcs, n_pix_x, n_pix_y, lmax, nest, cache_path):
    if cache_path.exists():
        cached = np.load(cache_path)
        if cached.shape == (n_pix_y, n_pix_x):
            print(f"  loaded cached SHT-projected truth from {cache_path}")
            return cached.astype(np.float32)

    import healpy as hp
    from pixell import curvedsky, enmap

    if nest:
        hmap_data = hp.reorder(hmap_data, n2r=True)
    nside = hp.get_nside(hmap_data)
    print(f"  map2alm (nside={nside}, lmax={lmax}) ...")
    t0 = time.time()
    alm = hp.map2alm(hmap_data, lmax=lmax, iter=0, use_pixel_weights=True)
    print(f"    map2alm done in {time.time() - t0:.1f} s")
    omap = enmap.zeros((n_pix_y, n_pix_x), wcs)
    omap = curvedsky.alm2map(alm, omap)
    arr = np.asarray(omap, dtype=np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, arr)
    print(f"  cached -> {cache_path}")
    return arr


# ---------------------------------------------------------------------------
# DPS inner loop
# ---------------------------------------------------------------------------


def make_dps_step(apply_fn, sigma_noise: float, lambda_data: float):
    """Build a JIT'd single-step DPS update.

    State is ``x`` of shape (1, H, W, 1). Returns a function
    ``step(x, t_scalar, dt, params, data) -> x_next``.
    """
    sigma2 = jnp.float32(sigma_noise ** 2)
    lam = jnp.float32(lambda_data)

    @partial(jax.jit, static_argnames=())
    def step(x, t_scalar, dt, params, data):
        t_b = jnp.full((x.shape[0],), t_scalar, dtype=jnp.float32)

        def loss_with_v(x_t):
            v = apply_fn(params, x_t, t_b)
            x_hat_1 = x_t + (1.0 - t_scalar) * v
            ll_neg = 0.5 * jnp.sum((data - x_hat_1[0, ..., 0]) ** 2) / sigma2
            return ll_neg, v

        (ll, v), grad_x = jax.value_and_grad(loss_with_v, has_aux=True)(x)
        drift = v - lam * grad_x
        return x + dt * drift, ll, jnp.linalg.norm(grad_x), jnp.linalg.norm(v)

    return step


def dps_sample(
    apply_fn, params, init_noise, data, sigma_noise, n_steps, lambda_data,
    log_every: int = 0, sample_label: str = "",
):
    """Run one DPS posterior sample. ``data`` is (H, W) in gaussianized space."""
    step_fn = make_dps_step(apply_fn, sigma_noise, lambda_data)
    x = init_noise
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t = float(i) * dt
        x, ll, gnorm, vnorm = step_fn(x, jnp.float32(t), jnp.float32(dt), params, data)
        if log_every and (i % log_every == 0 or i == n_steps - 1):
            print(
                f"  {sample_label}step {i + 1:>3d}/{n_steps}  t={t:.3f}  "
                f"-log p={float(ll):.3e}  ||grad||={float(gnorm):.3e}  "
                f"||v||={float(vnorm):.3e}"
            )
    return x


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _save_panel(arr, path, title, cmap, lo=None, hi=None):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 5.0))
    kwargs = dict(cmap=cmap, origin="lower", aspect="equal")
    if lo is not None:
        kwargs["vmin"] = lo
        kwargs["vmax"] = hi
    im = ax.imshow(arr, **kwargs)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.012, pad=0.005)
    fig.tight_layout()
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True,
                        help="FM checkpoint .pkl from train_fm_wph.py.")
    parser.add_argument("--data", type=Path, default=Path("data/train.npz"),
                        help="Dataset .npz with quantile_grid + z_grid.")
    parser.add_argument("--healpix-truth", type=Path,
                        default=Path("/global/cfs/cdirs/mp107c/yomori/agora/products/"
                                     "components/tsz/len/"
                                     "agora_ltszNG_bahamas80_bnd_unb_1.0e+12_1.0e+18_lensed.fits"))
    parser.add_argument("--truth-nest", action="store_true")
    parser.add_argument("--truth-lmax", type=int, default=16000)
    parser.add_argument("--ra-min", type=float, default=-50.0)
    parser.add_argument("--ra-max", type=float, default=50.0)
    parser.add_argument("--dec-min", type=float, default=-70.0)
    parser.add_argument("--dec-max", type=float, default=-40.0)
    parser.add_argument("--pixel-arcmin", type=float, default=1.6)
    parser.add_argument("--patch-pad", type=int, default=94)
    parser.add_argument("--sigma-noise", type=float, default=0.5)
    parser.add_argument("--lambda-data", type=float, default=1.0,
                        help="Strength of the data-attraction term in the DPS drift.")
    parser.add_argument("--n-steps", type=int, default=30)
    parser.add_argument("--n-posterior-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vmin", type=float, default=-4.0)
    parser.add_argument("--vmax", type=float, default=4.0)
    parser.add_argument("--out", type=Path, default=Path("samples/inference_fm_dps.png"))
    parser.add_argument("--samples-dir", type=Path, default=None)
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args(argv)

    print(f"jax devices: {jax.devices()}")

    # ----- load FM checkpoint
    print(f"loading FM checkpoint {args.ckpt} ...")
    with open(args.ckpt, "rb") as f:
        ckpt = pickle.load(f)
    params = ckpt["params"]
    cfg = ckpt["config"]
    quantile_grid = np.asarray(ckpt["quantile_grid"], dtype=np.float64)
    z_grid = np.asarray(ckpt["z_grid"], dtype=np.float64)
    y0 = float(ckpt["y0"])
    pixel_arcmin = float(ckpt["pixel_size_arcmin"])
    if abs(pixel_arcmin - args.pixel_arcmin) > 1e-6:
        print(f"  WARNING: ckpt pixel size {pixel_arcmin} != --pixel-arcmin {args.pixel_arcmin}")
    n_channels = cfg["out_channels"]

    model = UNet(
        channels=tuple(cfg["channels"]),
        t_dim=cfg["t_dim"],
        bottleneck_blocks=cfg["bottleneck_blocks"],
        out_channels=n_channels,
    )
    apply_fn = model.apply

    # ----- build SPT-1500 ZEA grid
    wcs, n_pix_x, n_pix_y, crpix_x, crpix_y = _build_zea_wcs(
        args.ra_min, args.ra_max, args.dec_min, args.dec_max,
        args.pixel_arcmin, args.patch_pad,
    )
    print(f"global ZEA grid: {n_pix_y} x {n_pix_x}")

    # ----- project Agora truth via SHT
    if not args.healpix_truth.exists():
        raise FileNotFoundError(args.healpix_truth)
    print(f"projecting truth {args.healpix_truth} ...")
    import healpy as hp

    hmap = hp.read_map(str(args.healpix_truth))
    cache_tag = (
        f"{args.healpix_truth.stem}.sht_{n_pix_y}x{n_pix_x}"
        f"_crpix{int(crpix_x)}_{int(crpix_y)}.npy"
    )
    cache_path = args.healpix_truth.with_name(cache_tag)
    truth_y = _sht_project(
        hmap, wcs, n_pix_x, n_pix_y,
        lmax=args.truth_lmax, nest=args.truth_nest, cache_path=cache_path,
    )
    print(
        f"  truth y: {truth_y.shape}, "
        f"y range [{truth_y.min():.3e}, {truth_y.max():.3e}]"
    )

    # ----- gaussianize
    log_y = np.log(np.maximum(truth_y, 1e-30) + y0)
    truth_x = np.interp(log_y.ravel(), quantile_grid, z_grid).reshape(truth_y.shape).astype(np.float32)
    print(
        f"  truth x: range [{truth_x.min():.3f}, {truth_x.max():.3f}], "
        f"std {truth_x.std():.3f}"
    )

    # ----- mock observation
    rng = np.random.default_rng(args.seed)
    noise = rng.standard_normal(truth_x.shape) * args.sigma_noise
    obs = (truth_x + noise).astype(np.float32)
    print(
        f"  mock data: sigma_noise={args.sigma_noise}, "
        f"data range [{obs.min():.3f}, {obs.max():.3f}]"
    )

    # ----- DPS posterior sampling
    print(
        f"\nDPS posterior sampling: {args.n_posterior_samples} samples, "
        f"n_steps={args.n_steps}, lambda_data={args.lambda_data}"
    )
    obs_j = jnp.asarray(obs)
    posterior_samples = np.zeros(
        (args.n_posterior_samples, n_pix_y, n_pix_x), dtype=np.float32
    )
    t_total = time.time()
    for s in range(args.n_posterior_samples):
        z_np = rng.standard_normal((1, n_pix_y, n_pix_x, n_channels)).astype(np.float32)
        z_j = jnp.asarray(z_np)
        t0 = time.time()
        x_post = dps_sample(
            apply_fn, params,
            init_noise=z_j,
            data=obs_j,
            sigma_noise=args.sigma_noise,
            n_steps=args.n_steps,
            lambda_data=args.lambda_data,
            log_every=args.log_every if s == 0 else 0,
            sample_label=f"sample {s + 1}: " if args.log_every and s == 0 else "",
        )
        x_post.block_until_ready()
        elapsed = time.time() - t0
        x_post_np = np.asarray(x_post)[0, ..., 0]
        posterior_samples[s] = x_post_np
        print(
            f"  sample {s + 1:>3d}/{args.n_posterior_samples}: {elapsed:.1f} s "
            f"(range [{x_post_np.min():.2f}, {x_post_np.max():.2f}])"
        )
    print(f"DPS total time: {(time.time() - t_total) / 60.0:.1f} min")

    post_mean = posterior_samples.mean(axis=0)
    post_std = posterior_samples.std(axis=0)

    # ----- plot 5-panel summary
    print("plotting summary ...")
    import matplotlib.pyplot as plt

    panels = [
        ("truth x", truth_x, "viridis"),
        (f"data = x + N(0, {args.sigma_noise})", obs, "viridis"),
        (f"DPS posterior mean (n={args.n_posterior_samples})", post_mean, "viridis"),
        ("DPS posterior std", post_std, "magma"),
    ]
    fig, axes = plt.subplots(
        len(panels), 1,
        figsize=(14.0, 3.0 * len(panels) + 1),
        squeeze=False,
    )
    for ax, (title, img, cmap) in zip(axes[:, 0], panels):
        kwargs = dict(cmap=cmap, origin="lower", aspect="equal")
        if "std" not in title:
            kwargs["vmin"] = args.vmin
            kwargs["vmax"] = args.vmax
        im = ax.imshow(img, **kwargs)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.012, pad=0.005)
    fig.suptitle(
        f"DPS posterior with FM prior, {n_pix_y}x{n_pix_x}, "
        f"sigma_noise={args.sigma_noise}, lambda_data={args.lambda_data}, "
        f"{args.n_steps} ODE steps",
        fontsize=11,
    )
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {args.out}")

    # ----- individual sample PNGs
    samples_dir = (
        args.samples_dir if args.samples_dir is not None
        else args.out.parent / f"{args.out.stem}_samples"
    )
    samples_dir.mkdir(parents=True, exist_ok=True)
    print(f"saving individual panels to {samples_dir}/ ...")
    _save_panel(truth_x, samples_dir / "truth.png", "truth x", "viridis", args.vmin, args.vmax)
    _save_panel(obs, samples_dir / "data.png",
                f"data = x + N(0, {args.sigma_noise})", "viridis", args.vmin, args.vmax)
    _save_panel(post_mean, samples_dir / "posterior_mean.png",
                f"DPS posterior mean (n={args.n_posterior_samples})", "viridis",
                args.vmin, args.vmax)
    _save_panel(post_std, samples_dir / "posterior_std.png",
                "DPS posterior std", "magma")
    for i in range(args.n_posterior_samples):
        _save_panel(
            posterior_samples[i],
            samples_dir / f"sample_{i:04d}.png",
            f"DPS posterior sample #{i}",
            "viridis", args.vmin, args.vmax,
        )
    print(f"saved {args.n_posterior_samples} sample PNGs in {samples_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
