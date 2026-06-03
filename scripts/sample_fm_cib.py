"""ODE-sample 3-channel CIB patches from a trained joint flow-matching model.

Multi-channel analog of sample_fm.py. Output is (N, H, W, C) gaussianized,
inverted per channel to physical intensity via the saved per-band
(z_grid, quantile_grid, y0). Optional per-band diagnostics: 1-pt PDF and
radial power spectrum vs. the training set, plus a per-band sample grid.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from agorasynth.data import gaussianized_to_physical_multi  # noqa: E402
from agorasynth.flow_matching import sample_euler_one_step, sample_heun  # noqa: E402
from agorasynth.unet import UNet  # noqa: E402


def _radial_power_spectrum(images, pixel_size_arcmin, n_bins=30):
    """Stack-averaged radial power spectrum. ``images`` (N, H, W)."""
    arr = np.asarray(images)
    N, H, W = arr.shape
    pix_rad = (pixel_size_arcmin / 60.0) * np.pi / 180.0
    F = np.fft.fft2(arr - arr.mean(axis=(-2, -1), keepdims=True))
    P2d = (np.abs(F) ** 2).mean(axis=0) * pix_rad ** 2 / (H * W)
    fy = np.fft.fftfreq(H, d=pix_rad) * 2 * np.pi
    fx = np.fft.fftfreq(W, d=pix_rad) * 2 * np.pi
    fxg, fyg = np.meshgrid(fx, fy, indexing="xy")
    ell = np.sqrt(fxg ** 2 + fyg ** 2).ravel()
    P_flat = P2d.ravel()
    ell_max = ell.max()
    bins = np.logspace(np.log10(max(ell_max / 1000, 1.0)), np.log10(ell_max), n_bins + 1)
    digit = np.digitize(ell, bins) - 1
    centers = 0.5 * (bins[:-1] + bins[1:])
    P_binned = np.zeros(n_bins)
    for k in range(n_bins):
        mask = digit == k
        if mask.any():
            P_binned[k] = P_flat[mask].mean()
    return centers, P_binned


def _plot_band_diagnostics(samples_x, train_x, pixel_size_arcmin, bands, save_path):
    import matplotlib.pyplot as plt

    c = samples_x.shape[-1]
    fig, axes = plt.subplots(2, c, figsize=(4.2 * c, 8), squeeze=False)
    for ch in range(c):
        sx = samples_x[..., ch].ravel()
        tx = train_x[..., ch].ravel()
        lo, hi = float(min(sx.min(), tx.min())), float(max(sx.max(), tx.max()))
        bins = np.linspace(lo, hi, 60)
        axes[0, ch].hist(tx, bins=bins, density=True, alpha=0.5, label="train", color="C0")
        axes[0, ch].hist(sx, bins=bins, density=True, alpha=0.5, label="FM", color="C1")
        axes[0, ch].set_title(f"{bands[ch]} GHz  1-pt PDF (gaussianized)")
        axes[0, ch].legend()
        ell_t, P_t = _radial_power_spectrum(train_x[..., ch], pixel_size_arcmin)
        ell_s, P_s = _radial_power_spectrum(samples_x[..., ch], pixel_size_arcmin)
        mt, ms = (P_t > 0) & (ell_t > 0), (P_s > 0) & (ell_s > 0)
        axes[1, ch].loglog(ell_t[mt], P_t[mt], label="train", color="C0")
        axes[1, ch].loglog(ell_s[ms], P_s[ms], label="FM", color="C1")
        axes[1, ch].set_title(f"{bands[ch]} GHz  P(ell)")
        axes[1, ch].set_xlabel(r"$\ell$ [rad$^{-1}$]")
        axes[1, ch].legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--n-samples", type=int, default=16)
    parser.add_argument("--sampler", choices=("heun", "euler"), default="heun")
    parser.add_argument("--n-steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("samples/fm_cib.npz"))
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--compare-data", type=Path, default=Path("data/train_cib.npz"))
    parser.add_argument("--no-compare", action="store_true")
    args = parser.parse_args(argv)

    print(f"jax devices: {jax.devices()}")
    with open(args.ckpt, "rb") as f:
        ckpt = pickle.load(f)
    params = ckpt["params"]
    cfg = ckpt["config"]
    quantile_grid = np.asarray(ckpt["quantile_grid"], dtype=np.float64)   # (C, nq)
    z_grid = np.asarray(ckpt["z_grid"], dtype=np.float64)
    y0 = np.asarray(ckpt["y0"], dtype=np.float64)                          # (C,)
    bands = ckpt.get("bands", list(range(cfg["out_channels"])))
    pixel_size_arcmin = float(ckpt["pixel_size_arcmin"])
    h, w = cfg["patch_hw"]
    c = cfg["out_channels"]
    print(f"  patch={h}x{w}, {c} bands {bands}")

    model = UNet(channels=tuple(cfg["channels"]), t_dim=cfg["t_dim"],
                 bottleneck_blocks=cfg["bottleneck_blocks"], out_channels=c)

    if args.sampler == "euler":
        print(f"sampling {args.n_samples} patches via one-step Euler ...")
        x_samples = sample_euler_one_step(
            model.apply, params, n_samples=args.n_samples,
            spatial_shape=(h, w), n_channels=c, seed=args.seed,
        )
    else:
        print(f"sampling {args.n_samples} patches via Heun ({args.n_steps} steps) ...")
        x_samples = sample_heun(
            model.apply, params, n_samples=args.n_samples,
            spatial_shape=(h, w), n_channels=c, n_steps=args.n_steps, seed=args.seed,
        )
    x_np = np.asarray(x_samples)                                  # (N, H, W, C)
    y_np = np.asarray(gaussianized_to_physical_multi(
        jnp.asarray(x_np), quantile_grid, z_grid, y0=y0
    ))                                                            # (N, H, W, C)
    for ch, band in enumerate(bands):
        b = y_np[..., ch]
        print(f"  {band} GHz: min={b.min():.3e} median={float(np.median(b)):.3e} "
              f"max={b.max():.3e}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out, x_samples=x_np, y_samples=y_np,
        quantile_grid=quantile_grid, z_grid=z_grid, y0=y0,
        bands=np.asarray([str(b) for b in bands]),
        pixel_size_arcmin=np.float64(pixel_size_arcmin),
        sampler=args.sampler, n_steps=np.int64(args.n_steps),
    )
    print(f"saved samples -> {args.out}")

    if args.diagnostics and not args.no_compare and args.compare_data.exists():
        td = np.load(args.compare_data)
        train_x = np.transpose(np.asarray(td["x_train"]), (0, 2, 3, 1))  # (N,H,W,C)
        diag_path = args.out.with_name(f"{args.out.stem}_diagnostics.png")
        _plot_band_diagnostics(
            x_np, train_x[: max(args.n_samples, 256)], pixel_size_arcmin,
            [str(b) for b in bands], diag_path,
        )
        print(f"saved diagnostics -> {diag_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
