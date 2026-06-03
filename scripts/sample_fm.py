"""ODE-sample from a trained flow-matching velocity field.

Heun (2nd-order) integration of dx/dt = v_theta(x, t) from t=0 (noise) to
t=1 (data). The output is in gaussianized log-y space and inverted via
the saved (z_grid, quantile_grid) to physical Compton-y.

If --plot is given, compares against --compare-data when available. Use
--diagnostics to also write the 1-pt PDF + P(ell) diagnostics PNG.
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

from agorasynth.data import gaussianized_to_physical  # noqa: E402
from agorasynth.flow_matching import sample_euler_one_step, sample_heun  # noqa: E402
from agorasynth.unet import UNet  # noqa: E402


def _radial_power_spectrum(images, pixel_size_arcmin, n_bins=30):
    """Stack-averaged radial power spectrum. ``images`` shape ``(N, H, W)``."""
    arr = np.asarray(images)
    if arr.ndim != 3:
        raise ValueError(f"expected (N, H, W), got {arr.shape}")
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
    counts = np.zeros(n_bins)
    for k in range(n_bins):
        mask = digit == k
        if mask.any():
            P_binned[k] = P_flat[mask].mean()
            counts[k] = mask.sum()
    return centers, P_binned


def _plot_grid(samples_y, save_path, ncols=4, truth_y=None):
    import matplotlib.pyplot as plt

    n = samples_y.shape[0]
    log_samples = np.log10(np.maximum(samples_y, 1e-12))
    if truth_y is not None:
        m = min(truth_y.shape[0], n)
        log_truth = np.log10(np.maximum(truth_y[:m], 1e-12))
        combined = np.concatenate([log_samples.ravel(), log_truth.ravel()])
        vmin = float(np.percentile(combined, 1))
        vmax = float(np.percentile(combined, 99))
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(
            2 * nrows, ncols,
            figsize=(2.6 * ncols, 2.6 * 2 * nrows + 0.5),
            squeeze=False,
        )
        im = None
        # Truth first (top half).
        for i in range(nrows * ncols):
            ax = axes[i // ncols, i % ncols]
            if i < m:
                im = ax.imshow(log_truth[i], cmap="inferno", origin="lower",
                               vmin=vmin, vmax=vmax)
                ax.set_title(f"truth #{i}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            if i >= m:
                ax.axis("off")
        # Samples below.
        for i in range(nrows * ncols):
            ax = axes[nrows + i // ncols, i % ncols]
            if i < n:
                im = ax.imshow(log_samples[i], cmap="inferno", origin="lower",
                               vmin=vmin, vmax=vmax)
                ax.set_title(f"sample #{i}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            if i >= n:
                ax.axis("off")
        fig.suptitle(r"Training truth (top) vs. FM samples (bottom) — $\log_{10}\,y$")
        fig.tight_layout(rect=(0.0, 0.0, 0.92, 0.97))
        cax = fig.add_axes((0.94, 0.05, 0.015, 0.9))
        fig.colorbar(im, cax=cax, label=r"$\log_{10}\,y$")
    else:
        vmin = float(np.percentile(log_samples, 1))
        vmax = float(np.percentile(log_samples, 99))
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(2.6 * ncols, 2.6 * nrows + 0.5), squeeze=False
        )
        im = None
        for i, ax in enumerate(axes.ravel()):
            if i < n:
                im = ax.imshow(log_samples[i], cmap="inferno", origin="lower",
                               vmin=vmin, vmax=vmax)
                ax.set_title(f"#{i}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            if i >= n:
                ax.axis("off")
        fig.suptitle(r"FM samples — $\log_{10}\,y$")
        fig.tight_layout(rect=(0.0, 0.0, 0.92, 0.97))
        cax = fig.add_axes((0.94, 0.05, 0.015, 0.9))
        fig.colorbar(im, cax=cax, label=r"$\log_{10}\,y$")
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def _plot_diagnostics(samples_x, train_x, pixel_size_arcmin, save_path):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    lo = float(min(samples_x.min(), train_x.min()))
    hi = float(max(samples_x.max(), train_x.max()))
    bins = np.linspace(lo, hi, 60)
    axes[0].hist(train_x.ravel(), bins=bins, density=True, alpha=0.5,
                 label="training", color="C0")
    axes[0].hist(samples_x.ravel(), bins=bins, density=True, alpha=0.5,
                 label="FM samples", color="C1")
    axes[0].set_xlabel("gaussianized log-y")
    axes[0].set_ylabel("density")
    axes[0].set_title("1-point PDF")
    axes[0].legend()

    ell_t, P_t = _radial_power_spectrum(train_x, pixel_size_arcmin)
    ell_s, P_s = _radial_power_spectrum(samples_x, pixel_size_arcmin)
    mask_t = (P_t > 0) & (ell_t > 0)
    mask_s = (P_s > 0) & (ell_s > 0)
    axes[1].loglog(ell_t[mask_t], P_t[mask_t], label="training", color="C0")
    axes[1].loglog(ell_s[mask_s], P_s[mask_s], label="FM samples", color="C1")
    axes[1].set_xlabel(r"$\ell$ [rad$^{-1}$]")
    axes[1].set_ylabel(r"$P(\ell)$")
    axes[1].set_title("Radial power spectrum")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True,
                        help="Checkpoint .pkl from train_fm_wph.py.")
    parser.add_argument("--n-samples", type=int, default=16)
    parser.add_argument("--sampler", choices=("heun", "euler"), default="heun",
                        help="heun integrates the ODE; euler is one network pass.")
    parser.add_argument("--n-steps", type=int, default=30,
                        help="Heun ODE steps; ignored by --sampler euler.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("samples/fm.npz"))
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--diagnostics", action="store_true",
                        help="Compute 1-point PDF and power-spectrum diagnostics.")
    parser.add_argument("--compare-data", type=Path, default=Path("data/train.npz"),
                        help="Training .npz for plots/diagnostics (skipped if missing).")
    parser.add_argument("--no-compare", action="store_true")
    args = parser.parse_args(argv)

    print(f"jax devices: {jax.devices()}")
    print(f"loading checkpoint {args.ckpt} ...")
    with open(args.ckpt, "rb") as f:
        ckpt = pickle.load(f)
    params = ckpt["params"]
    cfg = ckpt["config"]
    quantile_grid = np.asarray(ckpt["quantile_grid"], dtype=np.float64)
    z_grid = np.asarray(ckpt["z_grid"], dtype=np.float64)
    y0 = float(ckpt["y0"])
    pixel_size_arcmin = float(ckpt["pixel_size_arcmin"])
    h, w = cfg["patch_hw"]
    print(f"  patch={h}x{w}, channels={cfg['channels']}, y0={y0}")

    model = UNet(
        channels=tuple(cfg["channels"]),
        t_dim=cfg["t_dim"],
        bottleneck_blocks=cfg["bottleneck_blocks"],
        out_channels=cfg["out_channels"],
    )
    apply_fn = model.apply

    if args.sampler == "euler":
        print(f"sampling {args.n_samples} patches via one-step Euler ...")
        x_samples = sample_euler_one_step(
            apply_fn, params,
            n_samples=args.n_samples,
            spatial_shape=(h, w),
            n_channels=cfg["out_channels"],
            seed=args.seed,
        )
    else:
        print(f"sampling {args.n_samples} patches via Heun ({args.n_steps} steps) ...")
        x_samples = sample_heun(
            apply_fn, params,
            n_samples=args.n_samples,
            spatial_shape=(h, w),
            n_channels=cfg["out_channels"],
            n_steps=args.n_steps,
            seed=args.seed,
        )
    x_np = np.asarray(x_samples)                                # (N, H, W, 1)
    x_np_flat = x_np[..., 0]                                    # (N, H, W)
    y_np = np.asarray(gaussianized_to_physical(jnp.asarray(x_np_flat),
                                               quantile_grid, z_grid, y0=y0))
    print(
        f"  samples (x): mean={x_np.mean():.3f}, std={x_np.std():.3f}, "
        f"range [{x_np.min():.3f}, {x_np.max():.3f}]"
    )
    print(
        f"  samples (y): min={y_np.min():.3e}, median={float(np.median(y_np)):.3e}, "
        f"max={y_np.max():.3e}"
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out, x_samples=x_np, y_samples=y_np,
        quantile_grid=quantile_grid, z_grid=z_grid, y0=np.float64(y0),
        pixel_size_arcmin=np.float64(pixel_size_arcmin),
        sampler=args.sampler,
        n_steps=np.int64(args.n_steps),
    )
    print(f"saved samples -> {args.out}")

    train_x = None
    truth_y = None
    use_compare = (
        not args.no_compare
        and (args.plot or args.diagnostics)
        and args.compare_data is not None
        and args.compare_data.exists()
    )
    if use_compare:
        print(f"loading training data for comparison: {args.compare_data}")
        td = np.load(args.compare_data)
        train_x_chw = np.asarray(td["x_train"])
        train_x = train_x_chw[:, 0] if train_x_chw.ndim == 4 else train_x_chw
        train_qg = np.asarray(td["quantile_grid"], dtype=np.float64)
        train_zg = np.asarray(td["z_grid"], dtype=np.float64)
        train_y0 = float(td["y0"])
        truth_y = np.asarray(gaussianized_to_physical(
            jnp.asarray(train_x[:args.n_samples]), train_qg, train_zg, y0=train_y0
        ))

    if args.plot:
        plot_path = args.out.with_suffix(".png")
        _plot_grid(y_np, plot_path, truth_y=truth_y)
        print(f"saved sample grid -> {plot_path}")

    if args.diagnostics and train_x is not None:
        diag_path = args.out.with_name(f"{args.out.stem}_diagnostics.png")
        _plot_diagnostics(x_np_flat, train_x, pixel_size_arcmin, diag_path)
        print(f"saved diagnostics -> {diag_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
