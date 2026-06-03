"""Run the sampler N times in one process; measure wall-clock and stack results.

JIT compile happens once during the warmup pass; the N timed rounds reuse
the cached compile, so per-round time reflects the steady-state ODE cost
(no compile in the number).

Output:
- console summary with per-round and total wall-clock.
- a stacked grid PNG: ``n_rounds`` rows x ``n_per_round`` columns,
  shared color scale, with optional truth row on the bottom if
  ``--compare-data`` exists.
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--n-rounds", type=int, default=10)
    parser.add_argument("--n-per-round", type=int, default=8)
    parser.add_argument("--sampler", choices=("heun", "euler"), default="heun")
    parser.add_argument("--n-steps", type=int, default=30,
                        help="Heun ODE steps; ignored by --sampler euler.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("samples/benchmark.png"))
    parser.add_argument("--compare-data", type=Path, default=Path("data/train.npz"),
                        help="Training .npz; if present, adds a truth row at the bottom.")
    parser.add_argument("--no-compare", action="store_true")
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
    h, w = cfg["patch_hw"]
    c = cfg["out_channels"]
    print(
        f"  patch={h}x{w}, channels={cfg['channels']}, "
        f"sampler={args.sampler}, n_steps={args.n_steps if args.sampler == 'heun' else 1}"
    )

    model = UNet(
        channels=tuple(cfg["channels"]),
        t_dim=cfg["t_dim"],
        bottleneck_blocks=cfg["bottleneck_blocks"],
        out_channels=c,
    )

    def _sample(seed: int) -> jnp.ndarray:
        if args.sampler == "heun":
            return sample_heun(
                model.apply, params,
                n_samples=args.n_per_round, spatial_shape=(h, w),
                n_channels=c, n_steps=args.n_steps, seed=seed,
            )
        return sample_euler_one_step(
            model.apply, params,
            n_samples=args.n_per_round, spatial_shape=(h, w),
            n_channels=c, seed=seed,
        )

    print("warmup pass (triggers JIT compile) ...")
    t0 = time.time()
    _warm = _sample(args.seed)
    _warm.block_until_ready()
    t_warmup = time.time() - t0
    print(f"  warmup: {t_warmup:.2f} s")

    all_y = np.zeros((args.n_rounds, args.n_per_round, h, w), dtype=np.float32)
    times = []
    print(f"running {args.n_rounds} rounds x {args.n_per_round} samples ...")
    t_total_0 = time.time()
    for r in range(args.n_rounds):
        t0 = time.time()
        x = _sample(args.seed + 1 + r)
        x.block_until_ready()
        dt = time.time() - t0
        times.append(dt)
        x_np = np.asarray(x)[..., 0]
        y_np = np.asarray(gaussianized_to_physical(
            jnp.asarray(x_np), quantile_grid, z_grid, y0=y0
        ))
        all_y[r] = y_np
        print(f"  round {r + 1:>2d}/{args.n_rounds}: {dt:.3f} s")
    t_total = time.time() - t_total_0

    times_arr = np.asarray(times)
    print()
    print("timing summary")
    print(f"  warmup (incl. JIT compile): {t_warmup:.3f} s")
    print(f"  total of {args.n_rounds} timed rounds: {t_total:.3f} s")
    print(
        f"  per-round: mean {times_arr.mean():.3f} s, "
        f"min {times_arr.min():.3f} s, "
        f"max {times_arr.max():.3f} s"
    )
    print(
        f"  per-sample (n_per_round={args.n_per_round}): "
        f"{(times_arr.mean() / args.n_per_round) * 1000:.2f} ms/sample"
    )

    truth_y = None
    if not args.no_compare and args.compare_data.exists():
        td = np.load(args.compare_data)
        train_x_chw = np.asarray(td["x_train"])
        train_x = train_x_chw[:, 0] if train_x_chw.ndim == 4 else train_x_chw
        train_qg = np.asarray(td["quantile_grid"], dtype=np.float64)
        train_zg = np.asarray(td["z_grid"], dtype=np.float64)
        train_y0 = float(td["y0"])
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(train_x.shape[0], size=args.n_per_round, replace=False)
        truth_y = np.asarray(gaussianized_to_physical(
            jnp.asarray(train_x[idx]), train_qg, train_zg, y0=train_y0
        ))

    print(f"plotting {args.n_rounds} x {args.n_per_round} grid ...")
    import matplotlib.pyplot as plt

    panels_for_color = [all_y]
    if truth_y is not None:
        panels_for_color.append(truth_y[None, :, :, :])
    log_all_for_vlim = np.log10(np.maximum(np.concatenate(
        [p.reshape(-1, h, w) for p in panels_for_color], axis=0
    ), 1e-12))
    vmin = float(np.percentile(log_all_for_vlim, 1))
    vmax = float(np.percentile(log_all_for_vlim, 99))

    nrows = args.n_rounds + (1 if truth_y is not None else 0)
    ncols = args.n_per_round
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(2.0 * ncols, 2.0 * nrows + 0.5),
        squeeze=False,
    )
    im = None
    truth_offset = 1 if truth_y is not None else 0
    if truth_y is not None:
        for col in range(ncols):
            ax = axes[0, col]
            im = ax.imshow(
                np.log10(np.maximum(truth_y[col], 1e-12)),
                cmap="inferno", origin="lower", vmin=vmin, vmax=vmax,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel("truth", fontsize=9, color="C0")
            ax.set_title(f"#{col}", fontsize=9)
    for r in range(args.n_rounds):
        for col in range(ncols):
            ax = axes[r + truth_offset, col]
            im = ax.imshow(
                np.log10(np.maximum(all_y[r, col], 1e-12)),
                cmap="inferno", origin="lower", vmin=vmin, vmax=vmax,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(f"round {r + 1}", fontsize=9)
            if truth_y is None and r == 0:
                ax.set_title(f"#{col}", fontsize=9)

    title = (
        f"{args.sampler} sampler"
        + (f", n_steps={args.n_steps}" if args.sampler == "heun" else "")
        + f"  |  {args.n_rounds} rounds x {args.n_per_round} samples"
        + f"  |  per-round mean {times_arr.mean():.2f}s, "
        + f"warmup {t_warmup:.1f}s"
    )
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0.0, 0.0, 0.92, 0.97))
    cax = fig.add_axes((0.94, 0.05, 0.015, 0.9))
    fig.colorbar(im, cax=cax, label=r"$\log_{10}\,y$")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
