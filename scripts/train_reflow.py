"""Rectified-flow reflow training: straighten the FM ODE to enable few-step sampling.

Loads paired (z, x) data produced by precompute_reflow_pairs.py and
retrains a velocity field on those coupled endpoints. The key difference
from the original FM training is the *coupling*: noise and data are
paired, not independent. After reflow:

- 1-step Euler sampling: ``x_pred = z + v_θ(z, t=0)`` -- usually 90%+ as
  good as the original 30-step Heun result.
- 4-step Heun: typically indistinguishable from the original.

Warm-starts from the source checkpoint by default (``--init-from-ckpt``)
so the new velocity field begins close to the parent's and converges in
~10 epochs instead of ~50.
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
import optax  # noqa: E402

from agorasynth.flow_matching import (  # noqa: E402
    FlowMatchingTrainState,
    make_fm_reflow_train_step,
)
from agorasynth.unet import UNet  # noqa: E402


def _save_checkpoint(
    path: Path, *, params, config, quantile_grid, z_grid, y0,
    patch_size_deg, pixel_size_arcmin, epoch, last_loss,
    reflow_round: int, source_ckpt: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(
            {
                "arch": "fm_unet_rectified",
                "params": jax.device_get(params),
                "config": config,
                "quantile_grid": np.asarray(quantile_grid, dtype=np.float64),
                "z_grid": np.asarray(z_grid, dtype=np.float64),
                "y0": float(y0),
                "patch_size_deg": float(patch_size_deg),
                "pixel_size_arcmin": float(pixel_size_arcmin),
                "epoch": int(epoch),
                "last_loss": float(last_loss),
                "reflow_round": int(reflow_round),
                "source_ckpt": str(source_ckpt),
            },
            f,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True,
                        help="Reflow pairs .npz from precompute_reflow_pairs.py.")
    parser.add_argument("--init-from-ckpt", type=Path, default=None,
                        help="Warm-start from this checkpoint (recommended). If "
                             "omitted, train from random init.")
    parser.add_argument("--out", type=Path, default=Path("checkpoints/fm_reflow.pkl"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Lower than the FM-only LR since we're warm-starting.")
    parser.add_argument("--channels", type=int, nargs="+", default=None,
                        help="Override config from --init-from-ckpt; required if "
                             "starting from random init.")
    parser.add_argument("--t-dim", type=int, default=128)
    parser.add_argument("--bottleneck-blocks", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=0)
    args = parser.parse_args(argv)

    print(f"jax devices: {jax.devices()}")

    print(f"loading reflow pairs from {args.pairs} ...")
    pairs = np.load(args.pairs)
    z_pairs = np.asarray(pairs["z_pairs"], dtype=np.float32)
    x_pairs = np.asarray(pairs["x_pairs"], dtype=np.float32)
    quantile_grid = np.asarray(pairs["quantile_grid"], dtype=np.float64)
    z_grid = np.asarray(pairs["z_grid"], dtype=np.float64)
    y0 = float(pairs["y0"])
    patch_size_deg = float(pairs["patch_size_deg"])
    pixel_size_arcmin = float(pairs["pixel_size_arcmin"])
    n_samples, h, w, c = z_pairs.shape
    if x_pairs.shape != z_pairs.shape:
        raise ValueError(
            f"shape mismatch: z {z_pairs.shape} vs x {x_pairs.shape}"
        )
    print(
        f"  pairs: N={n_samples}, patch={h}x{w}, channels={c}; "
        f"z range [{z_pairs.min():.3f}, {z_pairs.max():.3f}], "
        f"x range [{x_pairs.min():.3f}, {x_pairs.max():.3f}]"
    )

    # Build model and init params (or load from checkpoint).
    init_params = None
    config_dict = None
    source_ckpt_path = ""
    reflow_round = 1
    if args.init_from_ckpt is not None:
        print(f"loading init checkpoint {args.init_from_ckpt} ...")
        with open(args.init_from_ckpt, "rb") as f:
            init_ckpt = pickle.load(f)
        init_params = init_ckpt["params"]
        config_dict = dict(init_ckpt["config"])
        source_ckpt_path = str(args.init_from_ckpt)
        reflow_round = int(init_ckpt.get("reflow_round", 0)) + 1
        print(
            f"  warm-starting from arch={init_ckpt.get('arch')}, "
            f"reflow_round={init_ckpt.get('reflow_round', 0)} -> {reflow_round}"
        )
    else:
        if args.channels is None:
            raise ValueError(
                "--channels required when training reflow from random init."
            )
        config_dict = {
            "channels": list(args.channels),
            "t_dim": args.t_dim,
            "bottleneck_blocks": args.bottleneck_blocks,
            "out_channels": c,
            "patch_hw": [h, w],
        }

    model = UNet(
        channels=tuple(config_dict["channels"]),
        t_dim=config_dict["t_dim"],
        bottleneck_blocks=config_dict["bottleneck_blocks"],
        out_channels=config_dict["out_channels"],
    )

    rng_key = jax.random.PRNGKey(args.seed)
    if init_params is None:
        rng_key, init_key = jax.random.split(rng_key)
        init_params = model.init(
            init_key, jnp.zeros((1, h, w, c)), jnp.zeros((1,))
        )
    n_params = sum(int(p.size) for p in jax.tree_util.tree_leaves(init_params))
    print(f"model: UNet channels={config_dict['channels']}, ~{n_params:,} parameters")

    total_steps = max(1, (n_samples // args.batch_size) * args.epochs)
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=args.lr,
        warmup_steps=min(500, total_steps // 10),
        decay_steps=total_steps,
        end_value=0.0,
    )
    optimizer = optax.adam(schedule)
    state = FlowMatchingTrainState.create(
        apply_fn=model.apply, params=init_params, tx=optimizer
    )
    train_step_fn = make_fm_reflow_train_step(model.apply)

    rng_np = np.random.default_rng(args.seed + 1)
    n_batches = n_samples // args.batch_size
    rng_key, step_key = jax.random.split(rng_key)
    t0 = time.time()
    last_loss = float("nan")
    epoch = 0
    interrupted = False
    try:
        for epoch in range(1, args.epochs + 1):
            perm = rng_np.permutation(n_samples)
            running = 0.0
            for b in range(n_batches):
                idx = perm[b * args.batch_size : (b + 1) * args.batch_size]
                batch_x = jnp.asarray(x_pairs[idx])
                batch_z = jnp.asarray(z_pairs[idx])
                state, loss, step_key = train_step_fn(
                    state, batch_x, batch_z, step_key
                )
                running += float(loss)
            last_loss = running / max(n_batches, 1)
            if epoch % args.log_every == 0 or epoch == args.epochs:
                elapsed = (time.time() - t0) / 60.0
                print(
                    f"epoch {epoch:>3d}/{args.epochs}  loss={last_loss:.4f}  "
                    f"elapsed={elapsed:.1f} min"
                )
            if args.save_every and epoch % args.save_every == 0 and epoch != args.epochs:
                mid = args.out.with_name(f"{args.out.stem}.epoch{epoch}{args.out.suffix}")
                _save_checkpoint(
                    mid, params=state.params, config=config_dict,
                    quantile_grid=quantile_grid, z_grid=z_grid, y0=y0,
                    patch_size_deg=patch_size_deg, pixel_size_arcmin=pixel_size_arcmin,
                    epoch=epoch, last_loss=last_loss,
                    reflow_round=reflow_round, source_ckpt=source_ckpt_path,
                )
    except KeyboardInterrupt:
        interrupted = True
        print(f"\ninterrupted at epoch {epoch} -- saving current state ...")

    _save_checkpoint(
        args.out, params=state.params, config=config_dict,
        quantile_grid=quantile_grid, z_grid=z_grid, y0=y0,
        patch_size_deg=patch_size_deg, pixel_size_arcmin=pixel_size_arcmin,
        epoch=epoch, last_loss=last_loss,
        reflow_round=reflow_round, source_ckpt=source_ckpt_path,
    )
    tag = "interrupted" if interrupted else "completed"
    print(
        f"{tag}: saved checkpoint -> {args.out} "
        f"(reflow_round={reflow_round}, epoch={epoch}, loss={last_loss:.4f})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
