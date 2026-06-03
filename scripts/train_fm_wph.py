"""Train a flow-matching velocity field with a WPH-distribution-matching loss.

Combined loss:

    L_total = L_FM + lambda_wph * gate(t) * L_WPH

where ``L_FM`` is the standard rectified-flow regression loss and ``L_WPH``
is a moment-match between the batch's WPH features and the prior's (mu,
Sigma), in whitened feature coordinates. The gate(t) sigmoid-mask focuses
the WPH term on samples where the predicted clean field x_hat_1 is a
confident reconstruction (large t).
"""

from __future__ import annotations

import argparse
import math
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
    make_fm_only_train_step,
    make_train_step,
    make_train_step_persample,
    make_wph_features_fn,
    whitener_from_prior,
)
from agorasynth.unet import UNet  # noqa: E402
from agorasynth.wph import WPHOp, WPHPriorStats  # noqa: E402


def _to_nhwc(x: np.ndarray) -> np.ndarray:
    """(N, 1, H, W) -> (N, H, W, 1)."""
    if x.ndim != 4:
        raise ValueError(f"expected 4D x_train, got shape {x.shape}")
    return np.transpose(x, (0, 2, 3, 1))


def _save_checkpoint(
    path: Path,
    *,
    params,
    config: dict,
    quantile_grid,
    z_grid,
    y0,
    patch_size_deg,
    pixel_size_arcmin,
    epoch: int,
    last_loss: float,
) -> None:
    import pickle

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(
            {
                "arch": "fm_unet",
                "params": jax.device_get(params),
                "config": config,
                "quantile_grid": np.asarray(quantile_grid, dtype=np.float64),
                "z_grid": np.asarray(z_grid, dtype=np.float64),
                "y0": float(y0),
                "patch_size_deg": float(patch_size_deg),
                "pixel_size_arcmin": float(pixel_size_arcmin),
                "epoch": int(epoch),
                "last_loss": float(last_loss),
            },
            f,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True,
                        help="Dataset .npz from build_dataset.py.")
    parser.add_argument("--wph-prior", type=Path, default=None,
                        help="WPH prior .npz from build_wph_prior.py. "
                             "If omitted, train flow matching only (no WPH loss).")
    parser.add_argument("--wph-mode", type=str, default="persample",
                        choices=["persample", "distribution"],
                        help="WPH loss mode. 'persample' (default) regresses each "
                             "predicted clean sample's WPH features against its "
                             "specific training-target features (recommended -- "
                             "captures rare features like cluster cores). "
                             "'distribution' matches batch-mean and per-feature var "
                             "to the prior's (mu, Sigma) -- weak for rare features.")
    parser.add_argument("--wph-targets", type=Path,
                        default=Path("data/wph_targets.npz"),
                        help="Precomputed WPH features from precompute_wph_targets.py "
                             "(used in --wph-mode persample).")
    parser.add_argument("--out", type=Path, default=Path("checkpoints/fm_wph.pkl"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lambda-wph", type=float, default=0.1,
                        help="Weight on the WPH distribution-matching loss term.")
    parser.add_argument("--wph-t-min", type=float, default=0.5,
                        help="Sigmoid gate midpoint for the WPH loss (t in [0, 1]).")
    parser.add_argument("--wph-chunk-size", type=int, default=1,
                        help="WPH feature batch chunk size. 1 = sequential per-patch "
                             "(safest for cuFFT scratch); >1 = vmap'd chunks (faster, "
                             "uses more memory). Try 4 or 8 if you have headroom.")
    parser.add_argument("--wph-warmup-epochs", type=float, default=5.0,
                        help="Linearly ramp lambda_wph from 0 to its target value over "
                             "this many epochs. Prevents mode collapse from the huge "
                             "initial WPH loss at random init. Set to 0 to disable.")
    parser.add_argument("--channels", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--t-dim", type=int, default=128)
    parser.add_argument("--bottleneck-blocks", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=0)
    args = parser.parse_args(argv)

    print(f"jax devices: {jax.devices()}")

    print(f"loading {args.data} ...")
    data = np.load(args.data)
    x_train_chw = np.asarray(data["x_train"], dtype=np.float32)        # (N, 1, H, W)
    x_train = _to_nhwc(x_train_chw)                                    # (N, H, W, 1)
    n_samples, h, w, c = x_train.shape
    print(
        f"  x_train: {x_train.shape} (NHWC) range "
        f"[{x_train.min():.3f}, {x_train.max():.3f}]"
    )

    quantile_grid = np.asarray(data["quantile_grid"], dtype=np.float64)
    z_grid = np.asarray(data["z_grid"], dtype=np.float64)
    y0 = float(data["y0"])
    patch_size_deg = float(data["patch_size_deg"])
    pixel_size_arcmin = float(data["pixel_size_arcmin"])

    use_wph = args.wph_prior is not None
    F_targets_np = None
    inv_std_np = None
    whitener = None
    mu_prior_j = None
    if use_wph:
        print(f"loading WPH prior {args.wph_prior} ...")
        prior = WPHPriorStats.load(args.wph_prior)
        if prior.config.M != h or prior.config.N != w:
            raise ValueError(
                f"WPH prior config (M, N) = ({prior.config.M}, {prior.config.N}) "
                f"does not match patch size ({h}, {w})."
            )
        print(
            f"  prior: n_features={prior.n_features}, "
            f"fit on n_samples={prior.n_samples}, mode={args.wph_mode}"
        )
        op = WPHOp.build(prior.config)
        wph_features_fn = make_wph_features_fn(op, chunk_size=args.wph_chunk_size)
        print(f"  wph features: chunk_size={args.wph_chunk_size}")
        if args.wph_mode == "distribution":
            whitener = whitener_from_prior(prior.mean, prior.cov)
            mu_prior_j = jnp.asarray(prior.mean, dtype=jnp.float32)
        else:  # persample
            if not args.wph_targets.exists():
                raise FileNotFoundError(
                    f"--wph-mode persample needs precomputed targets at "
                    f"{args.wph_targets}; run scripts/precompute_wph_targets.py first."
                )
            print(f"  loading per-sample WPH targets from {args.wph_targets} ...")
            tgt = np.load(args.wph_targets)
            F_targets_np = np.asarray(tgt["F_targets"], dtype=np.float32)
            inv_std_np = np.asarray(tgt["inv_std"], dtype=np.float32)
            if F_targets_np.shape[0] != n_samples:
                raise ValueError(
                    f"WPH targets ({F_targets_np.shape[0]}) and x_train "
                    f"({n_samples}) have different sample counts."
                )
            if F_targets_np.shape[1] != prior.n_features:
                raise ValueError(
                    f"WPH targets feature count ({F_targets_np.shape[1]}) "
                    f"differs from prior's ({prior.n_features}) -- "
                    "rebuild targets against the same prior."
                )
            print(
                f"  targets: shape={F_targets_np.shape}, "
                f"inv_std median={float(np.median(inv_std_np)):.3e}"
            )
    else:
        print("WPH prior not provided; training flow matching only (L_WPH = 0).")

    rng_key = jax.random.PRNGKey(args.seed)
    model = UNet(
        channels=tuple(args.channels),
        t_dim=args.t_dim,
        bottleneck_blocks=args.bottleneck_blocks,
        out_channels=c,
    )

    rng_key, init_key = jax.random.split(rng_key)
    dummy_x = jnp.zeros((1, h, w, c), dtype=jnp.float32)
    dummy_t = jnp.zeros((1,), dtype=jnp.float32)
    params = model.init(init_key, dummy_x, dummy_t)
    n_params = sum(int(p.size) for p in jax.tree_util.tree_leaves(params))
    print(f"model: UNet channels={args.channels}, ~{n_params:,} parameters")

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
        apply_fn=model.apply, params=params, tx=optimizer
    )

    if use_wph:
        n_batches_per_epoch_for_warmup = max(1, n_samples // args.batch_size)
        warmup_steps_for_wph = int(args.wph_warmup_epochs * n_batches_per_epoch_for_warmup)
        print(
            f"  wph lambda warmup: {args.wph_warmup_epochs:.1f} epochs "
            f"({warmup_steps_for_wph} steps), peak lambda={args.lambda_wph}"
        )
        if args.wph_mode == "distribution":
            train_step_fn = make_train_step(
                apply_fn=model.apply,
                wph_features_fn=wph_features_fn,
                mu_prior=mu_prior_j,
                whitener=whitener,
                z_grid=z_grid,
                quantile_grid=quantile_grid,
                y0=y0,
                lambda_wph=args.lambda_wph,
                wph_t_min=args.wph_t_min,
                lambda_warmup_steps=warmup_steps_for_wph,
            )
        else:  # persample
            train_step_fn = make_train_step_persample(
                apply_fn=model.apply,
                wph_features_fn=wph_features_fn,
                inv_std_per_feature=jnp.asarray(inv_std_np, dtype=jnp.float32),
                z_grid=z_grid,
                quantile_grid=quantile_grid,
                y0=y0,
                lambda_wph=args.lambda_wph,
                wph_t_min=args.wph_t_min,
                lambda_warmup_steps=warmup_steps_for_wph,
            )
    else:
        train_step_fn = make_fm_only_train_step(model.apply)

    config_dict = {
        "channels": list(args.channels),
        "t_dim": args.t_dim,
        "bottleneck_blocks": args.bottleneck_blocks,
        "out_channels": c,
        "patch_hw": [h, w],
    }

    rng_np = np.random.default_rng(args.seed + 1)
    n_batches_per_epoch = n_samples // args.batch_size

    rng_key, step_key = jax.random.split(rng_key)
    t0 = time.time()
    last_loss = float("nan")
    epoch = 0
    interrupted = False
    try:
        for epoch in range(1, args.epochs + 1):
            perm = rng_np.permutation(n_samples)
            running, running_fm, running_wph = 0.0, 0.0, 0.0
            for b in range(n_batches_per_epoch):
                idx = perm[b * args.batch_size : (b + 1) * args.batch_size]
                batch = jnp.asarray(x_train[idx])

                if use_wph:
                    if args.wph_mode == "persample":
                        F_target_batch = jnp.asarray(F_targets_np[idx])
                        state, loss, l_fm, l_wph, step_key = train_step_fn(
                            state, batch, F_target_batch, step_key
                        )
                    else:
                        state, loss, l_fm, l_wph, step_key = train_step_fn(
                            state, batch, step_key
                        )
                    running_fm += float(l_fm)
                    running_wph += float(l_wph)
                else:
                    state, loss, step_key = train_step_fn(state, batch, step_key)
                    running_fm += float(loss)
                running += float(loss)

            last_loss = running / max(n_batches_per_epoch, 1)
            avg_fm = running_fm / max(n_batches_per_epoch, 1)
            avg_wph = running_wph / max(n_batches_per_epoch, 1)
            if epoch % args.log_every == 0 or epoch == args.epochs:
                elapsed = (time.time() - t0) / 60.0
                if use_wph:
                    print(
                        f"epoch {epoch:>3d}/{args.epochs}  loss={last_loss:.4f}  "
                        f"L_FM={avg_fm:.4f}  L_WPH={avg_wph:.4f}  "
                        f"elapsed={elapsed:.1f} min"
                    )
                else:
                    print(
                        f"epoch {epoch:>3d}/{args.epochs}  L_FM={avg_fm:.4f}  "
                        f"elapsed={elapsed:.1f} min"
                    )

            if args.save_every and epoch % args.save_every == 0 and epoch != args.epochs:
                mid = args.out.with_name(f"{args.out.stem}.epoch{epoch}{args.out.suffix}")
                _save_checkpoint(
                    mid, params=state.params, config=config_dict,
                    quantile_grid=quantile_grid, z_grid=z_grid, y0=y0,
                    patch_size_deg=patch_size_deg, pixel_size_arcmin=pixel_size_arcmin,
                    epoch=epoch, last_loss=last_loss,
                )
    except KeyboardInterrupt:
        interrupted = True
        print(f"\ninterrupted at epoch {epoch} — saving current state ...")

    _save_checkpoint(
        args.out, params=state.params, config=config_dict,
        quantile_grid=quantile_grid, z_grid=z_grid, y0=y0,
        patch_size_deg=patch_size_deg, pixel_size_arcmin=pixel_size_arcmin,
        epoch=epoch, last_loss=last_loss,
    )
    tag = "interrupted" if interrupted else "completed"
    print(f"{tag}: saved checkpoint -> {args.out} (epoch {epoch}, loss {last_loss:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
