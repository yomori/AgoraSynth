"""Train a joint 3-channel CIB flow-matching model with multi-channel WPH loss.

Multi-channel analog of train_fm_wph.py. One UNet emits all bands
(out_channels = C) from shared input noise, so inter-band CIB correlation is
preserved by construction. The WPH loss uses per-channel + cross-band features
(make_wph_features_multi_fn) with per-channel inverse rank transforms.

    L_total = L_FM + lambda_wph * gate(t) * L_WPH
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
    channel_pairs,
    make_fm_only_train_step,
    make_train_step_multi,
    make_train_step_persample_multi,
    make_wph_features_multi_fn,
    whitener_from_prior,
)
from agorasynth.unet import UNet  # noqa: E402
from agorasynth.wph import WPHConfig, WPHOp  # noqa: E402


def _save_checkpoint(path: Path, *, params, config, quantile_grid, z_grid, y0,
                     bands, patch_size_deg, pixel_size_arcmin, epoch, last_loss):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(
            {
                "arch": "fm_unet_cib",
                "params": jax.device_get(params),
                "config": config,
                "quantile_grid": np.asarray(quantile_grid, dtype=np.float64),  # (C, nq)
                "z_grid": np.asarray(z_grid, dtype=np.float64),
                "y0": np.asarray(y0, dtype=np.float64),                        # (C,)
                "bands": list(bands),
                "patch_size_deg": float(patch_size_deg),
                "pixel_size_arcmin": float(pixel_size_arcmin),
                "epoch": int(epoch),
                "last_loss": float(last_loss),
            },
            f,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--wph-prior", type=Path, default=None,
                        help="Multi-channel WPH prior from build_wph_prior_cib.py. "
                             "Omit for FM-only.")
    parser.add_argument("--wph-mode", choices=["persample", "distribution"],
                        default="persample")
    parser.add_argument("--wph-targets", type=Path,
                        default=Path("data/wph_targets_cib.npz"))
    parser.add_argument("--out", type=Path, default=Path("checkpoints/fm_wph_cib.pkl"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lambda-wph", type=float, default=1.0)
    parser.add_argument("--wph-t-min", type=float, default=0.5)
    parser.add_argument("--wph-chunk-size", type=int, default=1)
    parser.add_argument("--wph-warmup-epochs", type=float, default=5.0)
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
    x_train_nchw = np.asarray(data["x_train"], dtype=np.float32)   # (N, C, H, W)
    if x_train_nchw.ndim != 4:
        raise ValueError(f"expected (N, C, H, W) x_train, got {x_train_nchw.shape}")
    x_train = np.transpose(x_train_nchw, (0, 2, 3, 1))            # (N, H, W, C)
    n_samples, h, w, c = x_train.shape
    quantile_grid = np.asarray(data["quantile_grid"], dtype=np.float64)   # (C, nq)
    z_grid = np.asarray(data["z_grid"], dtype=np.float64)
    y0 = np.asarray(data["y0"], dtype=np.float64)                 # (C,)
    bands = [str(b) for b in data["bands"]] if "bands" in data.files else list(range(c))
    patch_size_deg = float(data["patch_size_deg"])
    pixel_size_arcmin = float(data["pixel_size_arcmin"])
    if quantile_grid.shape[0] != c or y0.shape[0] != c:
        raise ValueError(
            f"quantile_grid/y0 channel dim must be {c}, got "
            f"{quantile_grid.shape} / {y0.shape}"
        )
    print(f"  x_train: {x_train.shape} (NHWC), {c} bands {bands}, "
          f"range [{x_train.min():.3f}, {x_train.max():.3f}]")

    use_wph = args.wph_prior is not None
    train_step_fn = None
    F_targets_np = None
    if use_wph:
        print(f"loading WPH prior {args.wph_prior} ...")
        prior = np.load(args.wph_prior)
        if int(prior["n_channels"]) != c:
            raise ValueError(
                f"prior n_channels={int(prior['n_channels'])} != data channels={c}"
            )
        cfg = WPHConfig(M=int(prior["M"]), N=int(prior["N"]), J=int(prior["J"]),
                        L=int(prior["L"]), dn=int(prior["dn"]), A=int(prior["A"]))
        if cfg.M != h or cfg.N != w:
            raise ValueError(f"prior M=N={cfg.M} != patch {h}x{w}")
        op = WPHOp.build(cfg)
        wph_features_fn = make_wph_features_multi_fn(
            op, n_channels=c, chunk_size=args.wph_chunk_size, checkpoint=True,
        )
        n_blocks = c + len(channel_pairs(c))
        print(f"  WPH: J={cfg.J} L={cfg.L} dn={cfg.dn} A={cfg.A}, n_blocks={n_blocks}, "
              f"mode={args.wph_mode}, chunk_size={args.wph_chunk_size}")

        n_batches_per_epoch = max(1, n_samples // args.batch_size)
        warmup_steps = int(args.wph_warmup_epochs * n_batches_per_epoch)
        print(f"  lambda warmup: {args.wph_warmup_epochs:.1f} epochs "
              f"({warmup_steps} steps), peak lambda={args.lambda_wph}")

        if args.wph_mode == "distribution":
            whitener = whitener_from_prior(
                np.asarray(prior["mean"]), np.asarray(prior["cov"])
            )
            train_step_fn = make_train_step_multi(
                apply_fn=None, wph_features_fn=wph_features_fn,
                mu_prior=jnp.asarray(prior["mean"], jnp.float32), whitener=whitener,
                z_grid=z_grid, quantile_grid=quantile_grid, y0=y0, n_channels=c,
                lambda_wph=args.lambda_wph, wph_t_min=args.wph_t_min,
                lambda_warmup_steps=warmup_steps,
            )
        else:
            if not args.wph_targets.exists():
                raise FileNotFoundError(
                    f"--wph-mode persample needs {args.wph_targets}; "
                    "run precompute_wph_targets_cib.py first."
                )
            tgt = np.load(args.wph_targets)
            F_targets_np = np.asarray(tgt["F_targets"], dtype=np.float32)
            inv_std_np = np.asarray(tgt["inv_std"], dtype=np.float32)
            if F_targets_np.shape[0] != n_samples:
                raise ValueError(
                    f"WPH targets ({F_targets_np.shape[0]}) != x_train ({n_samples})"
                )
            if "n_features" in prior.files and F_targets_np.shape[1] != int(prior["n_features"]):
                raise ValueError(
                    f"targets n_features {F_targets_np.shape[1]} != prior "
                    f"{int(prior['n_features'])} — rebuild against same prior."
                )
            print(f"  targets: shape={F_targets_np.shape}, "
                  f"inv_std median={float(np.median(inv_std_np)):.3e}")
            train_step_fn = make_train_step_persample_multi(
                apply_fn=None, wph_features_fn=wph_features_fn,
                inv_std_per_feature=jnp.asarray(inv_std_np, jnp.float32),
                z_grid=z_grid, quantile_grid=quantile_grid, y0=y0, n_channels=c,
                lambda_wph=args.lambda_wph, wph_t_min=args.wph_t_min,
                lambda_warmup_steps=warmup_steps,
            )
    else:
        print("WPH prior not provided; FM-only (L_WPH = 0).")

    rng_key = jax.random.PRNGKey(args.seed)
    model = UNet(channels=tuple(args.channels), t_dim=args.t_dim,
                 bottleneck_blocks=args.bottleneck_blocks, out_channels=c)
    rng_key, init_key = jax.random.split(rng_key)
    params = model.init(init_key, jnp.zeros((1, h, w, c), jnp.float32),
                        jnp.zeros((1,), jnp.float32))
    n_params = sum(int(p.size) for p in jax.tree_util.tree_leaves(params))
    print(f"model: UNet channels={args.channels}, out_channels={c}, ~{n_params:,} params")

    if not use_wph:
        train_step_fn = make_fm_only_train_step(model.apply)

    total_steps = max(1, (n_samples // args.batch_size) * args.epochs)
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=args.lr,
        warmup_steps=min(500, total_steps // 10),
        decay_steps=total_steps, end_value=0.0,
    )
    state = FlowMatchingTrainState.create(
        apply_fn=model.apply, params=params, tx=optax.adam(schedule)
    )

    config_dict = {
        "channels": list(args.channels), "t_dim": args.t_dim,
        "bottleneck_blocks": args.bottleneck_blocks, "out_channels": c,
        "patch_hw": [h, w], "bands": bands,
    }

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
            running, running_fm, running_wph = 0.0, 0.0, 0.0
            for b in range(n_batches):
                idx = perm[b * args.batch_size : (b + 1) * args.batch_size]
                batch = jnp.asarray(x_train[idx])
                if use_wph and args.wph_mode == "persample":
                    Ft = jnp.asarray(F_targets_np[idx])
                    state, loss, l_fm, l_wph, step_key = train_step_fn(
                        state, batch, Ft, step_key
                    )
                    running_fm += float(l_fm)
                    running_wph += float(l_wph)
                elif use_wph:
                    state, loss, l_fm, l_wph, step_key = train_step_fn(
                        state, batch, step_key
                    )
                    running_fm += float(l_fm)
                    running_wph += float(l_wph)
                else:
                    state, loss, step_key = train_step_fn(state, batch, step_key)
                    running_fm += float(loss)
                running += float(loss)

            last_loss = running / max(n_batches, 1)
            if epoch % args.log_every == 0 or epoch == args.epochs:
                elapsed = (time.time() - t0) / 60.0
                if use_wph:
                    print(f"epoch {epoch:>3d}/{args.epochs}  loss={last_loss:.4f}  "
                          f"L_FM={running_fm / max(n_batches,1):.4f}  "
                          f"L_WPH={running_wph / max(n_batches,1):.4f}  "
                          f"elapsed={elapsed:.1f} min")
                else:
                    print(f"epoch {epoch:>3d}/{args.epochs}  "
                          f"L_FM={running_fm / max(n_batches,1):.4f}  "
                          f"elapsed={elapsed:.1f} min")

            if args.save_every and epoch % args.save_every == 0 and epoch != args.epochs:
                mid = args.out.with_name(f"{args.out.stem}.epoch{epoch}{args.out.suffix}")
                _save_checkpoint(mid, params=state.params, config=config_dict,
                                 quantile_grid=quantile_grid, z_grid=z_grid, y0=y0,
                                 bands=bands, patch_size_deg=patch_size_deg,
                                 pixel_size_arcmin=pixel_size_arcmin,
                                 epoch=epoch, last_loss=last_loss)
    except KeyboardInterrupt:
        interrupted = True
        print(f"\ninterrupted at epoch {epoch} — saving ...")

    _save_checkpoint(args.out, params=state.params, config=config_dict,
                     quantile_grid=quantile_grid, z_grid=z_grid, y0=y0, bands=bands,
                     patch_size_deg=patch_size_deg, pixel_size_arcmin=pixel_size_arcmin,
                     epoch=epoch, last_loss=last_loss)
    print(f"{'interrupted' if interrupted else 'completed'}: saved -> {args.out} "
          f"(epoch {epoch}, loss {last_loss:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
