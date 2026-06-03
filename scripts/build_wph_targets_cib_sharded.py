"""Robust sharded build of the per-sample WPH targets.

The single-process precompute_wph_targets_cib.py dies deterministically after
~470 batches (a per-process JAX/XLA resource buildup triggers a slow recompile
that gets OOM-killed, independent of GPU size). This driver sidesteps that by
running each shard in a FRESH subprocess (clean JAX/CUDA state), then
concatenates the raw shard features and computes the global per-feature
inv_std exactly as the single-process path would.

    python scripts/build_wph_targets_cib_sharded.py \
        --data data/train_cib.npz --wph-prior runs/wph_prior_cib.npz \
        --out data/wph_targets_cib.npz --shard 2000

Resumable: existing shard .npy files are reused, so a killed run continues.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent


def _npz_nrows(path: Path, key: str) -> int:
    """Read array shape[0] from an .npz member without loading it."""
    with zipfile.ZipFile(path) as z:
        with z.open(f"{key}.npy") as f:
            ver = np.lib.format.read_magic(f)
            shp, _, _ = np.lib.format._read_array_header(f, ver)
    return int(shp[0])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("data/train_cib.npz"))
    ap.add_argument("--wph-prior", type=Path, default=Path("runs/wph_prior_cib.npz"))
    ap.add_argument("--out", type=Path, default=Path("data/wph_targets_cib.npz"))
    ap.add_argument("--shard", type=int, default=2000,
                    help="Patches per fresh-process shard (<<~3700 failure point).")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--std-floor-rel", type=float, default=1e-3)
    ap.add_argument("--tmpdir", type=Path, default=Path("runs/targets_shards"))
    args = ap.parse_args(argv)

    n = _npz_nrows(args.data, "x_train")
    prior = np.load(args.wph_prior)
    c = int(prior["n_channels"])
    n_feat_expected = int(prior["n_features"]) if "n_features" in prior.files else None
    args.tmpdir.mkdir(parents=True, exist_ok=True)
    starts = list(range(0, n, args.shard))
    print(f"{n} patches -> {len(starts)} shards of {args.shard} "
          f"(fresh process each); tmp={args.tmpdir}")

    env = {**os.environ, "XLA_PYTHON_CLIENT_PREALLOCATE": "false"}
    shard_files = []
    for s in starts:
        e = min(s + args.shard, n)
        raw = args.tmpdir / f"shard_{s:06d}_{e:06d}.npy"
        shard_files.append(raw)
        if raw.exists():
            print(f"  [shard {s}:{e}] reuse {raw.name}")
            continue
        print(f"  [shard {s}:{e}] computing in fresh subprocess ...", flush=True)
        subprocess.run(
            [sys.executable, str(REPO / "scripts/precompute_wph_targets_cib.py"),
             "--data", str(args.data), "--wph-prior", str(args.wph_prior),
             "--start", str(s), "--stop", str(e),
             "--batch-size", str(args.batch_size), "--raw-out", str(raw)],
            check=True, cwd=str(REPO), env=env,
        )

    print("concatenating shards ...")
    F = np.concatenate([np.load(f) for f in shard_files], axis=0).astype(np.float32)
    if F.shape[0] != n:
        raise RuntimeError(f"concatenated {F.shape[0]} rows != {n} patches")
    if n_feat_expected is not None and F.shape[1] != n_feat_expected:
        raise RuntimeError(f"n_features {F.shape[1]} != prior {n_feat_expected}")

    std = F.std(axis=0).astype(np.float64)
    med = float(np.median(std))
    floor = args.std_floor_rel * med if med > 0 else 1e-12
    n_floored = int((std < floor).sum())
    std = np.maximum(std, floor)
    inv_std = (1.0 / std).astype(np.float32)
    if n_floored:
        print(f"  floored {n_floored}/{F.shape[1]} features with std < {floor:.3e}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        F_targets=F,
        std_per_feature=std.astype(np.float32),
        inv_std=inv_std,
        n_features=np.int64(F.shape[1]),
        n_channels=np.int64(c),
    )
    print(f"saved {F.shape} targets -> {args.out}  (inv_std median "
          f"{float(np.median(inv_std)):.3e})")
    # tidy shard scratch
    for f in shard_files:
        try:
            f.unlink()
        except OSError:
            pass
    try:
        args.tmpdir.rmdir()
    except OSError:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
