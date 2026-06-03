"""Plot CIB truth vs. model realizations across 95/150/220 GHz.

Grid layout: one row per band, columns = [truth, realization 1..N]. Each row
(band) shares a color scale. Realizations are independent unconditional draws
from the trained model, so they are NOT the same field as the truth patch —
the point is that they look statistically like it (morphology, dynamic range,
point-source population) and stay mutually consistent across bands.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


def _to_display(img, transform, floor):
    if transform == "log":
        return np.log10(np.maximum(img, floor))
    if transform == "asinh":
        return np.arcsinh(img / floor)
    return img


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=Path("samples/fm_cib.npz"),
                        help="sample_fm_cib.py output (has y_samples, bands).")
    parser.add_argument("--data", type=Path, default=Path("data/train_cib.npz"),
                        help="Dataset for truth panels (has cib_patches).")
    parser.add_argument("--n-real", type=int, default=5)
    parser.add_argument("--truth-idx", type=int, default=-1,
                        help="Truth patch index; -1 picks a bright/structured one.")
    parser.add_argument("--transform", choices=("log", "asinh", "linear"), default="log")
    parser.add_argument("--cmap", type=str, default="inferno")
    parser.add_argument("--out", type=Path, default=Path("samples/cib_realizations.png"))
    args = parser.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sd = np.load(args.samples, allow_pickle=True)
    y_samples = np.asarray(sd["y_samples"])                     # (N, H, W, C)
    bands = [str(b) for b in sd["bands"]] if "bands" in sd.files else \
        [str(i) for i in range(y_samples.shape[-1])]
    n, h, w, c = y_samples.shape
    n_real = min(args.n_real, n)

    td = np.load(args.data, allow_pickle=True)
    cib = np.asarray(td["cib_patches"])                          # (M, C, H, W)
    truth = np.transpose(cib, (0, 2, 3, 1))                      # (M, H, W, C)

    # Pick a structured truth patch (brightest 150 GHz, the middle band) unless given.
    if args.truth_idx >= 0:
        ti = args.truth_idx
    else:
        mid = min(1, c - 1)
        ti = int(np.argmax(truth[..., mid].max(axis=(1, 2))))
    truth_patch = truth[ti]                                      # (H, W, C)

    # Per-band display floor: 1% of the positive median across truth+samples.
    floors = np.empty(c)
    for ch in range(c):
        vals = np.concatenate([truth_patch[..., ch].ravel(),
                               y_samples[:n_real, ..., ch].ravel()])
        pos = vals[vals > 0]
        floors[ch] = 0.01 * float(np.median(pos)) if pos.size else 1e-6

    ncols = 1 + n_real
    fig, axes = plt.subplots(c, ncols, figsize=(2.5 * ncols, 2.7 * c), squeeze=False)
    for ch in range(c):
        disp_truth = _to_display(truth_patch[..., ch], args.transform, floors[ch])
        disp_real = [_to_display(y_samples[k, ..., ch], args.transform, floors[ch])
                     for k in range(n_real)]
        stack = np.concatenate([disp_truth.ravel()] + [d.ravel() for d in disp_real])
        vmin, vmax = np.percentile(stack, [1, 99])
        panels = [disp_truth] + disp_real
        titles = ["truth"] + [f"real #{k+1}" for k in range(n_real)]
        im = None
        for col in range(ncols):
            ax = axes[ch, col]
            im = ax.imshow(panels[col], cmap=args.cmap, origin="lower",
                           vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if ch == 0:
                ax.set_title(titles[col], fontsize=10)
            if col == 0:
                ax.set_ylabel(f"{bands[ch]} GHz", fontsize=11)
        cax = fig.add_axes((0.92, axes[ch, -1].get_position().y0,
                            0.012, axes[ch, -1].get_position().height))
        lbl = {"log": r"$\log_{10} I$", "asinh": "asinh(I/floor)",
               "linear": "I"}[args.transform]
        fig.colorbar(im, cax=cax, label=lbl)

    fig.suptitle(f"Agora CIB: truth (patch #{ti}) vs. {n_real} model realizations",
                 fontsize=13)
    fig.subplots_adjust(left=0.06, right=0.9, top=0.93, bottom=0.03,
                        wspace=0.05, hspace=0.12)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    plt.close(fig)
    print(f"saved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
