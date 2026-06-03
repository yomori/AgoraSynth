"""Plot a full-footprint y-map FITS from synthesize_full_map[_coherent].py.

Default: log10 stretch, inferno colormap, sky coordinates from the FITS WCS.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fits_path", type=Path, help="Path to the y-map FITS.")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output PNG (default: same stem as input + .png).")
    parser.add_argument("--lo-pct", type=float, default=2.0,
                        help="Lower color-scale percentile.")
    parser.add_argument("--hi-pct", type=float, default=99.5,
                        help="Upper color-scale percentile.")
    parser.add_argument("--floor", type=float, default=1e-9,
                        help="Floor before log10.")
    parser.add_argument("--cmap", type=str, default="inferno")
    parser.add_argument("--figsize", type=float, nargs=2, default=(14.0, 5.0))
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args(argv)

    import matplotlib.pyplot as plt
    from astropy.io import fits
    from astropy.wcs import WCS

    out = args.out if args.out is not None else args.fits_path.with_suffix(".png")

    with fits.open(args.fits_path) as hdul:
        primary = hdul[0]
        data = np.asarray(primary.data, dtype=np.float64)
        wcs = WCS(primary.header)

    if data.ndim != 2:
        raise ValueError(f"expected 2D image, got shape {data.shape}")

    log_y = np.log10(np.maximum(data, args.floor))
    vmin = float(np.percentile(log_y, args.lo_pct))
    vmax = float(np.percentile(log_y, args.hi_pct))
    print(
        f"  data shape={data.shape}, y range [{data.min():.3e}, {data.max():.3e}]; "
        f"log10 color [{vmin:.2f}, {vmax:.2f}]"
    )

    fig = plt.figure(figsize=tuple(args.figsize))
    ax = fig.add_subplot(1, 1, 1, projection=wcs)
    im = ax.imshow(log_y, cmap=args.cmap, origin="lower", vmin=vmin, vmax=vmax)

    ax.set_xlabel("RA")
    ax.set_ylabel("Dec")
    ax.coords.grid(True, color="white", alpha=0.25, linestyle=":")
    ax.coords["ra"].set_format_unit("deg")
    ax.coords["dec"].set_format_unit("deg")

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label(r"$\log_{10}\,y$")

    title = f"{args.fits_path.name}"
    if "METHOD" in primary.header:
        title += f"  ({primary.header['METHOD']})"
    ax.set_title(title)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
