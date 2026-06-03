"""HEALPix -> ZEA patch extraction + per-pixel rank/quantile transform.

Mirrors the data pipeline used by AgoraScore but keeps the inverse
transform JAX-callable so it can be applied inside the training graph
(needed to feed the predicted clean sample to the WPH operator in
physical-y space).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Sphere -> ZEA tangent-plane projection
# ---------------------------------------------------------------------------


def random_sphere_directions(
    n: int, seed: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Sample ``n`` directions uniformly on the sphere. Returns (ra_deg, dec_deg)."""
    rng = np.random.default_rng(seed)
    ra = rng.uniform(0.0, 360.0, size=n)
    dec = np.rad2deg(np.arcsin(rng.uniform(-1.0, 1.0, size=n)))
    return ra, dec


def _zea_wcs(ra_deg: float, dec_deg: float, n_pix: int, pixel_size_arcmin: float):
    from astropy.wcs import WCS

    w = WCS(naxis=2)
    center = (n_pix + 1) / 2.0
    w.wcs.crpix = [center, center]
    w.wcs.cdelt = [-pixel_size_arcmin / 60.0, pixel_size_arcmin / 60.0]
    w.wcs.crval = [float(ra_deg), float(dec_deg)]
    w.wcs.ctype = ["RA---ZEA", "DEC--ZEA"]
    return w


def project_healpix_to_zea(
    healpix_map: NDArray,
    ra_center_deg: float,
    dec_center_deg: float,
    n_pix: int,
    pixel_size_arcmin: float,
    nest: bool = False,
) -> NDArray[np.float32]:
    """Project a HEALPix map onto a ZEA tangent-plane patch via bilinear interpolation."""
    import healpy as hp

    w = _zea_wcs(ra_center_deg, dec_center_deg, n_pix, pixel_size_arcmin)
    ii, jj = np.meshgrid(np.arange(n_pix), np.arange(n_pix), indexing="xy")
    ra, dec = w.wcs_pix2world(ii, jj, 0)
    theta = np.deg2rad(90.0 - dec)
    phi = np.deg2rad(ra)
    return hp.get_interp_val(healpix_map, theta, phi, nest=nest).astype(np.float32)


def extract_patches(
    healpix_map: NDArray,
    ra_centers_deg: NDArray,
    dec_centers_deg: NDArray,
    n_pix: int,
    pixel_size_arcmin: float,
    nest: bool = False,
    progress: bool = True,
) -> NDArray[np.float32]:
    """Project ``len(ra_centers)`` ZEA patches from a HEALPix map."""
    n = len(ra_centers_deg)
    out = np.empty((n, n_pix, n_pix), dtype=np.float32)
    iterator: object = range(n)
    if progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(range(n), total=n, desc="extracting patches")
        except ImportError:
            pass
    for k in iterator:  # type: ignore[assignment]
        out[k] = project_healpix_to_zea(
            healpix_map,
            float(ra_centers_deg[k]),
            float(dec_centers_deg[k]),
            n_pix=n_pix,
            pixel_size_arcmin=pixel_size_arcmin,
            nest=nest,
        )
    return out


# ---------------------------------------------------------------------------
# Per-pixel rank/quantile transform: marginal -> N(0, 1)
# ---------------------------------------------------------------------------


def gaussianize_patches(
    patches: NDArray,
    y0: float = 1e-7,
    n_quantiles: int = 1024,
    z_max: float = 5.5,
) -> tuple[NDArray[np.float32], NDArray[np.float64], NDArray[np.float64]]:
    """Map each pixel of log(y + y0) through the empirical CDF then Phi^-1.

    The marginal of the returned ``x`` is exactly N(0, 1) by construction.
    Heavy tails (cluster cores at z=10+ in naive standardize) become z<=5.5.
    Inverted via :func:`gaussianized_to_physical` using the saved
    ``(z_grid, quantile_grid)`` pair.
    """
    from scipy.stats import norm

    log_y = np.log(np.maximum(patches.astype(np.float64), 1e-30) + y0)

    z_grid = np.linspace(-z_max, z_max, n_quantiles).astype(np.float64)
    p_grid = norm.cdf(z_grid)
    quantile_grid = np.quantile(log_y.ravel(), p_grid).astype(np.float64)
    if not np.all(np.diff(quantile_grid) >= 0):
        quantile_grid = np.sort(quantile_grid)

    x_flat = np.interp(log_y.ravel(), quantile_grid, z_grid)
    x = x_flat.reshape(patches.shape).astype(np.float32)
    return x, quantile_grid, z_grid


def gaussianized_to_physical(
    x,
    quantile_grid,
    z_grid,
    y0: float,
):
    """Invert :func:`gaussianize_patches`: x -> physical y. Numpy or JAX-aware.

    If ``x`` is a JAX array, returns a JAX array (uses ``jnp.interp``); if
    numpy or torch, returns a numpy array. The JAX path is what the
    flow-matching training graph uses to push the predicted clean sample
    into physical-y space for the WPH loss.
    """
    try:
        import jax.numpy as jnp

        if isinstance(x, jnp.ndarray):
            log_y = jnp.interp(
                x,
                jnp.asarray(z_grid, dtype=x.dtype),
                jnp.asarray(quantile_grid, dtype=x.dtype),
            )
            return jnp.exp(log_y) - y0
    except ImportError:
        pass

    try:
        import torch

        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
    except ImportError:
        pass

    log_y = np.interp(np.asarray(x), z_grid, quantile_grid)
    return np.exp(log_y) - y0


# ---------------------------------------------------------------------------
# Multi-channel variants (CIB: co-located 95/150/220 GHz patches)
# ---------------------------------------------------------------------------


def extract_patches_multi(
    healpix_maps: list,
    ra_centers_deg: NDArray,
    dec_centers_deg: NDArray,
    n_pix: int,
    pixel_size_arcmin: float,
    nest: bool = False,
    progress: bool = True,
) -> NDArray[np.float32]:
    """Project ``C`` co-located ZEA patch stacks from ``C`` HEALPix maps.

    Every channel is sampled at the *same* sky directions, so the returned
    ``(N, C, n_pix, n_pix)`` array preserves per-pixel cross-channel
    correspondence (essential for CIB: the 95/150/220 GHz maps trace the
    same galaxies).
    """
    n = len(ra_centers_deg)
    c = len(healpix_maps)
    out = np.empty((n, c, n_pix, n_pix), dtype=np.float32)
    iterator: object = range(n)
    if progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(range(n), total=n, desc="extracting patches")
        except ImportError:
            pass
    for k in iterator:  # type: ignore[assignment]
        ra = float(ra_centers_deg[k])
        dec = float(dec_centers_deg[k])
        for ch in range(c):
            out[k, ch] = project_healpix_to_zea(
                healpix_maps[ch], ra, dec,
                n_pix=n_pix, pixel_size_arcmin=pixel_size_arcmin, nest=nest,
            )
    return out


def gaussianize_patches_multi(
    patches: NDArray,
    y0=1e-7,
    n_quantiles: int = 1024,
    z_max: float = 5.5,
) -> tuple[NDArray[np.float32], NDArray[np.float64], NDArray[np.float64]]:
    """Per-channel rank/quantile transform of ``log(I + y0)`` -> N(0, 1).

    Same construction as :func:`gaussianize_patches` but applied independently
    to each channel of a ``(N, C, H, W)`` stack, so every band's marginal is
    exactly N(0, 1). ``y0`` may be a scalar (shared) or a length-``C`` array
    (per-band floor). The single ``z_grid`` is shared across channels; each
    channel gets its own ``quantile_grid`` row.

    Returns
    -------
    x : (N, C, H, W) float32
        Gaussianized stack.
    quantile_grid : (C, n_quantiles) float64
        Per-channel log-intensity quantiles.
    z_grid : (n_quantiles,) float64
        Shared standard-normal grid.
    """
    from scipy.stats import norm

    arr = np.asarray(patches, dtype=np.float64)
    if arr.ndim != 4:
        raise ValueError(f"expected (N, C, H, W), got shape {arr.shape}")
    n, c, h, w = arr.shape
    y0_arr = np.broadcast_to(
        np.atleast_1d(np.asarray(y0, dtype=np.float64)), (c,)
    ).astype(np.float64)

    z_grid = np.linspace(-z_max, z_max, n_quantiles).astype(np.float64)
    p_grid = norm.cdf(z_grid)

    x = np.empty((n, c, h, w), dtype=np.float32)
    quantile_grid = np.empty((c, n_quantiles), dtype=np.float64)
    for ch in range(c):
        log_y = np.log(np.maximum(arr[:, ch], 1e-30) + y0_arr[ch])
        qg = np.quantile(log_y.ravel(), p_grid).astype(np.float64)
        if not np.all(np.diff(qg) >= 0):
            qg = np.sort(qg)
        quantile_grid[ch] = qg
        x[:, ch] = (
            np.interp(log_y.ravel(), qg, z_grid)
            .reshape(n, h, w)
            .astype(np.float32)
        )
    return x, quantile_grid, z_grid


def gaussianized_to_physical_multi(x, quantile_grid, z_grid, y0):
    """Invert :func:`gaussianize_patches_multi`, channel-LAST.

    ``x`` has shape ``(..., C)`` (NHWC convention used by the model output);
    ``quantile_grid`` is ``(C, n_quantiles)``; ``y0`` is a scalar or ``(C,)``.
    JAX-aware: returns a JAX array if ``x`` is a JAX array (used by the
    training graph and the sampler), otherwise numpy. ``C`` is read from the
    last axis and the per-channel loop is unrolled, so this stays jit-safe
    when ``C`` is a concrete dimension.
    """
    c = int(x.shape[-1])
    y0_arr = np.broadcast_to(
        np.atleast_1d(np.asarray(y0, dtype=np.float64)), (c,)
    )
    qg = np.asarray(quantile_grid, dtype=np.float64)
    zg = np.asarray(z_grid, dtype=np.float64)

    try:
        import jax.numpy as jnp

        if isinstance(x, jnp.ndarray):
            zg_j = jnp.asarray(zg, dtype=x.dtype)
            qg_j = jnp.asarray(qg, dtype=x.dtype)
            chans = []
            for ch in range(c):
                log_y = jnp.interp(x[..., ch], zg_j, qg_j[ch])
                chans.append(jnp.exp(log_y) - jnp.asarray(y0_arr[ch], x.dtype))
            return jnp.stack(chans, axis=-1)
    except ImportError:
        pass

    try:
        import torch

        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
    except ImportError:
        pass

    x_np = np.asarray(x)
    chans = []
    for ch in range(c):
        log_y = np.interp(x_np[..., ch], zg, qg[ch])
        chans.append(np.exp(log_y) - y0_arr[ch])
    return np.stack(chans, axis=-1)
