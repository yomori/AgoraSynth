"""Wavelet filter bank for the WPH operator.

Ports ``pywph.filters.BumpSteerableWavelet`` and ``pywph.filters.GaussianFilter``
into pure NumPy. Filters are constructed once and stored in Fourier space; the
forward pass in :mod:`diffusiontsz.wph` consumes them as JAX arrays.

Conventions match pywph 1.1.3 exactly so coefficients can be cross-validated.
The filter parameters used by the default model are:

    k0     = 0.85 * pi
    sigma0 = 1 / (0.496 * 2**(-0.55) * k0)

These are hardcoded inside ``WPHOp.load_filters`` in pywph; we surface them as
keyword arguments here.
"""

from __future__ import annotations

import math
import multiprocessing as mp
from dataclasses import dataclass
from functools import partial
from itertools import product

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating]


PYWPH_K0 = 0.85 * np.pi
PYWPH_SIGMA0 = 1.0 / (0.496 * (2.0 ** -0.55) * PYWPH_K0)


def _periodization(filter_f: np.ndarray, M: int, N: int) -> np.ndarray:
    """Anti-aliased periodization of a filter built on a 2x grid.

    Direct port of ``BumpSteerableWavelet._periodization`` from pywph 1.1.3.
    """

    filter_f_shifted = np.fft.fftshift(filter_f)

    filter_f_shifted[
        M - M // 2 : M, N - N // 2 : N + (N + 1) // 2,
    ] += filter_f_shifted[
        M + (M + 1) // 2 :, N - N // 2 : N + (N + 1) // 2,
    ]
    filter_f_shifted[
        M : M + (M + 1) // 2, N - N // 2 : N + (N + 1) // 2,
    ] += filter_f_shifted[
        : M - M // 2, N - N // 2 : N + (N + 1) // 2,
    ]

    filter_f_shifted[
        M - M // 2 : M + (M + 1) // 2, N - N // 2 : N,
    ] += filter_f_shifted[
        M - M // 2 : M + (M + 1) // 2, N + (N + 1) // 2 :,
    ]
    filter_f_shifted[
        M - M // 2 : M + (M + 1) // 2, N : N + (N + 1) // 2,
    ] += filter_f_shifted[
        M - M // 2 : M + (M + 1) // 2, : N - N // 2,
    ]

    filter_f_shifted[
        M : M + (M + 1) // 2, N : N + (N + 1) // 2,
    ] += filter_f_shifted[: M - M // 2, : N - N // 2]
    filter_f_shifted[
        M - M // 2 : M, N - N // 2 : N,
    ] += filter_f_shifted[M + (M + 1) // 2 :, N + (N + 1) // 2 :]

    filter_f_shifted[
        M : M + (M + 1) // 2, N - N // 2 : N,
    ] += filter_f_shifted[: M - M // 2, N + (N + 1) // 2 :]
    filter_f_shifted[
        M - M // 2 : M, N : N + (N + 1) // 2,
    ] += filter_f_shifted[M + (M + 1) // 2 :, : N - N // 2]

    return np.fft.ifftshift(
        filter_f_shifted[
            M - M // 2 : M + (M + 1) // 2, N - N // 2 : N + (N + 1) // 2,
        ]
    )


def bump_steerable_filter(
    M: int,
    N: int,
    j: int,
    theta: float,
    *,
    k0: float = PYWPH_K0,
    L: int = 4,
    n: int = 0,
    alpha: float = 0.0,
) -> np.ndarray:
    """Bump-steerable bandpass wavelet in Fourier space (real if n==0).

    Returns a complex array of shape ``(M, N)`` (real-valued when no
    translation is applied, i.e. ``n == 0``).
    """

    c = (
        2 ** (L - 1)
        / 1.29
        * math.factorial(L - 1)
        / np.sqrt(L * math.factorial(2 * (L - 1)))
    )

    sigma = 2.0 ** j
    xi = k0 / sigma

    kx = 2 * 2 * np.pi * np.fft.fftfreq(2 * N)
    ky = 2 * 2 * np.pi * np.fft.fftfreq(2 * M)
    k2d_x, k2d_y = np.meshgrid(kx, ky)
    k2d = k2d_x + 1j * k2d_y
    k2d_mod = np.absolute(k2d)
    k2d_angle = np.angle(k2d)

    car_argk_0_pi2 = np.logical_or(
        (k2d_angle - theta) % (2 * np.pi) <= np.pi / 2,
        (k2d_angle - theta) % (2 * np.pi) >= 3 * np.pi / 2,
    ).astype(float)
    car_k_0_2xi = np.logical_and(k2d_mod > 0.0, k2d_mod < 2 * xi).astype(float)
    exp_var = (
        -((k2d_mod - xi) ** 2) * car_k_0_2xi
        / (xi ** 2 * car_k_0_2xi - (k2d_mod - xi) ** 2)
    )
    psi_f = (
        c
        * np.exp(exp_var)
        * car_k_0_2xi
        * np.cos(k2d_angle - theta) ** (L - 1)
        * car_argk_0_pi2
    )

    if n != 0:
        nx = n * np.cos(theta - alpha)
        ny = n * np.sin(theta - alpha)
        psi_f = psi_f * np.exp(-1j * sigma * (k2d_x * nx + k2d_y * ny))

    psi_f = _periodization(psi_f, M, N)

    if n == 0:
        return psi_f.real.astype(np.float64)
    return psi_f.astype(np.complex128)


def gaussian_lowpass_filter(
    M: int,
    N: int,
    j: int,
    *,
    sigma0: float = PYWPH_SIGMA0,
    theta: float = 0.0,
    gamma: float = 1.0,
) -> np.ndarray:
    """Real Gaussian low-pass envelope in *real* space.

    Direct port of ``GaussianFilter.build`` (pywph 1.1.3) without the FFT
    -- we return the real-space envelope, since the WPH operator transforms
    it to Fourier space jointly with the input. Output dtype: float64.

    The envelope is anchored at the corner ``(0, 0)`` and tiled across the
    eight neighbouring images to make it periodic on the ``(M, N)`` torus.
    """

    sigma = sigma0 * 2.0 ** j

    R = np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ]
    )
    Rinv = np.array(
        [
            [np.cos(theta), np.sin(theta)],
            [-np.sin(theta), np.cos(theta)],
        ]
    )
    D = np.array([[1, 0], [0, gamma ** 2]])
    curv = R @ D @ Rinv / (2 * sigma ** 2)

    data = np.zeros((M, N), dtype=np.float64)
    for ex in (-2, -1, 0, 1):
        for ey in (-2, -1, 0, 1):
            xx, yy = np.mgrid[
                ex * M : M + ex * M, ey * N : N + ey * N,
            ]
            arg = -(
                curv[0, 0] * xx ** 2
                + (curv[0, 1] + curv[1, 0]) * xx * yy
                + curv[1, 1] * yy ** 2
            )
            data += np.exp(arg)

    norm = 2 * np.pi * sigma ** 2 / gamma
    return data / norm


def gaussian_lowpass_filter_fourier(
    M: int, N: int, j: int, *, sigma0: float = PYWPH_SIGMA0,
) -> np.ndarray:
    """Gaussian low-pass envelope in Fourier space (real).

    Convenience wrapper returning ``np.fft.fft2(real_envelope).real`` to
    match what pywph stores internally when ``fourier=True``.
    """

    real = gaussian_lowpass_filter(M, N, j, sigma0=sigma0)
    return np.fft.fft2(real).real.astype(np.float64)


@dataclass(frozen=True)
class FilterBankConfig:
    """Hyper-parameters of the WPH filter bank.

    Mirrors the pywph defaults exactly. ``M`` and ``N`` are image height /
    width, ``J`` is the number of dyadic scales, ``L`` is the number of
    orientations on ``[0, pi)``, ``j_min`` is the lowest scale included.
    ``sm_j_list`` is the list of dyadic scales used by the scaling
    moments ``L``; following pywph, ``j = -1`` means "no smoothing"
    (the unsmoothed image, with mean removed in the WPH operator).
    """

    M: int
    N: int
    J: int
    L: int = 4
    j_min: int = 0
    k0: float = PYWPH_K0
    sigma0: float = PYWPH_SIGMA0
    sm_j_list: tuple[int, ...] = (-1, 0, 1, 2)


def _build_psi_chunk(
    work_list: np.ndarray, M: int, N: int, k0: float, L: int,
) -> np.ndarray:
    """Build a contiguous slice of the bandpass bank in a worker process."""

    out = np.zeros((work_list.shape[0], M, N), dtype=np.float64)
    for i in range(work_list.shape[0]):
        j = float(work_list[i, 0])
        t = work_list[i, 1]
        out[i] = bump_steerable_filter(
            M, N, j, t * np.pi / L, k0=k0, L=L,
        )
    return out


def build_psi_bank(
    cfg: FilterBankConfig, *, parallel: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the bandpass bank ``psi_f`` and its index table.

    Returns
    -------
    psi_f : (n_psi, M, N) float64
        Fourier-space wavelets, real-valued (no translation built-in --
        translations are applied analytically in the operator).
    psi_indices : (n_psi, 2) int
        ``(j, t)`` for each row of ``psi_f``.
    """

    indices = np.array(
        list(product(range(cfg.j_min, cfg.J), range(cfg.L))),
        dtype=int,
    )
    n_psi = indices.shape[0]
    if not parallel or n_psi <= 1:
        psi = np.zeros((n_psi, cfg.M, cfg.N), dtype=np.float64)
        for i, (j, t) in enumerate(indices):
            psi[i] = bump_steerable_filter(
                cfg.M, cfg.N, float(j), t * np.pi / cfg.L,
                k0=cfg.k0, L=cfg.L,
            )
        return psi, indices

    nb_proc = min(mp.cpu_count(), n_psi)
    work = np.array_split(indices, nb_proc)
    fn = partial(_build_psi_chunk, M=cfg.M, N=cfg.N, k0=cfg.k0, L=cfg.L)
    with mp.get_context("fork").Pool(processes=nb_proc) as pool:
        results = pool.map(fn, work)

    psi = np.zeros((n_psi, cfg.M, cfg.N), dtype=np.float64)
    cnt = 0
    for r in results:
        psi[cnt : cnt + r.shape[0]] = r
        cnt += r.shape[0]
    return psi, indices


def build_phi_bank(
    cfg: FilterBankConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the low-pass bank ``phi_f`` for the requested ``sm_j_list``.

    Mirrors ``pywph.WPHOp.load_filters``: builds a Gaussian for *every* j
    in ``sm_j_list``, including ``j == -1`` (sigma = sigma0 / 2, sub-pixel
    -- functionally near-identity but not exactly).

    Returns
    -------
    phi_f : (n_phi, M, N) float64
        Fourier-space low-pass envelopes (real-valued).
    phi_j : (n_phi,) int
        ``j`` value associated with each row of ``phi_f``, sorted.
    """

    phi_j = np.array(sorted(set(cfg.sm_j_list)), dtype=int)
    n_phi = phi_j.shape[0]
    phi = np.zeros((n_phi, cfg.M, cfg.N), dtype=np.float64)
    for i, j in enumerate(phi_j):
        phi[i] = gaussian_lowpass_filter_fourier(
            cfg.M, cfg.N, int(j), sigma0=cfg.sigma0,
        )
    return phi, phi_j
