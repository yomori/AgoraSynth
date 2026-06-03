"""Wavelet Phase Harmonic (WPH) statistics in JAX.

A re-implementation of the subset of ``pywph`` (Régaldo-Saint Blancard et
al. 2021; see https://github.com/bregaldo/pywph) needed for non-Gaussian
field synthesis on the Compton-y map.

The default model matches pywph's defaults:

    classes  = ["S11", "S00", "S01", "C01", "Cphase", "Cphase_inv", "L"]
    sm_p_list = [(0, 0), (0, 1), (1, 1)]
    tau_grid  = "exp"  (so tau_x = round(2^n cos(theta - alpha)), n in {1..dn})

Parameters whose values are *not* meant to vary between runs are exposed as
``WPHConfig`` fields: M, N, J, L, j_min, dn, A, dj, dl, sm_j_list, classes.

What is *not* implemented here (would require pywph defaults):

- Complex-valued inputs (``cplx=True``) and the corresponding pseudo-moments.
  Our Compton-y field is real, so this is fine.
- Cross-WPH (apply to a pair of fields).
- Non-periodic-boundary padding (``pbc=False``). The synthesis loop here
  uses a cosine window inside the loss instead, which serves the same
  purpose without changing the operator signature.

The forward pass is jit-friendly and autograd-friendly. It returns a
complex array; a helper ``to_real_features`` produces the real-valued
feature vector used to build the Gaussian prior.
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from numpy.typing import NDArray

from .wph_filters import (
    PYWPH_K0,
    PYWPH_SIGMA0,
    FilterBankConfig,
    build_phi_bank,
    build_psi_bank,
)

FloatArray = NDArray[np.floating]
IntArray = NDArray[np.integer]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_CLASSES: tuple[str, ...] = (
    "S11", "S00", "S01", "C01", "Cphase", "Cphase_inv", "L",
)
DEFAULT_SM_P_LIST: tuple[tuple[int, int], ...] = ((0, 0), (0, 1), (1, 1))


@dataclass(frozen=True)
class WPHConfig:
    """Hyper-parameters of the WPH operator (mirror of pywph defaults)."""

    M: int
    N: int
    J: int
    L: int = 4
    j_min: int = 0
    dn: int = 0                                          # translation radii
    A: int = 4                                           # translation angles
    dj: int | None = None
    dl: int | None = None
    sm_j_list: tuple[int, ...] = (-1, 0, 1, 2)
    sm_p_list: tuple[tuple[int, int], ...] = DEFAULT_SM_P_LIST
    classes: tuple[str, ...] = DEFAULT_CLASSES
    k0: float = PYWPH_K0
    sigma0: float = PYWPH_SIGMA0
    tau_grid: str = "exp"

    def filter_config(self) -> FilterBankConfig:
        return FilterBankConfig(
            M=self.M, N=self.N, J=self.J, L=self.L, j_min=self.j_min,
            k0=self.k0, sigma0=self.sigma0, sm_j_list=self.sm_j_list,
        )

    def to_dict(self) -> dict[str, Any]:
        d = {
            "M": self.M, "N": self.N, "J": self.J, "L": self.L,
            "j_min": self.j_min, "dn": self.dn, "A": self.A,
            "dj": self.dj, "dl": self.dl,
            "sm_j_list": list(self.sm_j_list),
            "sm_p_list": [list(t) for t in self.sm_p_list],
            "classes": list(self.classes),
            "k0": self.k0, "sigma0": self.sigma0, "tau_grid": self.tau_grid,
        }
        return d


# ---------------------------------------------------------------------------
# Moment table
# ---------------------------------------------------------------------------


def _tau_xy(t: int, n: int, a: int, *, L: int, A: int, tau_grid: str) -> tuple[int, int]:
    theta = t * np.pi / L
    alpha = a * np.pi / A
    if tau_grid == "exp":
        tx = int(np.rint((n != 0) * (2.0 ** n) * np.cos(theta - alpha)))
        ty = int(np.rint((n != 0) * (2.0 ** n) * np.sin(theta - alpha)))
    elif tau_grid == "legacy":
        # j is needed for legacy; not supported here -- exp is the default.
        msg = "tau_grid='legacy' not supported; use 'exp'"
        raise ValueError(msg)
    else:
        msg = f"unknown tau_grid {tau_grid!r}"
        raise ValueError(msg)
    return tx, ty


@dataclass
class MomentTable:
    """All Python-side index arrays needed by the forward pass.

    Attributes
    ----------
    wph_classes : list[str]
        Class label per row of ``wph_indices``.
    wph_indices : (n_wph, 9) int
        Raw [j1, t1, p1, j2, t2, p2, n, alpha, pseudo].
    cov_psi : (n_cov, 2) int
        Linear (j, t) -> psi-bank indices for (Z1, Z2) of each cov.
    cov_p : (n_cov, 2) int
        (p1, p2) of each cov.
    cov_class : (n_cov,) int
        Class id (0..len(classes)-1) of each cov, used to drive class-by-
        class processing.
    wph_id_cov : (n_wph,) int
        For each WPH moment, which cov it lives on.
    wph_tau : (n_wph, 2) int
        (ty, tx) shift to sample for each WPH moment.
    sm_indices : (n_sm, 4) int
        [j, p1, p2, pseudo] for each scaling moment.
    sm_phi_id : (n_sm,) int
        Index into the low-pass bank (-1 for the unsmoothed j=-1).
    """

    wph_classes: list[str]
    wph_indices: IntArray
    cov_psi: IntArray
    cov_p: IntArray
    cov_class: IntArray
    wph_id_cov: IntArray
    wph_tau: IntArray
    sm_indices: IntArray
    sm_phi_id: IntArray

    @property
    def n_wph(self) -> int:
        return int(self.wph_indices.shape[0])

    @property
    def n_cov(self) -> int:
        return int(self.cov_psi.shape[0])

    @property
    def n_sm(self) -> int:
        return int(self.sm_indices.shape[0])

    @property
    def n_total(self) -> int:
        return self.n_wph + self.n_sm


def _psi_lin_id(j: int, t: int, *, j_min: int, L: int) -> int:
    return (j - j_min) * L + t


def build_moment_table(cfg: WPHConfig) -> MomentTable:
    """Port of ``pywph.WPHOp.load_model`` for ``cplx=False``.

    Returns a :class:`MomentTable` containing every index array the JAX
    forward pass needs. No JAX calls here -- pure Python / numpy.
    """

    L, A, dn = cfg.L, cfg.A, cfg.dn
    dj = cfg.dj if cfg.dj is not None else cfg.J - cfg.j_min - 1
    dl = cfg.dl if cfg.dl is not None else L // 2
    tau_grid = cfg.tau_grid

    # Reorder classes to canonical order (matches pywph).
    canonical = (
        "S11", "S00", "C00", "S01", "C01", "S10", "C10",
        "Cphase", "Cphase_inv", "L",
    )
    classes = [c for c in canonical if c in cfg.classes]

    wph_indices: list[list[int]] = []
    wph_classes: list[str] = []
    sm_indices: list[list[int]] = []

    def emit(clas: str, j1: int, t1: int, p1: int, j2: int, t2: int, p2: int,
             n: int, a: int, pseudo: int = 0) -> None:
        wph_indices.append([j1, t1, p1, j2, t2, p2, n, a, pseudo])
        wph_classes.append(clas)

    for clas in classes:
        if clas == "S11":
            for j1 in range(cfg.j_min, cfg.J):
                for t1 in range(L):
                    for n in range(dn + 1):
                        if n == 0:
                            emit("S11", j1, t1, 1, j1, t1, 1, 0, 0)
                        else:
                            for a in range(A):
                                emit("S11", j1, t1, 1, j1, t1, 1, n, a)
        elif clas == "S00":
            for j1 in range(cfg.j_min, cfg.J):
                for t1 in range(L):
                    for n in range(dn + 1):
                        if n == 0:
                            emit("S00", j1, t1, 0, j1, t1, 0, 0, 0)
                        else:
                            for a in range(A):
                                emit("S00", j1, t1, 0, j1, t1, 0, n, a)
        elif clas == "C00":
            for j1 in range(cfg.j_min, cfg.J):
                for j2 in range(j1 + 1, min(j1 + 1 + dj, cfg.J)):
                    for t1 in range(L):
                        for t2 in range(t1 - dl, t1 + dl):
                            emit("C00", j1, t1, 0, j2, t2 % L, 0, 0, 0)
        elif clas == "S01":
            for j1 in range(cfg.j_min, cfg.J):
                for t1 in range(L):
                    emit("S01", j1, t1, 0, j1, t1, 1, 0, 0)
        elif clas == "C01":
            for j1 in range(cfg.j_min, cfg.J):
                for j2 in range(j1 + 1, min(j1 + 1 + dj, cfg.J)):
                    for t1 in range(L):
                        for t2 in range(t1 - dl, t1 + dl):
                            if t1 == t2:
                                for n in range(dn + 1):
                                    if n == 0:
                                        emit("C01", j1, t1, 0, j2, t2, 1, 0, 0)
                                    else:
                                        for a in range(2 * A):
                                            emit("C01", j1, t1, 0, j2, t2, 1, n, a)
                            else:
                                emit("C01", j1, t1, 0, j2, t2 % L, 1, 0, 0)
        elif clas == "S10":
            for j1 in range(cfg.j_min, cfg.J):
                for t1 in range(L):
                    emit("S10", j1, t1, 1, j1, t1, 0, 0, 0)
        elif clas == "C10":
            for j1 in range(cfg.j_min, cfg.J):
                for j2 in range(j1 + 1, min(j1 + 1 + dj, cfg.J)):
                    for t1 in range(L):
                        for t2 in range(t1 - dl, t1 + dl):
                            if t1 == t2:
                                for n in range(dn + 1):
                                    if n == 0:
                                        emit("C10", j2, t2, 1, j1, t1, 0, 0, 0)
                                    else:
                                        for a in range(2 * A):
                                            emit("C10", j2, t2, 1, j1, t1, 0, n, a)
                            else:
                                emit("C10", j2, t2 % L, 1, j1, t1, 0, 0, 0)
        elif clas == "Cphase":
            for j1 in range(cfg.j_min, cfg.J):
                for j2 in range(j1 + 1, min(j1 + 1 + dj, cfg.J)):
                    for t1 in range(L):
                        dn_eff = min(cfg.J - 1 - j2, dn)
                        for n in range(dn_eff + 1):
                            if n == 0:
                                emit("Cphase", j1, t1, 1, j2, t1, 2 ** (j2 - j1), 0, 0)
                            else:
                                for a in range(2 * A):
                                    emit("Cphase", j1, t1, 1, j2, t1, 2 ** (j2 - j1), n, a)
        elif clas == "Cphase_inv":
            for j1 in range(cfg.j_min, cfg.J):
                for j2 in range(j1 + 1, min(j1 + 1 + dj, cfg.J)):
                    for t1 in range(L):
                        for n in range(dn + 1):
                            if n == 0:
                                emit("Cphase_inv", j2, t1, 2 ** (j2 - j1), j1, t1, 1, 0, 0)
                            else:
                                for a in range(2 * A):
                                    emit("Cphase_inv", j2, t1, 2 ** (j2 - j1), j1, t1, 1, n, a)
        elif clas == "L":
            for j in cfg.sm_j_list:
                for p1, p2 in cfg.sm_p_list:
                    sm_indices.append([j, p1, p2, 0])
        else:
            msg = f"Unknown class {clas!r}"
            raise ValueError(msg)

    wph_arr = np.array(wph_indices, dtype=np.int64)

    # Group consecutive WPH rows that share (j1,t1,p1,j2,t2,p2,pseudo) into
    # one cov (= one cross-correlation FFT). Translations only differ within
    # a cov by (n, a), which we handle by sampling the same FFT result.
    cov_psi: list[list[int]] = []
    cov_p: list[list[int]] = []
    cov_class: list[int] = []
    wph_id_cov = np.empty(wph_arr.shape[0], dtype=np.int64)
    wph_tau = np.empty((wph_arr.shape[0], 2), dtype=np.int64)
    classes_in_order = list(dict.fromkeys(wph_classes))    # preserve order
    class_to_id = {c: i for i, c in enumerate(classes_in_order)}

    id_cov = -1
    curr_key: tuple[int, ...] | None = None
    for i, row in enumerate(wph_arr):
        j1, t1, p1, j2, t2, p2, n, a, pseudo = row.tolist()
        key = (j1, t1, p1, j2, t2, p2, pseudo)
        if key != curr_key:
            curr_key = key
            id_cov += 1
            cov_psi.append([
                _psi_lin_id(j1, t1, j_min=cfg.j_min, L=L),
                _psi_lin_id(j2, t2, j_min=cfg.j_min, L=L),
            ])
            cov_p.append([p1, p2])
            cov_class.append(class_to_id[wph_classes[i]])
        wph_id_cov[i] = id_cov
        tx, ty = _tau_xy(t2, n, a, L=L, A=A, tau_grid=tau_grid)
        wph_tau[i, 0] = ty % cfg.M    # (row, col) order
        wph_tau[i, 1] = tx % cfg.N

    sm_arr = (
        np.array(sm_indices, dtype=np.int64)
        if sm_indices else np.zeros((0, 4), dtype=np.int64)
    )
    # Map sm_arr[:,0] (j) to phi-bank index, matching pywph: phi_bank
    # contains every j in sm_j_list (including j=-1, sub-pixel Gaussian).
    phi_j_list = sorted(set(cfg.sm_j_list))
    phi_j_to_id = {j: i for i, j in enumerate(phi_j_list)}
    sm_phi_id = np.array(
        [phi_j_to_id[int(row[0])] for row in sm_arr],
        dtype=np.int64,
    )

    return MomentTable(
        wph_classes=wph_classes,
        wph_indices=wph_arr,
        cov_psi=np.array(cov_psi, dtype=np.int64),
        cov_p=np.array(cov_p, dtype=np.int64),
        cov_class=np.array(cov_class, dtype=np.int64),
        wph_id_cov=wph_id_cov,
        wph_tau=wph_tau,
        sm_indices=sm_arr,
        sm_phi_id=sm_phi_id,
    )


# ---------------------------------------------------------------------------
# Phase harmonic
# ---------------------------------------------------------------------------


def _phase_harmonic(z: jnp.ndarray, p: int) -> jnp.ndarray:
    """Phase harmonic of order ``p`` (matches ``pywph.utils.phase_harmonics``).

    p = 0 : |z|       (cast to complex)
    p = 1 : z         (unchanged)
    p >= 2: |z| * exp(i p arg(z))
    """

    if p == 0:
        return jnp.abs(z).astype(z.dtype)
    if p == 1:
        return z
    return jnp.abs(z) * jnp.exp(1j * p * jnp.angle(z))


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------


@dataclass
class WPHOp:
    """JAX WPH operator: filters + moment table baked together.

    Use ``compute_S`` for a single field (P, P) -> (n_total,) complex, or
    ``compute_S_batch`` for (B, P, P) -> (B, n_total) complex.
    """

    config: WPHConfig
    table: MomentTable
    psi_f: jnp.ndarray            # (n_psi, M, N) float64
    phi_f: jnp.ndarray            # (n_phi, M, N) float64

    @classmethod
    def build(cls, config: WPHConfig, *, dtype: jnp.dtype = jnp.float32) -> WPHOp:
        psi, _ = build_psi_bank(config.filter_config())
        phi, _ = build_phi_bank(config.filter_config())
        psi_jax = jnp.asarray(psi, dtype=dtype)
        phi_jax = (
            jnp.asarray(phi, dtype=dtype) if phi.size
            else jnp.zeros((0, config.M, config.N), dtype)
        )
        table = build_moment_table(config)
        return cls(config=config, table=table, psi_f=psi_jax, phi_f=phi_jax)

    @property
    def n_total(self) -> int:
        return self.table.n_total

    @property
    def n_wph(self) -> int:
        return self.table.n_wph

    @property
    def n_sm(self) -> int:
        return self.table.n_sm


def _wavelet_transform(x_f: jnp.ndarray, psi_f: jnp.ndarray) -> jnp.ndarray:
    """Wavelet transform: returns u of shape ``(n_psi, M, N)`` complex."""

    return jnp.fft.ifft2(x_f[None, :, :] * psi_f, axes=(-2, -1))


def _build_Z(
    u: jnp.ndarray,
    abs_u: jnp.ndarray,
    psi_id: jnp.ndarray,        # (n_cov,) int
    p: jnp.ndarray,             # (n_cov,) int (static via numpy array)
    p_unique: tuple[int, ...],
) -> jnp.ndarray:
    """Build ``(n_cov, M, N)`` complex Z = phase_harmonic(u[psi_id], p).

    Uses a small Python loop over the unique p values (typically <=5) so
    each branch is handled with constant ``p``, then ``jnp.where`` selects
    the right one per-row. Avoids per-row Python loops while keeping ``p``
    static for jit.
    """

    z = u[psi_id]                                # (n_cov, M, N) complex
    # Start with p == 1 (most common).
    out = z
    for p_val in p_unique:
        mask = (p == p_val)[:, None, None]       # (n_cov, 1, 1)
        if p_val == 1:
            continue
        ph = _phase_harmonic(z, int(p_val))
        out = jnp.where(mask, ph, out)
    return out


def _compute_wph_chunk(
    u: jnp.ndarray,
    abs_u: jnp.ndarray,
    cov_psi: jnp.ndarray,
    cov_p: jnp.ndarray,
    p_unique: tuple[int, ...],
    wph_tau_in_cov: jnp.ndarray,    # (n_wph, 2) (ty, tx)
    wph_id_cov: jnp.ndarray,        # (n_wph,)
    M: int,
    N: int,
) -> jnp.ndarray:
    """Compute all WPH moments via per-cov FFT cross-correlation.

    Returns ``(n_wph,)`` complex.
    """

    Z1 = _build_Z(u, abs_u, cov_psi[:, 0], cov_p[:, 0], p_unique)
    Z2 = _build_Z(u, abs_u, cov_psi[:, 1], cov_p[:, 1], p_unique)
    # pywph subtracts the per-row spatial mean of Z1 and Z2 before the
    # cross-correlation even when ``norm=None`` (see ``_wph_normalization``
    # line 851). Without this, S00 = <|W|^2> instead of Var(|W|), and
    # similarly for any class where Z1 or Z2 has a non-zero spatial mean.
    Z1 = Z1 - jnp.mean(Z1, axis=(-2, -1), keepdims=True)
    Z2 = Z2 - jnp.mean(Z2, axis=(-2, -1), keepdims=True)

    # cross_corr(tau) = ifft(fft(Z1) * conj(fft(Z2))) / (M*N)
    Z1_f = jnp.fft.fft2(Z1, axes=(-2, -1))
    Z2_f = jnp.fft.fft2(Z2, axes=(-2, -1))
    corr = jnp.fft.ifft2(Z1_f * jnp.conj(Z2_f), axes=(-2, -1)) / (M * N)
    # Sample at (ty, tx) for each WPH moment.
    return corr[wph_id_cov, wph_tau_in_cov[:, 0], wph_tau_in_cov[:, 1]]


def _compute_sm(
    x: jnp.ndarray,
    x_f: jnp.ndarray,
    phi_f: jnp.ndarray,
    sm_indices: jnp.ndarray,        # (n_sm, 4)
    sm_phi_id: jnp.ndarray,         # (n_sm,)
    p_unique_sm: tuple[int, ...],
) -> jnp.ndarray:
    """Scaling moments: <ph(x_phi, p1) * conj(ph(x_phi, p2))>.

    Handles ``j == -1`` by feeding the mean-subtracted input directly.
    """

    if sm_indices.shape[0] == 0:
        return jnp.zeros((0,), dtype=x_f.dtype)

    # Convolve x with each phi (mean-subtraction matches pywph).
    x_phi_bank = jnp.fft.ifft2(
        x_f[None, :, :] * phi_f, axes=(-2, -1),
    )
    x_phi_bank = x_phi_bank - jnp.mean(x_phi_bank, axis=(-2, -1), keepdims=True)
    sel = x_phi_bank[sm_phi_id]                     # (n_sm, M, N) complex

    # Apply phase harmonic to each side using static-p partitioning.
    p1 = sm_indices[:, 1]
    p2 = sm_indices[:, 2]
    n_sm = sel.shape[0]
    psi_id = jnp.arange(n_sm)
    Z1 = _build_Z(sel, jnp.abs(sel), psi_id, p1, p_unique_sm)
    Z2 = _build_Z(sel, jnp.abs(sel), psi_id, p2, p_unique_sm)
    # pywph applies a second mean removal post-phase-harmonic even when
    # norm=None (see ``_sm_normalization`` line 928). Without it, (0,0)
    # and (1,1) for real fields collapse to the same value.
    Z1 = Z1 - jnp.mean(Z1, axis=(-2, -1), keepdims=True)
    Z2 = Z2 - jnp.mean(Z2, axis=(-2, -1), keepdims=True)
    return jnp.mean(Z1 * jnp.conj(Z2), axis=(-2, -1))


def _make_forward(op: WPHOp):
    """Build a jit-friendly forward closure for one ``WPHOp`` instance.

    The static (Python-side) data are baked in via closure; the resulting
    function only depends on the input field ``x`` of shape ``(M, N)``.
    """

    cfg = op.config
    table = op.table

    psi_f = op.psi_f
    phi_f = op.phi_f

    cov_psi = jnp.asarray(table.cov_psi, dtype=jnp.int32)
    cov_p = jnp.asarray(table.cov_p, dtype=jnp.int32)
    wph_id_cov = jnp.asarray(table.wph_id_cov, dtype=jnp.int32)
    wph_tau = jnp.asarray(table.wph_tau, dtype=jnp.int32)
    sm_indices = jnp.asarray(table.sm_indices, dtype=jnp.int32)
    sm_phi_id = jnp.asarray(table.sm_phi_id, dtype=jnp.int32)

    p_unique_wph = tuple(sorted(set(table.cov_p.ravel().tolist())))
    p_unique_sm = tuple(sorted(set(
        table.sm_indices[:, 1:3].ravel().tolist()
        if table.sm_indices.size else [0]
    )))

    M, N = cfg.M, cfg.N
    real_dtype = psi_f.dtype
    complex_dtype = jnp.complex128 if real_dtype == jnp.float64 else jnp.complex64

    @jax.jit
    def forward(x: jnp.ndarray) -> jnp.ndarray:
        x_complex = x.astype(complex_dtype)
        x_f = jnp.fft.fft2(x_complex)
        u = _wavelet_transform(x_f, psi_f.astype(complex_dtype))
        abs_u = jnp.abs(u)
        wph = _compute_wph_chunk(
            u, abs_u, cov_psi, cov_p, p_unique_wph,
            wph_tau, wph_id_cov, M, N,
        )
        sm = _compute_sm(
            x_complex, x_f, phi_f.astype(complex_dtype),
            sm_indices, sm_phi_id, p_unique_sm,
        )
        return jnp.concatenate([wph, sm])

    return forward


# ---------------------------------------------------------------------------
# Cross-WPH: the operator applied to a *pair* of fields (x_a, x_b)
#
# Auto-WPH computes S = <Z1(x) conj(Z2(x))>; cross-WPH computes
# S_ab = <Z1(x_a) conj(Z2(x_b))> using the SAME moment table. For CIB this
# constrains inter-band phase coherence (95/150/220 GHz trace the same
# galaxies). Auto-WPH is the special case x_a == x_b.
# ---------------------------------------------------------------------------


def _compute_wph_chunk_pair(
    u1: jnp.ndarray,
    u2: jnp.ndarray,
    cov_psi: jnp.ndarray,
    cov_p: jnp.ndarray,
    p_unique: tuple[int, ...],
    wph_tau_in_cov: jnp.ndarray,
    wph_id_cov: jnp.ndarray,
    M: int,
    N: int,
) -> jnp.ndarray:
    """Like :func:`_compute_wph_chunk` but Z1 is built from ``u1`` and Z2 from
    ``u2`` (wavelet transforms of two possibly-different fields).
    """
    Z1 = _build_Z(u1, jnp.abs(u1), cov_psi[:, 0], cov_p[:, 0], p_unique)
    Z2 = _build_Z(u2, jnp.abs(u2), cov_psi[:, 1], cov_p[:, 1], p_unique)
    Z1 = Z1 - jnp.mean(Z1, axis=(-2, -1), keepdims=True)
    Z2 = Z2 - jnp.mean(Z2, axis=(-2, -1), keepdims=True)
    Z1_f = jnp.fft.fft2(Z1, axes=(-2, -1))
    Z2_f = jnp.fft.fft2(Z2, axes=(-2, -1))
    corr = jnp.fft.ifft2(Z1_f * jnp.conj(Z2_f), axes=(-2, -1)) / (M * N)
    return corr[wph_id_cov, wph_tau_in_cov[:, 0], wph_tau_in_cov[:, 1]]


def _compute_sm_pair(
    x1_f: jnp.ndarray,
    x2_f: jnp.ndarray,
    phi_f: jnp.ndarray,
    sm_indices: jnp.ndarray,
    sm_phi_id: jnp.ndarray,
    p_unique_sm: tuple[int, ...],
) -> jnp.ndarray:
    """Cross scaling moments: ``<ph(x1_phi, p1) * conj(ph(x2_phi, p2))>``."""
    if sm_indices.shape[0] == 0:
        return jnp.zeros((0,), dtype=x1_f.dtype)

    x1_phi = jnp.fft.ifft2(x1_f[None, :, :] * phi_f, axes=(-2, -1))
    x2_phi = jnp.fft.ifft2(x2_f[None, :, :] * phi_f, axes=(-2, -1))
    x1_phi = x1_phi - jnp.mean(x1_phi, axis=(-2, -1), keepdims=True)
    x2_phi = x2_phi - jnp.mean(x2_phi, axis=(-2, -1), keepdims=True)
    sel1 = x1_phi[sm_phi_id]
    sel2 = x2_phi[sm_phi_id]

    p1 = sm_indices[:, 1]
    p2 = sm_indices[:, 2]
    n_sm = sel1.shape[0]
    psi_id = jnp.arange(n_sm)
    Z1 = _build_Z(sel1, jnp.abs(sel1), psi_id, p1, p_unique_sm)
    Z2 = _build_Z(sel2, jnp.abs(sel2), psi_id, p2, p_unique_sm)
    Z1 = Z1 - jnp.mean(Z1, axis=(-2, -1), keepdims=True)
    Z2 = Z2 - jnp.mean(Z2, axis=(-2, -1), keepdims=True)
    return jnp.mean(Z1 * jnp.conj(Z2), axis=(-2, -1))


def _make_forward_cross(op: WPHOp):
    """Build a jit-friendly cross-WPH forward ``forward(x_a, x_b) -> (n_total,)``.

    Identical wiring to :func:`_make_forward` except the two sides of every
    moment are drawn from ``x_a`` and ``x_b`` respectively.
    """
    cfg = op.config
    table = op.table
    psi_f = op.psi_f
    phi_f = op.phi_f

    cov_psi = jnp.asarray(table.cov_psi, dtype=jnp.int32)
    cov_p = jnp.asarray(table.cov_p, dtype=jnp.int32)
    wph_id_cov = jnp.asarray(table.wph_id_cov, dtype=jnp.int32)
    wph_tau = jnp.asarray(table.wph_tau, dtype=jnp.int32)
    sm_indices = jnp.asarray(table.sm_indices, dtype=jnp.int32)
    sm_phi_id = jnp.asarray(table.sm_phi_id, dtype=jnp.int32)

    p_unique_wph = tuple(sorted(set(table.cov_p.ravel().tolist())))
    p_unique_sm = tuple(sorted(set(
        table.sm_indices[:, 1:3].ravel().tolist()
        if table.sm_indices.size else [0]
    )))

    M, N = cfg.M, cfg.N
    real_dtype = psi_f.dtype
    complex_dtype = jnp.complex128 if real_dtype == jnp.float64 else jnp.complex64

    @jax.jit
    def forward(x_a: jnp.ndarray, x_b: jnp.ndarray) -> jnp.ndarray:
        xa_c = x_a.astype(complex_dtype)
        xb_c = x_b.astype(complex_dtype)
        xa_f = jnp.fft.fft2(xa_c)
        xb_f = jnp.fft.fft2(xb_c)
        psi_c = psi_f.astype(complex_dtype)
        ua = _wavelet_transform(xa_f, psi_c)
        ub = _wavelet_transform(xb_f, psi_c)
        wph = _compute_wph_chunk_pair(
            ua, ub, cov_psi, cov_p, p_unique_wph, wph_tau, wph_id_cov, M, N,
        )
        sm = _compute_sm_pair(
            xa_f, xb_f, phi_f.astype(complex_dtype),
            sm_indices, sm_phi_id, p_unique_sm,
        )
        return jnp.concatenate([wph, sm])

    return forward


def compute_S_cross(op: WPHOp, x_a: jnp.ndarray, x_b: jnp.ndarray) -> jnp.ndarray:
    """Single-pair cross-WPH coefficients. ``x_a``, ``x_b`` shape ``(M, N)``."""
    return _make_forward_cross(op)(x_a, x_b)


def compute_S_cross_batch(
    op: WPHOp, xs_a: jnp.ndarray, xs_b: jnp.ndarray
) -> jnp.ndarray:
    """Batched cross-WPH. ``xs_a``, ``xs_b`` shape ``(B, M, N)`` -> ``(B, n_total)``."""
    fwd = _make_forward_cross(op)
    return jax.vmap(fwd)(xs_a, xs_b)


# ---------------------------------------------------------------------------
# Public API: single + batched forward
# ---------------------------------------------------------------------------


def compute_S(op: WPHOp, x: jnp.ndarray) -> jnp.ndarray:
    """Single-field WPH coefficient vector. ``x`` shape ``(M, N)``.

    Returns complex array of length ``op.n_total``.
    """

    fwd = _make_forward(op)
    return fwd(x)


def compute_S_batch(op: WPHOp, xs: jnp.ndarray) -> jnp.ndarray:
    """Batched. ``xs`` shape ``(B, M, N)``; returns ``(B, n_total)`` complex."""

    fwd = _make_forward(op)
    return jax.vmap(fwd)(xs)


def to_real_features(s_complex: jnp.ndarray) -> jnp.ndarray:
    """Real-valued feature vector for prior fitting.

    For complex coefficients we keep both real and imaginary parts as
    separate features. This doubles the count but avoids losing phase
    information from translated WPH moments and Cphase coefficients.
    Concretely: ``[Re(s_0), Im(s_0), Re(s_1), Im(s_1), ...]``.
    """

    s = jnp.asarray(s_complex)
    return jnp.stack([s.real, s.imag], axis=-1).reshape(*s.shape[:-1], -1)


# ---------------------------------------------------------------------------
# Prior fitting
# ---------------------------------------------------------------------------


def random_patches(
    field: FloatArray, *, patch_size: int, n_patches: int,
    rng: np.random.Generator | None = None,
) -> FloatArray:
    """Sample ``n_patches`` random crops."""

    rng = rng if rng is not None else np.random.default_rng()
    arr = np.asarray(field)
    if arr.ndim != 2:
        raise ValueError(f"expected 2D field, got shape {arr.shape}")
    h, w = arr.shape
    if patch_size > min(h, w):
        raise ValueError(
            f"patch_size {patch_size} exceeds field shape {arr.shape}"
        )
    out = np.empty((n_patches, patch_size, patch_size), dtype=np.float32)
    for i in range(n_patches):
        y0 = int(rng.integers(0, h - patch_size + 1))
        x0 = int(rng.integers(0, w - patch_size + 1))
        out[i] = arr[y0 : y0 + patch_size, x0 : x0 + patch_size]
    return out


def d4_orbit(patch: FloatArray) -> FloatArray:
    """8 D4-group transforms stacked on axis 0."""

    arr = np.asarray(patch)
    if arr.ndim != 2:
        raise ValueError(f"expected 2D patch, got shape {arr.shape}")
    out = []
    for flipped in (arr, arr[:, ::-1]):
        for k in range(4):
            out.append(np.rot90(flipped, k=k))
    return np.stack(out, axis=0).astype(np.float32, copy=False)


@dataclass
class WPHPriorStats:
    """Empirical mean/cov of the real-valued WPH feature vector."""

    mean: FloatArray              # (n_features,)
    cov: FloatArray               # (n_features, n_features)
    n_samples: int
    config: WPHConfig
    coeff_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_features(self) -> int:
        return int(self.mean.shape[0])

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            mean=self.mean.astype(np.float64),
            cov=self.cov.astype(np.float64),
            n_samples=np.array(self.n_samples),
            config_json=np.array(repr(self.config.to_dict())),
            coeff_metadata_json=np.array(json.dumps(self.coeff_metadata)),
        )

    @classmethod
    def load(cls, path: Path) -> WPHPriorStats:
        data = np.load(path, allow_pickle=False)
        cfg_dict = ast.literal_eval(str(data["config_json"]))
        cfg = WPHConfig(
            M=int(cfg_dict["M"]), N=int(cfg_dict["N"]),
            J=int(cfg_dict["J"]), L=int(cfg_dict["L"]),
            j_min=int(cfg_dict["j_min"]),
            dn=int(cfg_dict["dn"]), A=int(cfg_dict["A"]),
            dj=cfg_dict["dj"], dl=cfg_dict["dl"],
            sm_j_list=tuple(cfg_dict["sm_j_list"]),
            sm_p_list=tuple(tuple(t) for t in cfg_dict["sm_p_list"]),
            classes=tuple(cfg_dict["classes"]),
            k0=float(cfg_dict["k0"]), sigma0=float(cfg_dict["sigma0"]),
            tau_grid=str(cfg_dict["tau_grid"]),
        )
        if "coeff_metadata_json" in data:
            coeff_metadata = json.loads(str(data["coeff_metadata_json"]))
        else:
            coeff_metadata = {}
        return cls(
            mean=np.asarray(data["mean"], dtype=np.float64),
            cov=np.asarray(data["cov"], dtype=np.float64),
            n_samples=int(data["n_samples"]),
            config=cfg,
            coeff_metadata=coeff_metadata,
        )


def _transform_y_for_wph(
    y: FloatArray,
    *,
    transform: str,
    log_eps: float,
) -> FloatArray:
    arr = np.asarray(y, dtype=np.float32)
    if transform == "none":
        return arr
    if transform == "log":
        return np.log(np.maximum(arr, 0.0) + log_eps).astype(np.float32)
    if transform == "sqrt":
        return np.sqrt(np.maximum(arr, 0.0)).astype(np.float32)
    raise ValueError(f"unknown WPH transform {transform!r}")


def estimate_wph_prior(
    y_truth: FloatArray, cfg: WPHConfig, *,
    n_patches: int = 256, augment_d4: bool = True, seed: int = 0,
    batch_size: int = 8, cov_ridge_rel: float = 1e-6,
    transform: str = "none", log_eps: float | None = None,
    standardize: bool = False,
) -> WPHPriorStats:
    """Estimate ``(mu, Sigma)`` of the real-valued WPH features.

    Same structure as the old WST prior fit: ``n_patches`` random crops
    times 8 D4-orbit augmentations -> empirical mean and covariance of the
    real-valued feature vector.
    """

    rng = np.random.default_rng(seed)
    patches = random_patches(
        y_truth, patch_size=cfg.M, n_patches=n_patches, rng=rng,
    )
    if transform == "log":
        if log_eps is None:
            positives = np.asarray(y_truth)[np.asarray(y_truth) > 0]
            log_eps = 0.1 * float(positives.min()) if positives.size else 1e-12
    elif log_eps is None:
        log_eps = 0.0

    patches_t = _transform_y_for_wph(
        patches, transform=transform, log_eps=float(log_eps),
    )
    patch_means = patches_t.mean(axis=(-2, -1), keepdims=True)
    patch_stds = patches_t.std(axis=(-2, -1), keepdims=True)
    safe_stds = np.maximum(patch_stds, 1e-30)
    if standardize:
        patches_for_wph = (patches_t - patch_means) / safe_stds
    else:
        patches_for_wph = patches_t

    if augment_d4:
        aug = np.stack([d4_orbit(p) for p in patches_for_wph], axis=0)
        all_patches = aug.reshape(-1, cfg.M, cfg.N)
    else:
        all_patches = patches_for_wph

    op = WPHOp.build(cfg)

    chunks = []
    for start in range(0, all_patches.shape[0], batch_size):
        chunk = all_patches[start : start + batch_size]
        s_complex = compute_S_batch(op, jnp.asarray(chunk))
        s_real = to_real_features(s_complex)
        chunks.append(np.asarray(s_real))
    s_all = np.concatenate(chunks, axis=0)

    mean = s_all.mean(axis=0)
    centered = s_all - mean
    n = centered.shape[0]
    cov = (centered.T @ centered) / max(n - 1, 1)
    if cov_ridge_rel:
        diag = float(np.trace(cov) / cov.shape[0])
        cov = cov + cov_ridge_rel * diag * np.eye(cov.shape[0], dtype=np.float64)

    return WPHPriorStats(
        mean=mean.astype(np.float64),
        cov=cov.astype(np.float64),
        n_samples=int(n),
        config=cfg,
        coeff_metadata={
            "d4_augmented": bool(augment_d4),
            "transform": transform,
            "log_eps": float(log_eps),
            "standardize": bool(standardize),
            "patch_mean_samples": patch_means[:, 0, 0].astype(float).tolist(),
            "patch_std_samples": patch_stds[:, 0, 0].astype(float).tolist(),
        },
    )


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------


def cosine_window(M: int, N: int, *, edge_frac: float = 0.1) -> np.ndarray:
    """Separable raised-cosine taper. ``edge_frac`` of pixels on each side
    are tapered from 0 -> 1; interior pixels are 1.
    """

    def axis_window(L: int) -> np.ndarray:
        n_edge = max(1, int(L * edge_frac))
        w = np.ones(L, dtype=np.float64)
        ramp = 0.5 * (1 - np.cos(np.pi * np.arange(n_edge) / n_edge))
        w[:n_edge] = ramp
        w[-n_edge:] = ramp[::-1]
        return w

    return np.outer(axis_window(M), axis_window(N))


def _normalize_target_S(
    target_S: jnp.ndarray | FloatArray | None,
    n_samples: int,
    n_total: int,
    prior: WPHPriorStats | None,
    target_mode: str,
    f_wph: float,
    seed: int,
) -> jnp.ndarray:
    if target_S is not None:
        t = jnp.asarray(target_S)
        if t.ndim == 1:
            t = jnp.broadcast_to(t, (n_samples, t.shape[0]))
        return t

    if prior is None:
        raise ValueError("must provide target_S or (prior + target_mode)")
    if target_mode not in ("sample", "mean"):
        raise ValueError(
            f"target_mode must be 'sample' or 'mean', got {target_mode!r}"
        )
    rng_t = np.random.default_rng(seed + 7)
    if target_mode == "mean":
        t_real = np.broadcast_to(prior.mean, (n_samples, prior.n_features))
    else:
        chol_p = np.linalg.cholesky(prior.cov)
        zs = rng_t.standard_normal((n_samples, prior.n_features))
        t_real = prior.mean + f_wph * (zs @ chol_p.T)
    t_real = t_real.reshape(n_samples, -1, 2)
    return jnp.asarray(t_real[..., 0] + 1j * t_real[..., 1])


def synthesize_from_prior(
    op: WPHOp,
    *,
    target_S: jnp.ndarray | FloatArray | None = None,
    n_samples: int = 4,
    n_iters: int = 200,
    init_std: float | None = None,
    init_kind: str = "white_noise",
    seed: int = 0,
    method: str = "lbfgs",
    lr: float = 0.05,
    grad_clip: float = 1.0,
    cosine_decay: bool = True,
    nonneg: bool = False,
    use_window: bool = False,
    window_edge_frac: float = 0.1,
    f_wph: float = 1.0,
    prior: WPHPriorStats | None = None,
    target_mode: str = "truth",
    verbose_every: int = 0,
    lbfgs_memory: int = 10,
    lbfgs_linesearch: str = "zoom",
    loss_norm: str = "none",
    loss_norm_eps_rel: float = 1e-3,
) -> tuple[FloatArray, dict[str, FloatArray]]:
    """Synthesize ``n_samples`` y-fields whose WPH features match ``target_S``.

    Recommended defaults match the standard WPH-synthesis recipe in the
    pywph papers: signed white-noise initialization at the same RMS as the
    targets, no softplus reparameterization, no cosine window, and L-BFGS
    optimization.

    Parameters
    ----------
    target_S : (n_total,) or (n_samples, n_total) complex, optional
        Target WPH coefficient vector(s). If ``None``, draw from ``prior``.
    n_iters : int
        L-BFGS iterations (each iteration does its own line search +
        multiple gradient evaluations) -- so n_iters=200 is plenty for
        most cases. With ``method="adam"``, this is the number of Adam
        steps; you'll typically need 2000+ for non-trivial syntheses.
    init_std : float or None
        Std of the white-noise init in physical units. If ``None``, set
        from the target: ``sqrt(2 * S00_real_part_at_largest_scale)`` --
        i.e., match the target field's variance roughly.
    init_kind : "white_noise" or "softplus"
        ``white_noise``: ``y_init ~ N(0, init_std^2)`` (signed). Standard
        for WPH synthesis -- the optimizer is free to grow signed peaks.
        ``softplus``: ``y_init = init_std * softplus(N(0, 1)) * window``
        which keeps ``y >= 0`` (the legacy nonneg path).
    method : "lbfgs" or "adam"
        L-BFGS is the standard choice for WPH/WST synthesis. Adam is kept
        for backward compatibility but typically underfits the morphology.
    use_window : bool
        Multiply ``y`` by a cosine window inside the loss to suppress
        wrap-around boundary artifacts. Off by default for ``white_noise``
        init -- the window kills the boundary morphology of the synthesized
        field, which usually isn't what you want.
    loss_norm : "none" or "abs_target"
        ``none`` uses the raw complex L2 loss, matching the standard WPH
        synthesis objective. ``abs_target`` divides each coefficient
        residual by ``max(|target_S|, eps)``. That can be useful for
        diagnostics, but it tends to over-weight near-zero WPH coefficients
        and can suppress high-contrast y-map structure.
    loss_norm_eps_rel : float
        Floor for ``loss_norm="abs_target"`` expressed as a fraction of
        ``mean(|target_S|)`` -- prevents tiny target coefficients from
        blowing up the loss.
    """

    cfg = op.config
    M, N = cfg.M, cfg.N
    fwd = _make_forward(op)

    target_S_arr = _normalize_target_S(
        target_S, n_samples, op.n_total, prior, target_mode, f_wph, seed,
    )

    if use_window:
        win = jnp.asarray(
            cosine_window(M, N, edge_frac=window_edge_frac), dtype=jnp.float32,
        )
    else:
        win = jnp.ones((M, N), dtype=jnp.float32)

    if prior is not None:
        chol = jnp.asarray(np.linalg.cholesky(prior.cov), dtype=jnp.float32)
    else:
        chol = None

    # Auto-pick init_std from the target if not supplied. The S00 entries
    # at the smallest j (largest spatial scale) are essentially Var(|W|),
    # whose rms ~ rms(y); matching this gets us in the right amplitude
    # ballpark without poking around in the prior.
    if init_std is None:
        s_mag = jnp.abs(target_S_arr)
        init_std_jax = jnp.sqrt(jnp.mean(s_mag[:, : op.n_wph]))
        init_std_arr = np.asarray(init_std_jax)
    else:
        init_std_arr = np.full((n_samples,), float(init_std))
    init_std_arr = np.broadcast_to(np.atleast_1d(init_std_arr), (n_samples,))

    if init_kind not in ("white_noise", "softplus"):
        raise ValueError("init_kind must be 'white_noise' or 'softplus'")
    if method not in ("lbfgs", "adam"):
        raise ValueError("method must be 'lbfgs' or 'adam'")
    if init_kind == "softplus" and not nonneg:
        # Legacy combo only made sense with nonneg=True.
        nonneg = True

    def y_from_u(u: jnp.ndarray, init_std_val: float) -> jnp.ndarray:
        if init_kind == "softplus":
            return float(init_std_val) * jax.nn.softplus(u) * win
        # white_noise: u IS the field (in physical units), no reparam.
        return u * win

    if loss_norm not in ("none", "abs_target"):
        raise ValueError("loss_norm must be 'none' or 'abs_target'")

    def make_loss(t_S: jnp.ndarray, init_std_val: float):
        if loss_norm == "abs_target":
            mag = jnp.abs(t_S)
            eps = float(loss_norm_eps_rel) * float(jnp.mean(mag))
            inv_w = 1.0 / jnp.maximum(mag, eps)
        else:
            inv_w = jnp.ones_like(jnp.abs(t_S))

        def loss(u: jnp.ndarray) -> jnp.ndarray:
            y = y_from_u(u, init_std_val)
            s = fwd(y)
            delta = (s - t_S) * inv_w.astype(s.dtype)
            if chol is None:
                return 0.5 * jnp.sum(jnp.abs(delta) ** 2) / (f_wph ** 2)
            delta_real = jnp.stack(
                [delta.real, delta.imag], axis=-1,
            ).reshape(-1)
            z = jax.scipy.linalg.solve_triangular(chol, delta_real, lower=True)
            return 0.5 * jnp.dot(z, z) / (f_wph ** 2)
        return loss

    samples = np.empty((n_samples, M, N), dtype=np.float32)
    losses = np.empty((n_samples, n_iters), dtype=np.float32)
    key = jax.random.PRNGKey(seed)

    for i in range(n_samples):
        key, key_u = jax.random.split(key, 2)
        std_i = float(init_std_arr[i])
        if init_kind == "softplus":
            u = jax.random.normal(key_u, (M, N))
        else:
            u = std_i * jax.random.normal(key_u, (M, N))
        loss_fn_i = make_loss(target_S_arr[i], std_i)

        if method == "lbfgs":
            opt = optax.lbfgs(
                memory_size=lbfgs_memory, linesearch=optax.scale_by_zoom_linesearch(
                    max_linesearch_steps=20
                ) if lbfgs_linesearch == "zoom" else None,
            )
            value_and_grad_fn = optax.value_and_grad_from_state(loss_fn_i)
            opt_state = opt.init(u)

            @jax.jit
            def step(u, opt_state):
                value, grad = value_and_grad_fn(u, state=opt_state)
                updates, opt_state = opt.update(
                    grad, opt_state, u,
                    value=value, grad=grad, value_fn=loss_fn_i,
                )
                u = optax.apply_updates(u, updates)
                return u, opt_state, value

            for it in range(n_iters):
                u, opt_state, lv = step(u, opt_state)
                losses[i, it] = float(lv)
                if verbose_every and (it == 0 or (it + 1) % verbose_every == 0):
                    print(f"  sample {i} iter {it + 1}: loss = {float(lv):.4e}")
        else:
            if cosine_decay:
                sched = optax.cosine_decay_schedule(
                    lr, decay_steps=n_iters, alpha=0.01,
                )
                opt = optax.chain(
                    optax.clip_by_global_norm(grad_clip), optax.adam(sched),
                )
            else:
                opt = optax.chain(
                    optax.clip_by_global_norm(grad_clip), optax.adam(lr),
                )
            opt_state = opt.init(u)

            @jax.jit
            def step(u, opt_state):
                lv, grad = jax.value_and_grad(loss_fn_i)(u)
                upd, opt_state = opt.update(grad, opt_state, u)
                u = optax.apply_updates(u, upd)
                return u, opt_state, lv

            for it in range(n_iters):
                u, opt_state, lv = step(u, opt_state)
                losses[i, it] = float(lv)
                if verbose_every and (it == 0 or (it + 1) % verbose_every == 0):
                    print(f"  sample {i} iter {it + 1}: loss = {float(lv):.4e}")

        samples[i] = np.asarray(y_from_u(u, std_i))

    return samples, {"loss": losses}


# Re-export FilterBankConfig so consumers can introspect the wavelet bank.
__all__ = [
    "DEFAULT_CLASSES",
    "DEFAULT_SM_P_LIST",
    "FilterBankConfig",
    "MomentTable",
    "WPHConfig",
    "WPHOp",
    "WPHPriorStats",
    "build_moment_table",
    "compute_S",
    "compute_S_batch",
    "compute_S_cross",
    "compute_S_cross_batch",
    "cosine_window",
    "d4_orbit",
    "estimate_wph_prior",
    "random_patches",
    "synthesize_from_prior",
    "to_real_features",
]
