"""Flow-matching training step + WPH-feature batch-distribution loss + ODE sampler.

Velocity parameterization (Lipman et al. 2022 / Liu et al. 2023 rectified flow):

    x_0 ~ noise (N(0, I)),  x_1 ~ data
    x_t = (1 - t) x_0 + t x_1                  # straight-line path
    target velocity = x_1 - x_0                # constant along the path
    L_FM = || v_theta(x_t, t) - (x_1 - x_0) ||^2

The model implicitly predicts the clean sample at every t:

    x_hat_1 = x_t + (1 - t) v_theta(x_t, t)

We pipe ``x_hat_1`` (in gaussianized log-y space) through the inverse rank
transform to physical y, compute the WPH feature batch ``F``, whiten it
with the prior's (mu, Sigma), and add a distribution-matching loss

    L_WPH = || mean(F_norm) ||^2 + || cov(F_norm) - I ||_F^2

so the *batch* of generated patches matches the WPH prior in feature
space. The whitening converts a poorly-conditioned multivariate Gaussian
match into a unit-variance moment match, which scales numerically.

Sampling is by ODE integration of dx/dt = v_theta(x, t) from t=0 to t=1
using Heun's method (2nd order). With ``n_steps=30-50`` this matches
diffusion-quality samples at 1/10th the NFE.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from flax.training import train_state

from .wph import WPHOp, _make_forward, _make_forward_cross, to_real_features


class FlowMatchingTrainState(train_state.TrainState):
    """Flax TrainState alias; carries params + opt_state + step counter."""


# ---------------------------------------------------------------------------
# WPH plumbing: build a JIT-friendly batched feature function from a WPHOp
# ---------------------------------------------------------------------------


def make_wph_features_fn(
    wph_op: WPHOp, chunk_size: int | None = None, checkpoint: bool = True
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Return ``y_batch -> F_real`` where ``y_batch`` is ``(B, M, N)`` and
    ``F_real`` is ``(B, 2 * n_total)`` real (Re/Im interleaved per coeff).

    The WPH forward involves wavelet-bank FFTs and many cross-correlation
    intermediates per patch, so:

    - At training time the forward intermediates would otherwise have to
      be stored across the entire batch for backward (often >100 GB).
      ``checkpoint=True`` (default) wraps the per-patch forward in
      ``jax.checkpoint`` so backward recomputes the forward instead of
      storing intermediates -- bounded memory regardless of batch size.

    - cuFFT's scratch allocator can't fit large batched FFT plans, so the
      default path uses ``jax.lax.map`` (sequential, per-patch FFT plans).
      Set ``chunk_size > 1`` to do a chunked ``vmap`` (faster but uses
      more cuFFT scratch and more activation memory).
    """
    fwd = _make_forward(wph_op)
    if checkpoint:
        # Per-patch remat: backward recomputes the WPH forward.
        fwd = jax.checkpoint(fwd)

    if chunk_size is None or chunk_size <= 1:
        def features(y_batch: jnp.ndarray) -> jnp.ndarray:
            s_complex = jax.lax.map(fwd, y_batch)
            return to_real_features(s_complex)

        return features

    fwd_chunk = jax.vmap(fwd)

    def features(y_batch: jnp.ndarray) -> jnp.ndarray:
        b = y_batch.shape[0]
        pad = (-b) % chunk_size
        if pad:
            y_batch = jnp.concatenate([y_batch, y_batch[:pad]], axis=0)
        y_chunked = y_batch.reshape(
            -1, chunk_size, *y_batch.shape[1:]
        )
        s_chunked = jax.lax.map(fwd_chunk, y_chunked)
        s_complex = s_chunked.reshape(-1, s_chunked.shape[-1])[:b]
        return to_real_features(s_complex)

    return features


def whitener_from_prior(
    prior_mean: jnp.ndarray,
    prior_cov: jnp.ndarray,
    ridge_rel: float = 1e-3,
    eig_floor_rel: float = 1e-6,
) -> jnp.ndarray:
    """Compute ``W`` such that ``(F - mu) @ W`` has covariance ~ I under the prior.

    The WPH feature covariance is typically near-singular: there are strong
    linear dependencies between coefficients (e.g., S00 across orientations
    at the same scale, or related Cphase entries). Cholesky fails on such
    matrices. We instead use the symmetric inverse-square-root via
    eigendecomposition

        W = V diag(eigs)^{-1/2} V^T

    with two regularizations: a relative ridge added to the diagonal, and a
    floor on the eigenvalues at ``eig_floor_rel * max(eigs)``. With the
    floor, ``(F - mu) @ W`` has covariance close to (but not exactly) I,
    with directions in the near-null space of Sigma getting heavily
    attenuated rather than infinitely amplified.
    """
    import numpy as np

    cov = np.asarray(prior_cov, dtype=np.float64)
    n = cov.shape[0]
    if n == 0:
        return jnp.zeros((0, 0), dtype=jnp.float32)
    diag = float(np.trace(cov) / n)
    cov_reg = cov + ridge_rel * diag * np.eye(n, dtype=np.float64)
    eigs, V = np.linalg.eigh(cov_reg)
    eig_max = float(eigs.max()) if eigs.size else 1.0
    floor = eig_floor_rel * eig_max
    n_floored = int((eigs < floor).sum())
    eigs = np.maximum(eigs, floor)
    if n_floored:
        print(
            f"  whitener: floored {n_floored}/{n} eigenvalues at "
            f"{eig_floor_rel:.1e} * eig_max"
        )
    w = (V * (eigs ** -0.5)[None, :]) @ V.T
    return jnp.asarray(w, dtype=jnp.float32)


# ---------------------------------------------------------------------------
# Loss components
# ---------------------------------------------------------------------------


def fm_loss(
    apply_fn: Callable,
    params: Any,
    x_data: jnp.ndarray,
    t: jnp.ndarray,
    x_noise: jnp.ndarray,
) -> jnp.ndarray:
    """Standard flow matching regression loss with linear-interp paths.

    ``x_data`` and ``x_noise`` have shape ``(B, H, W, C)`` (NHWC), ``t`` has
    shape ``(B,)``.
    """
    t_b = t[:, None, None, None]
    x_t = (1.0 - t_b) * x_noise + t_b * x_data
    v_target = x_data - x_noise
    v_pred = apply_fn(params, x_t, t)
    return jnp.mean((v_pred - v_target) ** 2)


def wph_distribution_loss(
    F_real: jnp.ndarray,
    mu_prior: jnp.ndarray,
    whitener: jnp.ndarray,
) -> jnp.ndarray:
    """Match a batch of WPH features to a Gaussian prior, in whitened space.

    ``F_real``: ``(B, n_features)`` real, returns a scalar.

    Loss has two terms, both *averaged over features* so the scalar is O(1)
    regardless of ``n_features``:

        L_mean = mean_i (mean_b F_norm[b, i])^2
        L_var  = mean_i (var_b F_norm[b, i] - 1)^2

    The full covariance ``cov(F_norm)`` is (B x n) under-determined when
    ``B << n_features`` -- its off-diagonal entries are pure noise of
    variance ~1/B and the Frobenius distance to ``I`` cannot be reduced
    below ~ (n_features^2 - n_features) / B. Matching only the diagonal
    of the covariance (i.e., the per-feature variance) avoids this floor.

    For a generator that draws perfectly from the prior, both terms are
    O(1/B); for a poorly-fit generator, both are O(1) or larger. With
    ``lambda_wph ~ 1`` the WPH gradient is on the same scale as the FM
    regression gradient.
    """
    F_centered = F_real - mu_prior[None, :]
    F_norm = F_centered @ whitener                            # (B, n_features)
    mean_per_feat = F_norm.mean(axis=0)                       # (n_features,)
    mean_loss = jnp.mean(mean_per_feat ** 2)
    var_per_feat = jnp.mean(F_norm ** 2, axis=0) - mean_per_feat ** 2  # biased var
    var_loss = jnp.mean((var_per_feat - 1.0) ** 2)
    return mean_loss + var_loss


# ---------------------------------------------------------------------------
# Combined training step (JIT-compiled closure)
# ---------------------------------------------------------------------------


def make_train_step(
    apply_fn: Callable,
    wph_features_fn: Callable[[jnp.ndarray], jnp.ndarray],
    mu_prior: jnp.ndarray,
    whitener: jnp.ndarray,
    z_grid: jnp.ndarray,
    quantile_grid: jnp.ndarray,
    y0: float,
    lambda_wph: float,
    wph_t_min: float = 0.5,
    lambda_warmup_steps: int = 0,
):
    """Build a JIT-compiled flow-matching + WPH train step.

    The WPH loss is gated to ``t > wph_t_min`` so it kicks in only when
    ``x_hat_1`` is a confident clean prediction. At small t the prediction
    is mostly noise and matching its WPH stats to the data prior is
    meaningless. Gating uses a soft mask: ``w(t) = sigmoid(20*(t - t_min))``.

    ``lambda_warmup_steps`` linearly ramps the WPH coefficient from 0 to
    ``lambda_wph`` over the first ``lambda_warmup_steps`` optimizer steps.
    This avoids mode collapse: at random init, the predicted clean sample
    is gibberish whose WPH features are wildly out of distribution, and
    even a moderate lambda would let that gradient dominate FM training
    and drive the model to a degenerate constant output.
    """
    mu_prior_j = jnp.asarray(mu_prior, dtype=jnp.float32)
    whitener_j = jnp.asarray(whitener, dtype=jnp.float32)
    z_grid_j = jnp.asarray(z_grid, dtype=jnp.float32)
    quantile_grid_j = jnp.asarray(quantile_grid, dtype=jnp.float32)
    y0_j = jnp.float32(y0)
    lambda_wph_j = jnp.float32(lambda_wph)
    wph_t_min_j = jnp.float32(wph_t_min)
    lambda_warmup_j = jnp.int32(max(0, int(lambda_warmup_steps)))

    @jax.jit
    def train_step(state: FlowMatchingTrainState, x_data: jnp.ndarray, key: jnp.ndarray):
        """One training step. ``x_data`` is ``(B, H, W, 1)`` gaussianized."""
        key, k_t, k_n = jax.random.split(key, 3)
        B = x_data.shape[0]
        t = jax.random.uniform(k_t, (B,), dtype=jnp.float32)
        x_noise = jax.random.normal(k_n, x_data.shape, dtype=x_data.dtype)
        t_b = t[:, None, None, None]
        x_t = (1.0 - t_b) * x_noise + t_b * x_data
        v_target = x_data - x_noise

        def loss_fn(params):
            v_pred = state.apply_fn(params, x_t, t)
            l_fm = jnp.mean((v_pred - v_target) ** 2)

            x_hat_1 = x_t + (1.0 - t_b) * v_pred             # (B, H, W, 1)
            log_y = jnp.interp(x_hat_1[..., 0], z_grid_j, quantile_grid_j)
            y_hat = jnp.exp(log_y) - y0_j                    # (B, M, N)

            F_real = wph_features_fn(y_hat)                  # (B, n_features)
            l_wph_raw = wph_distribution_loss(F_real, mu_prior_j, whitener_j)

            # Gate WPH loss by mean over batch of sigmoid(20*(t - t_min)).
            gate = jax.nn.sigmoid(20.0 * (t - wph_t_min_j)).mean()
            # Linear lambda warmup over the first lambda_warmup_j steps.
            warmup_factor = jnp.where(
                lambda_warmup_j > 0,
                jnp.minimum(state.step.astype(jnp.float32) / jnp.maximum(
                    lambda_warmup_j.astype(jnp.float32), 1.0
                ), 1.0),
                jnp.float32(1.0),
            )
            effective_lambda = lambda_wph_j * warmup_factor
            l_wph = gate * l_wph_raw

            return l_fm + effective_lambda * l_wph, (l_fm, l_wph)

        (loss, (l_fm, l_wph)), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(state.params)
        new_state = state.apply_gradients(grads=grads)
        return new_state, loss, l_fm, l_wph, key

    return train_step


# ---------------------------------------------------------------------------
# Per-sample WPH loss + matching train step
# ---------------------------------------------------------------------------


def wph_persample_loss(
    F_pred: jnp.ndarray,
    F_target: jnp.ndarray,
    inv_std_per_feature: jnp.ndarray,
    sample_weight: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Per-sample weighted L2 in WPH feature space.

    For each sample in the batch, penalizes ``WPH(x_hat_1) - WPH(x_target)``,
    rescaled by the per-feature inverse std so each feature contributes on
    the same scale. This is a "perceptual" loss in WPH feature space:
    cluster-containing training samples produce cluster-specific gradient,
    so the model learns to reproduce the right per-sample feature signature
    rather than just matching the prior's average.

    ``sample_weight`` (optional, shape ``(B,)``) gates per-sample contribution
    — used by the train step to apply the t-dependent sigmoid gate.
    """
    diff = (F_pred - F_target) * inv_std_per_feature[None, :]
    persample_err = jnp.mean(diff ** 2, axis=1)        # (B,)
    if sample_weight is None:
        return jnp.mean(persample_err)
    return jnp.sum(sample_weight * persample_err) / jnp.maximum(
        jnp.sum(sample_weight), 1e-8
    )


def make_train_step_persample(
    apply_fn: Callable,
    wph_features_fn: Callable[[jnp.ndarray], jnp.ndarray],
    inv_std_per_feature: jnp.ndarray,
    z_grid: jnp.ndarray,
    quantile_grid: jnp.ndarray,
    y0: float,
    lambda_wph: float,
    wph_t_min: float = 0.5,
    lambda_warmup_steps: int = 0,
):
    """Build a JIT-compiled FM + per-sample-WPH train step.

    Replaces the prior-distribution match in :func:`make_train_step` with
    a per-sample regression in WPH feature space:

        L_WPH_persample = mean_b gate(t_b) * mean_i ((F_pred_bi - F_target_bi) / std_i)^2

    where ``F_target`` is the precomputed WPH feature vector of the
    corresponding training patch (loaded from the precompute_wph_targets.py
    output and indexed in lockstep with ``x_data`` in the data loader).
    """
    inv_std_j = jnp.asarray(inv_std_per_feature, dtype=jnp.float32)
    z_grid_j = jnp.asarray(z_grid, dtype=jnp.float32)
    quantile_grid_j = jnp.asarray(quantile_grid, dtype=jnp.float32)
    y0_j = jnp.float32(y0)
    lambda_wph_j = jnp.float32(lambda_wph)
    wph_t_min_j = jnp.float32(wph_t_min)
    lambda_warmup_j = jnp.int32(max(0, int(lambda_warmup_steps)))

    @jax.jit
    def train_step(
        state: FlowMatchingTrainState,
        x_data: jnp.ndarray,
        F_target: jnp.ndarray,
        key: jnp.ndarray,
    ):
        """One training step. ``x_data``: (B, H, W, 1); ``F_target``: (B, n_features)."""
        key, k_t, k_n = jax.random.split(key, 3)
        B = x_data.shape[0]
        t = jax.random.uniform(k_t, (B,), dtype=jnp.float32)
        x_noise = jax.random.normal(k_n, x_data.shape, dtype=x_data.dtype)
        t_b = t[:, None, None, None]
        x_t = (1.0 - t_b) * x_noise + t_b * x_data
        v_target = x_data - x_noise

        def loss_fn(params):
            v_pred = state.apply_fn(params, x_t, t)
            l_fm = jnp.mean((v_pred - v_target) ** 2)

            x_hat_1 = x_t + (1.0 - t_b) * v_pred
            log_y = jnp.interp(x_hat_1[..., 0], z_grid_j, quantile_grid_j)
            y_hat = jnp.exp(log_y) - y0_j
            F_pred = wph_features_fn(y_hat)

            # Per-sample t-gate; samples near t=1 contribute fully, near t=0 don't.
            gate = jax.nn.sigmoid(20.0 * (t - wph_t_min_j))    # (B,)
            l_wph_raw = wph_persample_loss(F_pred, F_target, inv_std_j, sample_weight=gate)

            warmup_factor = jnp.where(
                lambda_warmup_j > 0,
                jnp.minimum(state.step.astype(jnp.float32) / jnp.maximum(
                    lambda_warmup_j.astype(jnp.float32), 1.0
                ), 1.0),
                jnp.float32(1.0),
            )
            effective_lambda = lambda_wph_j * warmup_factor
            return l_fm + effective_lambda * l_wph_raw, (l_fm, l_wph_raw)

        (loss, (l_fm, l_wph)), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(state.params)
        new_state = state.apply_gradients(grads=grads)
        return new_state, loss, l_fm, l_wph, key

    return train_step


# ---------------------------------------------------------------------------
# Multi-channel WPH (CIB: joint 95/150/220 GHz) — per-channel + cross-band
# ---------------------------------------------------------------------------


def channel_pairs(n_channels: int) -> list[tuple[int, int]]:
    """Unordered channel pairs (a < b), the cross-band WPH block order."""
    return [(a, b) for a in range(n_channels) for b in range(a + 1, n_channels)]


def _wph_batched(fwd: Callable, xs, chunk_size: int | None):
    """Map a per-sample WPH forward over a batch (pytree-aware).

    ``xs`` is either a single ``(B, M, N)`` array (auto) or a tuple of two
    such arrays (cross). Sequential ``lax.map`` by default (cuFFT-scratch
    safe); chunked ``vmap`` when ``chunk_size > 1``.
    """
    if chunk_size is None or chunk_size <= 1:
        return jax.lax.map(fwd, xs)

    leaves = jax.tree_util.tree_leaves(xs)
    b = leaves[0].shape[0]
    pad = (-b) % chunk_size

    def _pad(a):
        return jnp.concatenate([a, a[:pad]], axis=0) if pad else a

    def _reshape(a):
        return a.reshape(-1, chunk_size, *a.shape[1:])

    xs_chunked = jax.tree_util.tree_map(_reshape, jax.tree_util.tree_map(_pad, xs))
    s_chunked = jax.lax.map(jax.vmap(fwd), xs_chunked)
    return s_chunked.reshape(-1, s_chunked.shape[-1])[:b]


def make_wph_features_multi_fn(
    wph_op: WPHOp,
    n_channels: int,
    chunk_size: int | None = None,
    checkpoint: bool = True,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Return ``y_batch -> F_real`` for a multi-channel field.

    ``y_batch`` is ``(B, H, W, C)`` in physical units. The returned feature
    vector concatenates, in this fixed order:

        [ auto(ch 0), auto(ch 1), ..., auto(ch C-1),
          cross(0,1), cross(0,2), ..., cross(C-2, C-1) ]

    each block being ``to_real_features`` of that field's (or pair's) WPH
    coefficients. ``build_wph_prior_cib.py`` and ``precompute_wph_targets_cib.py``
    MUST assemble features in this same order for the targets/prior to line up.
    """
    fwd_auto = _make_forward(wph_op)
    fwd_cross = _make_forward_cross(wph_op)
    if checkpoint:
        fwd_auto = jax.checkpoint(fwd_auto)
        fwd_cross = jax.checkpoint(fwd_cross)
    pairs = channel_pairs(n_channels)

    def _cross_single(ab):
        return fwd_cross(ab[0], ab[1])

    def features(y_batch: jnp.ndarray) -> jnp.ndarray:
        blocks = []
        for c in range(n_channels):
            s = _wph_batched(fwd_auto, y_batch[..., c], chunk_size)
            blocks.append(to_real_features(s))
        for a, b in pairs:
            s = _wph_batched(
                _cross_single, (y_batch[..., a], y_batch[..., b]), chunk_size
            )
            blocks.append(to_real_features(s))
        return jnp.concatenate(blocks, axis=-1)

    return features


def _make_invert_multi(z_grid_j, quantile_grid_j, y0_j, n_channels):
    """Per-channel inverse rank transform: gaussianized (B,H,W,C) -> physical."""

    def invert(x_hat_1: jnp.ndarray) -> jnp.ndarray:
        chans = []
        for c in range(n_channels):
            log_y = jnp.interp(x_hat_1[..., c], z_grid_j, quantile_grid_j[c])
            chans.append(jnp.exp(log_y) - y0_j[c])
        return jnp.stack(chans, axis=-1)

    return invert


def make_train_step_persample_multi(
    apply_fn: Callable,
    wph_features_fn: Callable[[jnp.ndarray], jnp.ndarray],
    inv_std_per_feature: jnp.ndarray,
    z_grid: jnp.ndarray,
    quantile_grid: jnp.ndarray,            # (C, n_quantiles)
    y0,                                     # scalar or (C,)
    n_channels: int,
    lambda_wph: float,
    wph_t_min: float = 0.5,
    lambda_warmup_steps: int = 0,
):
    """Multi-channel per-sample FM + WPH train step (joint CIB bands).

    Identical structure to :func:`make_train_step_persample` but:
    - ``x_data`` is ``(B, H, W, C)``;
    - the inverse rank transform is applied per channel with its own
      ``quantile_grid`` row;
    - ``wph_features_fn`` returns the concatenated per-channel + cross-band
      feature vector matching the precomputed ``F_target``.
    """
    inv_std_j = jnp.asarray(inv_std_per_feature, dtype=jnp.float32)
    z_grid_j = jnp.asarray(z_grid, dtype=jnp.float32)
    quantile_grid_j = jnp.asarray(quantile_grid, dtype=jnp.float32)
    y0_j = jnp.broadcast_to(jnp.asarray(y0, dtype=jnp.float32), (n_channels,))
    lambda_wph_j = jnp.float32(lambda_wph)
    wph_t_min_j = jnp.float32(wph_t_min)
    lambda_warmup_j = jnp.int32(max(0, int(lambda_warmup_steps)))
    invert = _make_invert_multi(z_grid_j, quantile_grid_j, y0_j, n_channels)

    @jax.jit
    def train_step(
        state: FlowMatchingTrainState,
        x_data: jnp.ndarray,
        F_target: jnp.ndarray,
        key: jnp.ndarray,
    ):
        key, k_t, k_n = jax.random.split(key, 3)
        B = x_data.shape[0]
        t = jax.random.uniform(k_t, (B,), dtype=jnp.float32)
        x_noise = jax.random.normal(k_n, x_data.shape, dtype=x_data.dtype)
        t_b = t[:, None, None, None]
        x_t = (1.0 - t_b) * x_noise + t_b * x_data
        v_target = x_data - x_noise

        def loss_fn(params):
            v_pred = state.apply_fn(params, x_t, t)
            l_fm = jnp.mean((v_pred - v_target) ** 2)

            x_hat_1 = x_t + (1.0 - t_b) * v_pred       # (B, H, W, C)
            y_hat = invert(x_hat_1)                    # (B, H, W, C) physical
            F_pred = wph_features_fn(y_hat)

            gate = jax.nn.sigmoid(20.0 * (t - wph_t_min_j))     # (B,)
            l_wph_raw = wph_persample_loss(
                F_pred, F_target, inv_std_j, sample_weight=gate
            )
            warmup_factor = jnp.where(
                lambda_warmup_j > 0,
                jnp.minimum(state.step.astype(jnp.float32) / jnp.maximum(
                    lambda_warmup_j.astype(jnp.float32), 1.0
                ), 1.0),
                jnp.float32(1.0),
            )
            effective_lambda = lambda_wph_j * warmup_factor
            return l_fm + effective_lambda * l_wph_raw, (l_fm, l_wph_raw)

        (loss, (l_fm, l_wph)), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(state.params)
        new_state = state.apply_gradients(grads=grads)
        return new_state, loss, l_fm, l_wph, key

    return train_step


def make_train_step_multi(
    apply_fn: Callable,
    wph_features_fn: Callable[[jnp.ndarray], jnp.ndarray],
    mu_prior: jnp.ndarray,
    whitener: jnp.ndarray,
    z_grid: jnp.ndarray,
    quantile_grid: jnp.ndarray,            # (C, n_quantiles)
    y0,                                     # scalar or (C,)
    n_channels: int,
    lambda_wph: float,
    wph_t_min: float = 0.5,
    lambda_warmup_steps: int = 0,
):
    """Multi-channel distribution-mode FM + WPH train step.

    Like :func:`make_train_step` (whitened batch moment-match to the prior)
    but multi-channel: per-channel inversion and the concatenated
    per-channel + cross-band feature vector.
    """
    mu_prior_j = jnp.asarray(mu_prior, dtype=jnp.float32)
    whitener_j = jnp.asarray(whitener, dtype=jnp.float32)
    z_grid_j = jnp.asarray(z_grid, dtype=jnp.float32)
    quantile_grid_j = jnp.asarray(quantile_grid, dtype=jnp.float32)
    y0_j = jnp.broadcast_to(jnp.asarray(y0, dtype=jnp.float32), (n_channels,))
    lambda_wph_j = jnp.float32(lambda_wph)
    wph_t_min_j = jnp.float32(wph_t_min)
    lambda_warmup_j = jnp.int32(max(0, int(lambda_warmup_steps)))
    invert = _make_invert_multi(z_grid_j, quantile_grid_j, y0_j, n_channels)

    @jax.jit
    def train_step(state: FlowMatchingTrainState, x_data: jnp.ndarray, key: jnp.ndarray):
        key, k_t, k_n = jax.random.split(key, 3)
        B = x_data.shape[0]
        t = jax.random.uniform(k_t, (B,), dtype=jnp.float32)
        x_noise = jax.random.normal(k_n, x_data.shape, dtype=x_data.dtype)
        t_b = t[:, None, None, None]
        x_t = (1.0 - t_b) * x_noise + t_b * x_data
        v_target = x_data - x_noise

        def loss_fn(params):
            v_pred = state.apply_fn(params, x_t, t)
            l_fm = jnp.mean((v_pred - v_target) ** 2)

            x_hat_1 = x_t + (1.0 - t_b) * v_pred
            y_hat = invert(x_hat_1)
            F_real = wph_features_fn(y_hat)
            l_wph_raw = wph_distribution_loss(F_real, mu_prior_j, whitener_j)

            gate = jax.nn.sigmoid(20.0 * (t - wph_t_min_j)).mean()
            warmup_factor = jnp.where(
                lambda_warmup_j > 0,
                jnp.minimum(state.step.astype(jnp.float32) / jnp.maximum(
                    lambda_warmup_j.astype(jnp.float32), 1.0
                ), 1.0),
                jnp.float32(1.0),
            )
            effective_lambda = lambda_wph_j * warmup_factor
            l_wph = gate * l_wph_raw
            return l_fm + effective_lambda * l_wph, (l_fm, l_wph)

        (loss, (l_fm, l_wph)), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(state.params)
        new_state = state.apply_gradients(grads=grads)
        return new_state, loss, l_fm, l_wph, key

    return train_step


# ---------------------------------------------------------------------------
# Vanilla flow-matching train step (no WPH loss) — useful as A/B baseline
# ---------------------------------------------------------------------------


def make_fm_only_train_step(apply_fn: Callable):
    """Build a JIT-compiled flow-matching-only training step (no WPH)."""

    @jax.jit
    def train_step(state: FlowMatchingTrainState, x_data: jnp.ndarray, key: jnp.ndarray):
        key, k_t, k_n = jax.random.split(key, 3)
        B = x_data.shape[0]
        t = jax.random.uniform(k_t, (B,), dtype=jnp.float32)
        x_noise = jax.random.normal(k_n, x_data.shape, dtype=x_data.dtype)
        t_b = t[:, None, None, None]
        x_t = (1.0 - t_b) * x_noise + t_b * x_data
        v_target = x_data - x_noise

        def loss_fn(params):
            v_pred = state.apply_fn(params, x_t, t)
            return jnp.mean((v_pred - v_target) ** 2)

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        new_state = state.apply_gradients(grads=grads)
        return new_state, loss, key

    return train_step


# ---------------------------------------------------------------------------
# Rectified-flow reflow training step (paired (z, x) endpoints)
# ---------------------------------------------------------------------------


def make_fm_reflow_train_step(apply_fn: Callable):
    """Build a JIT-compiled flow-matching train step on *paired* (z, x) data.

    For rectified-flow reflow, the noise endpoint is not random per-batch
    but is the specific ``z`` that produced ``x`` under the original FM
    model's ODE. Training on this coupled pair drives the velocity field
    to be approximately constant along straight lines from z to x, so a
    1-step Euler sampler reaches the target with little error.
    """

    @jax.jit
    def train_step(
        state: FlowMatchingTrainState,
        x_data: jnp.ndarray,         # x endpoints (model samples)
        x_noise: jnp.ndarray,         # paired z endpoints
        key: jnp.ndarray,
    ):
        key, k_t = jax.random.split(key, 2)
        B = x_data.shape[0]
        t = jax.random.uniform(k_t, (B,), dtype=jnp.float32)
        t_b = t[:, None, None, None]
        x_t = (1.0 - t_b) * x_noise + t_b * x_data
        v_target = x_data - x_noise

        def loss_fn(params):
            v_pred = state.apply_fn(params, x_t, t)
            return jnp.mean((v_pred - v_target) ** 2)

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        new_state = state.apply_gradients(grads=grads)
        return new_state, loss, key

    return train_step


# Re-export for parity with the ``train_step`` symbol in __init__.
def train_step(*args, **kwargs):  # pragma: no cover - thin alias
    """See :func:`make_train_step` to construct a JIT-compiled step."""
    raise NotImplementedError("Use make_train_step(...) to build a JIT'd step.")


# ---------------------------------------------------------------------------
# ODE sampling
# ---------------------------------------------------------------------------


@partial(jax.jit, static_argnames=("apply_fn", "n_steps"))
def _sample_heun_from_noise(
    apply_fn: Callable,
    params: Any,
    init_noise: jnp.ndarray,
    n_steps: int,
) -> jnp.ndarray:
    """JIT-compiled Heun sampler from a fixed noise endpoint."""
    n = init_noise.shape[0]
    dt = jnp.float32(1.0 / n_steps)

    def body(i: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        t_now = i.astype(jnp.float32) * dt
        t1 = jnp.full((n,), t_now, dtype=jnp.float32)
        v1 = apply_fn(params, x, t1)
        x_pred = x + dt * v1

        def corrector(_: None) -> jnp.ndarray:
            t_next = (i.astype(jnp.float32) + 1.0) * dt
            t2 = jnp.full((n,), t_next, dtype=jnp.float32)
            v2 = apply_fn(params, x_pred, t2)
            return x + 0.5 * dt * (v1 + v2)

        return jax.lax.cond(i < n_steps - 1, corrector, lambda _: x_pred, None)

    return jax.lax.fori_loop(0, n_steps, body, init_noise)


@partial(jax.jit, static_argnames=("apply_fn",))
def _sample_euler_one_step_from_noise(
    apply_fn: Callable,
    params: Any,
    init_noise: jnp.ndarray,
) -> jnp.ndarray:
    """JIT-compiled one-step rectified-flow sampler."""
    n = init_noise.shape[0]
    t0 = jnp.zeros((n,), dtype=jnp.float32)
    return init_noise + apply_fn(params, init_noise, t0)


@partial(jax.jit, static_argnames=("apply_fn", "n_steps"))
def _sample_heun_conditional_from_noise(
    apply_fn: Callable,
    params: Any,
    init_noise: jnp.ndarray,
    known_mask: jnp.ndarray,
    known_value: jnp.ndarray,
    n_steps: int,
) -> jnp.ndarray:
    """JIT-compiled conditional Heun sampler from a fixed noise endpoint."""
    n = init_noise.shape[0]
    dt = jnp.float32(1.0 / n_steps)

    def path_at(t_scalar: jnp.ndarray) -> jnp.ndarray:
        return (1.0 - t_scalar) * init_noise + t_scalar * known_value

    def body(i: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        t_now = i.astype(jnp.float32) * dt
        x_constrained = jnp.where(known_mask, path_at(t_now), x)
        t1 = jnp.full((n,), t_now, dtype=jnp.float32)
        v1 = apply_fn(params, x_constrained, t1)
        x_pred = x_constrained + dt * v1

        def corrector(_: None) -> jnp.ndarray:
            t_next = (i.astype(jnp.float32) + 1.0) * dt
            x_pred_constrained = jnp.where(known_mask, path_at(t_next), x_pred)
            t2 = jnp.full((n,), t_next, dtype=jnp.float32)
            v2 = apply_fn(params, x_pred_constrained, t2)
            return x_constrained + 0.5 * dt * (v1 + v2)

        return jax.lax.cond(i < n_steps - 1, corrector, lambda _: x_pred, None)

    x = jax.lax.fori_loop(0, n_steps, body, init_noise)
    return jnp.where(known_mask, known_value, x)


def sample_heun(
    apply_fn: Callable,
    params: Any,
    n_samples: int | None = None,
    spatial_shape: tuple[int, int] | None = None,
    n_channels: int = 1,
    n_steps: int = 30,
    seed: int = 0,
    init_noise: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Heun (2nd-order) ODE integration of dx/dt = v(x, t) from t=0 to t=1.

    Returns ``(n_samples, H, W, C)`` in gaussianized log-y space; convert
    via :func:`agorasynth.data.gaussianized_to_physical` to get physical y.

    If ``init_noise`` is given it is used as the t=0 starting state and
    ``n_samples / spatial_shape`` are ignored. Useful for coherent
    stitching where the same noise must be paired with a constraint.
    """
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    if init_noise is None:
        if spatial_shape is None or n_samples is None:
            raise ValueError("spatial_shape and n_samples are required without init_noise")
        h, w = spatial_shape
        key = jax.random.PRNGKey(seed)
        init_noise = jax.random.normal(
            key, (n_samples, h, w, n_channels), dtype=jnp.float32
        )
    return _sample_heun_from_noise(apply_fn, params, init_noise, n_steps)


def sample_euler_one_step(
    apply_fn: Callable,
    params: Any,
    n_samples: int | None = None,
    spatial_shape: tuple[int, int] | None = None,
    n_channels: int = 1,
    seed: int = 0,
    init_noise: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """One network evaluation: ``x_1 = z + v_theta(z, t=0)``.

    This is the intended fast sampler for checkpoints that have been rectified
    via reflow. It can also be used as a deliberately crude prior from an
    unrectified checkpoint.
    """
    if init_noise is None:
        if spatial_shape is None or n_samples is None:
            raise ValueError("spatial_shape and n_samples are required without init_noise")
        h, w = spatial_shape
        key = jax.random.PRNGKey(seed)
        init_noise = jax.random.normal(
            key, (n_samples, h, w, n_channels), dtype=jnp.float32
        )
    return _sample_euler_one_step_from_noise(apply_fn, params, init_noise)


def sample_heun_conditional(
    apply_fn: Callable,
    params: Any,
    init_noise: jnp.ndarray,
    known_mask: jnp.ndarray,
    known_value: jnp.ndarray,
    n_steps: int = 30,
) -> jnp.ndarray:
    """Repaint-style conditional Heun ODE sampling for flow matching.

    Pixels marked by ``known_mask`` are constrained to track the linear-
    interpolation path from ``init_noise`` to ``known_value``:

        x_t[mask] = (1 - t) * init_noise[mask] + t * known_value[mask]

    At each ODE step, the constraint is enforced *before* every velocity
    evaluation (both predictor and corrector). The model thus sees a
    partially-constrained state and infills the unknown region
    consistently. At t=1 the constraint ensures ``x[mask] == known_value``.

    Used for coherent map stitching: when generating a patch that
    overlaps an already-generated region, the overlap pixels are the
    "known" constraint and the model infills the rest of the patch
    consistently with that boundary -- adjacent patches agree at the
    overlap by construction, not by blending.

    Parameters
    ----------
    init_noise : ``(B, H, W, C)`` array
        Starting state at t=0. Also serves as the noise endpoint of the
        linear path for the known region.
    known_mask : ``(B, H, W, C)`` boolean array
        True where the pixel is constrained.
    known_value : ``(B, H, W, C)`` array
        Target values for the constrained pixels at t=1.
    """
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    return _sample_heun_conditional_from_noise(
        apply_fn, params, init_noise, known_mask, known_value, n_steps
    )
