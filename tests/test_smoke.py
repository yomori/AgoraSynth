"""Fast smoke tests: imports, UNet shape, FM step shapes, ODE sample shape."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
flax = pytest.importorskip("flax")
optax = pytest.importorskip("optax")

from agorasynth.data import gaussianize_patches, gaussianized_to_physical
from agorasynth.flow_matching import (
    FlowMatchingTrainState,
    fm_loss,
    make_fm_only_train_step,
    make_fm_reflow_train_step,
    make_train_step,
    make_train_step_persample,
    make_wph_features_fn,
    sample_euler_one_step,
    sample_heun,
    sample_heun_conditional,
    whitener_from_prior,
    wph_distribution_loss,
    wph_persample_loss,
)
from agorasynth.unet import UNet
from agorasynth.wph import WPHConfig, WPHOp, compute_S_batch, to_real_features


def test_unet_forward_shape():
    model = UNet(channels=(8, 16, 32), t_dim=32, out_channels=1)
    key = jax.random.PRNGKey(0)
    x = jnp.zeros((2, 32, 32, 1))
    t = jnp.zeros((2,))
    params = model.init(key, x, t)
    y = model.apply(params, x, t)
    assert y.shape == x.shape


def test_gaussianize_round_trip():
    rng = np.random.default_rng(0)
    bulk = rng.lognormal(mean=-14.0, sigma=0.3, size=(20, 16, 16))
    bright = rng.lognormal(mean=-7.0, sigma=0.3, size=(20, 16, 16))
    mask = rng.random(size=(20, 16, 16)) < 0.002
    y_in = np.where(mask, bright, bulk).astype(np.float32)
    x, qg, zg = gaussianize_patches(y_in, y0=1e-7, n_quantiles=2048)
    y_back = gaussianized_to_physical(x, qg, zg, y0=1e-7)
    rel_err = np.abs(y_back - y_in) / (np.abs(y_in) + 1e-12)
    assert float(np.median(rel_err)) < 1e-2
    # Marginal of x is approximately N(0, 1).
    assert abs(float(x.mean())) < 0.05
    assert abs(float(x.std()) - 1.0) < 0.1


def test_fm_only_step_runs():
    model = UNet(channels=(8, 16, 32), t_dim=32, out_channels=1)
    key = jax.random.PRNGKey(0)
    dummy_x = jnp.zeros((1, 16, 16, 1))
    dummy_t = jnp.zeros((1,))
    params = model.init(key, dummy_x, dummy_t)
    optimizer = optax.adam(1e-3)
    state = FlowMatchingTrainState.create(
        apply_fn=model.apply, params=params, tx=optimizer
    )
    step = make_fm_only_train_step(model.apply)
    batch = jnp.asarray(np.random.randn(4, 16, 16, 1).astype(np.float32))
    state, loss, _ = step(state, batch, key)
    assert jnp.isfinite(loss)


def test_wph_distribution_loss_zero_for_perfect_match():
    rng = np.random.default_rng(0)
    n_features = 32
    mu = rng.standard_normal(n_features).astype(np.float32)
    A = rng.standard_normal((n_features, n_features)).astype(np.float32)
    cov = (A @ A.T) + 0.1 * np.eye(n_features, dtype=np.float32)
    whitener = whitener_from_prior(mu, cov)
    # Sample a large batch from N(mu, cov); the whitened batch should be ~ N(0, I).
    chol = np.linalg.cholesky(cov)
    F = mu[None, :] + rng.standard_normal((4096, n_features)).astype(np.float32) @ chol.T
    L = wph_distribution_loss(jnp.asarray(F), jnp.asarray(mu), whitener)
    # With per-feature averaging, a B=4096 batch from N(mu, Sigma) gives
    # L ~ O(1/B); even a noisy realization should be << 1.
    assert float(L) < 0.1


def test_full_train_step_with_wph():
    """End-to-end: build a tiny WPH op, fake prior, run one step."""
    M = 16
    cfg = WPHConfig(M=M, N=M, J=3, L=2, dn=0, A=2)
    op = WPHOp.build(cfg)
    wph_features_fn = make_wph_features_fn(op)

    rng = np.random.default_rng(0)
    fake_y = rng.uniform(1e-8, 1e-5, size=(8, M, M)).astype(np.float32)
    feats = np.asarray(wph_features_fn(jnp.asarray(fake_y)))
    mu_prior = feats.mean(axis=0).astype(np.float32)
    cov_prior = (np.cov(feats.T) + 1e-3 * np.eye(feats.shape[1])).astype(np.float32)
    whitener = whitener_from_prior(mu_prior, cov_prior)

    # Tiny dataset: gaussianize a heavy-tailed distribution.
    bulk = rng.lognormal(mean=-14.0, sigma=0.3, size=(8, M, M))
    bright = rng.lognormal(mean=-7.0, sigma=0.3, size=(8, M, M))
    mask = rng.random(size=(8, M, M)) < 0.005
    y_data = np.where(mask, bright, bulk).astype(np.float32)
    x_data, qg, zg = gaussianize_patches(y_data, y0=1e-7, n_quantiles=512)
    x_data_nhwc = jnp.asarray(x_data[:, :, :, None])

    model = UNet(channels=(8, 16), t_dim=16, bottleneck_blocks=1, out_channels=1)
    key = jax.random.PRNGKey(0)
    params = model.init(
        key, jnp.zeros((1, M, M, 1)), jnp.zeros((1,))
    )
    optimizer = optax.adam(1e-3)
    state = FlowMatchingTrainState.create(
        apply_fn=model.apply, params=params, tx=optimizer
    )
    step = make_train_step(
        apply_fn=model.apply,
        wph_features_fn=wph_features_fn,
        mu_prior=mu_prior, whitener=whitener,
        z_grid=zg, quantile_grid=qg, y0=1e-7,
        lambda_wph=0.1, wph_t_min=0.5,
    )
    state, loss, l_fm, l_wph, _ = step(state, x_data_nhwc, key)
    assert jnp.isfinite(loss)
    assert jnp.isfinite(l_fm)
    assert jnp.isfinite(l_wph)


def test_persample_train_step_runs():
    """End-to-end: per-sample WPH loss train step, one batch."""
    M = 16
    cfg = WPHConfig(M=M, N=M, J=3, L=2, dn=0, A=2)
    op = WPHOp.build(cfg)
    wph_features_fn = make_wph_features_fn(op)

    rng = np.random.default_rng(0)
    bulk = rng.lognormal(mean=-14.0, sigma=0.3, size=(8, M, M))
    bright = rng.lognormal(mean=-7.0, sigma=0.3, size=(8, M, M))
    mask = rng.random(size=(8, M, M)) < 0.005
    y_data = np.where(mask, bright, bulk).astype(np.float32)
    x_data, qg, zg = gaussianize_patches(y_data, y0=1e-7, n_quantiles=512)

    # Precompute WPH targets and inv_std (small dataset, just for shape).
    F_targets = np.asarray(wph_features_fn(jnp.asarray(y_data)))
    std_per_feat = np.maximum(F_targets.std(axis=0), 1e-6)
    inv_std = (1.0 / std_per_feat).astype(np.float32)

    model = UNet(channels=(8, 16), t_dim=16, bottleneck_blocks=1, out_channels=1)
    key = jax.random.PRNGKey(0)
    params = model.init(key, jnp.zeros((1, M, M, 1)), jnp.zeros((1,)))
    optimizer = optax.adam(1e-3)
    state = FlowMatchingTrainState.create(
        apply_fn=model.apply, params=params, tx=optimizer
    )
    step = make_train_step_persample(
        apply_fn=model.apply,
        wph_features_fn=wph_features_fn,
        inv_std_per_feature=inv_std,
        z_grid=zg, quantile_grid=qg, y0=1e-7,
        lambda_wph=0.1, wph_t_min=0.5,
    )
    state, loss, l_fm, l_wph, _ = step(
        state,
        jnp.asarray(x_data[:, :, :, None]),
        jnp.asarray(F_targets),
        key,
    )
    assert jnp.isfinite(loss)
    assert jnp.isfinite(l_fm)
    assert jnp.isfinite(l_wph)


def test_heun_sampling_shape():
    model = UNet(channels=(8, 16), t_dim=16, bottleneck_blocks=1, out_channels=1)
    key = jax.random.PRNGKey(0)
    params = model.init(key, jnp.zeros((1, 16, 16, 1)), jnp.zeros((1,)))
    samples = sample_heun(
        model.apply, params,
        n_samples=2, spatial_shape=(16, 16), n_channels=1, n_steps=4, seed=0,
    )
    assert samples.shape == (2, 16, 16, 1)
    assert jnp.isfinite(samples).all()


def test_euler_one_step_sampling_shape():
    model = UNet(channels=(8, 16), t_dim=16, bottleneck_blocks=1, out_channels=1)
    key = jax.random.PRNGKey(0)
    params = model.init(key, jnp.zeros((1, 16, 16, 1)), jnp.zeros((1,)))
    samples = sample_euler_one_step(
        model.apply, params,
        n_samples=2, spatial_shape=(16, 16), n_channels=1, seed=0,
    )
    assert samples.shape == (2, 16, 16, 1)
    assert jnp.isfinite(samples).all()


def test_reflow_train_step_runs():
    """Paired-data FM train step (rectified-flow reflow)."""
    model = UNet(channels=(8, 16), t_dim=16, bottleneck_blocks=1, out_channels=1)
    key = jax.random.PRNGKey(0)
    params = model.init(key, jnp.zeros((1, 16, 16, 1)), jnp.zeros((1,)))
    optimizer = optax.adam(1e-3)
    state = FlowMatchingTrainState.create(
        apply_fn=model.apply, params=params, tx=optimizer
    )
    step = make_fm_reflow_train_step(model.apply)
    rng = np.random.default_rng(0)
    z = jnp.asarray(rng.standard_normal((4, 16, 16, 1)).astype(np.float32))
    x = jnp.asarray(rng.standard_normal((4, 16, 16, 1)).astype(np.float32))
    state, loss, _ = step(state, x, z, key)
    assert jnp.isfinite(loss)


def test_heun_conditional_respects_known_region():
    """sample_heun_conditional must produce x[mask] == known_value at t=1."""
    model = UNet(channels=(8, 16), t_dim=16, bottleneck_blocks=1, out_channels=1)
    key = jax.random.PRNGKey(0)
    params = model.init(key, jnp.zeros((1, 16, 16, 1)), jnp.zeros((1,)))
    rng = np.random.default_rng(0)
    init_noise = jnp.asarray(rng.standard_normal((2, 16, 16, 1)).astype(np.float32))
    known_value = jnp.asarray(rng.standard_normal((2, 16, 16, 1)).astype(np.float32))
    # Constrain a left half-mask
    mask_np = np.zeros((2, 16, 16, 1), dtype=bool)
    mask_np[:, :, :8, :] = True
    known_mask = jnp.asarray(mask_np)
    out = sample_heun_conditional(
        model.apply, params,
        init_noise=init_noise,
        known_mask=known_mask,
        known_value=known_value,
        n_steps=4,
    )
    assert out.shape == (2, 16, 16, 1)
    assert jnp.isfinite(out).all()
    # Known region must equal known_value exactly.
    np.testing.assert_allclose(
        np.asarray(out)[mask_np], np.asarray(known_value)[mask_np], atol=1e-5
    )
