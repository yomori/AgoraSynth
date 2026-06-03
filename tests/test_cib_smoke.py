"""Smoke + correctness tests for the multi-channel (CIB) pipeline.

Tiny sizes; CPU. Run with:  JAX_PLATFORMS=cpu PYTHONPATH=src pytest tests/test_cib_smoke.py -q
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("flax")
optax = pytest.importorskip("optax")

from agorasynth.data import (
    gaussianize_patches_multi,
    gaussianized_to_physical_multi,
)
from agorasynth.flow_matching import (
    FlowMatchingTrainState,
    channel_pairs,
    make_train_step_persample_multi,
    make_wph_features_multi_fn,
    sample_heun,
)
from agorasynth.unet import UNet
from agorasynth.wph import (
    WPHConfig,
    WPHOp,
    compute_S,
    compute_S_cross,
)


def _positive_stack(rng, n, c, m):
    """Random strictly-positive multi-channel patches (CIB-like, correlated bands)."""
    base = rng.lognormal(mean=6.0, sigma=0.4, size=(n, 1, m, m))
    scale = np.array([1.0, 2.5, 6.0])[:c].reshape(1, c, 1, 1)
    noise = rng.lognormal(mean=0.0, sigma=0.1, size=(n, c, m, m))
    return (base * scale * noise).astype(np.float32)


def test_gaussianize_multi_round_trip():
    rng = np.random.default_rng(0)
    patches = _positive_stack(rng, n=24, c=3, m=16)             # (N, C, H, W)
    x, qg, zg = gaussianize_patches_multi(patches, y0=0.0, n_quantiles=512)
    assert x.shape == patches.shape
    assert qg.shape == (3, 512)
    # Per-channel marginal ~ N(0, 1).
    for ch in range(3):
        assert abs(float(x[:, ch].mean())) < 0.05
        assert abs(float(x[:, ch].std()) - 1.0) < 0.1
    # Round trip (channel-last).
    x_nhwc = np.transpose(x, (0, 2, 3, 1))
    y_back = np.asarray(gaussianized_to_physical_multi(x_nhwc, qg, zg, y0=0.0))
    y_true = np.transpose(patches, (0, 2, 3, 1))
    rel = np.abs(y_back - y_true) / (np.abs(y_true) + 1e-6)
    assert float(np.median(rel)) < 1e-2


def test_cross_wph_equals_auto_for_identical_fields():
    """compute_S_cross(op, x, x) must reproduce the auto compute_S(op, x)."""
    rng = np.random.default_rng(1)
    x = jnp.asarray(rng.standard_normal((16, 16)).astype(np.float32))
    op = WPHOp.build(WPHConfig(M=16, N=16, J=3, L=2))
    s_auto = np.asarray(compute_S(op, x))
    s_cross = np.asarray(compute_S_cross(op, x, x))
    assert s_auto.shape == s_cross.shape
    assert np.allclose(s_auto, s_cross, rtol=1e-4, atol=1e-5)


def test_cross_wph_differs_for_different_fields():
    rng = np.random.default_rng(2)
    xa = jnp.asarray(rng.standard_normal((16, 16)).astype(np.float32))
    xb = jnp.asarray(rng.standard_normal((16, 16)).astype(np.float32))
    op = WPHOp.build(WPHConfig(M=16, N=16, J=3, L=2))
    s_cross = np.asarray(compute_S_cross(op, xa, xb))
    s_auto_a = np.asarray(compute_S(op, xa))
    assert not np.allclose(s_cross, s_auto_a)


def test_multi_features_shape():
    rng = np.random.default_rng(3)
    c = 3
    op = WPHOp.build(WPHConfig(M=16, N=16, J=3, L=2))
    feats_fn = make_wph_features_multi_fn(op, n_channels=c, checkpoint=False)
    y = jnp.asarray(rng.standard_normal((2, 16, 16, c)).astype(np.float32))
    F = np.asarray(feats_fn(y))
    n_blocks = c + len(channel_pairs(c))                        # 3 auto + 3 cross
    assert n_blocks == 6
    assert F.shape == (2, n_blocks * 2 * op.n_total)


def test_unet_multichannel_sample_shape():
    model = UNet(channels=(8, 16), t_dim=16, out_channels=3)
    key = jax.random.PRNGKey(0)
    params = model.init(key, jnp.zeros((1, 16, 16, 3)), jnp.zeros((1,)))
    out = sample_heun(model.apply, params, n_samples=2,
                      spatial_shape=(16, 16), n_channels=3, n_steps=2, seed=0)
    assert out.shape == (2, 16, 16, 3)


def test_persample_multi_train_step_runs():
    rng = np.random.default_rng(4)
    c = 3
    patches = _positive_stack(rng, n=8, c=c, m=16)
    x, qg, zg = gaussianize_patches_multi(patches, y0=0.0, n_quantiles=256)
    x_nhwc = jnp.asarray(np.transpose(x, (0, 2, 3, 1)))         # (N, H, W, C)

    op = WPHOp.build(WPHConfig(M=16, N=16, J=3, L=2))
    feats_fn = make_wph_features_multi_fn(op, n_channels=c, checkpoint=True)
    n_feat = (c + len(channel_pairs(c))) * 2 * op.n_total

    model = UNet(channels=(8, 16), t_dim=16, out_channels=c)
    key = jax.random.PRNGKey(0)
    params = model.init(key, jnp.zeros((1, 16, 16, c)), jnp.zeros((1,)))
    state = FlowMatchingTrainState.create(
        apply_fn=model.apply, params=params, tx=optax.adam(1e-3)
    )
    step = make_train_step_persample_multi(
        apply_fn=model.apply, wph_features_fn=feats_fn,
        inv_std_per_feature=jnp.ones((n_feat,), jnp.float32),
        z_grid=zg, quantile_grid=qg, y0=np.zeros(c), n_channels=c,
        lambda_wph=1.0, wph_t_min=0.5, lambda_warmup_steps=0,
    )
    F_target = jnp.asarray(rng.standard_normal((8, n_feat)).astype(np.float32))
    state, loss, l_fm, l_wph, key = step(state, x_nhwc, F_target, key)
    assert jnp.isfinite(loss)
    assert jnp.isfinite(l_fm) and jnp.isfinite(l_wph)
