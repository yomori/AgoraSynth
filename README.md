# AgoraSynth

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

Amortized non-Gaussian Compton-y **and CIB** synthesis: rectified flow matching with a
WPH-feature batch-distribution loss. Trained once, samples in one ODE
integration (~30 NFE).

The pipeline:

1. Extract ZEA patches from a HEALPix Compton-y map and Gaussianize via a
   per-pixel rank/quantile transform so the marginal is exactly N(0, 1).
2. Fit a Gaussian prior `(mu, Sigma)` on the real-valued WPH feature vector
   computed from the same patches in physical-y space.
3. Train a time-conditional U-Net on flow matching with two losses:
   - `L_FM`: rectified-flow regression on linear-interpolation paths.
   - `L_WPH`: whitened batch-distribution match between the model's
     predicted clean sample (run through the WPH operator in physical-y
     space) and the prior. Gated by `sigmoid(20*(t - t_min))` so it kicks
     in only when the predicted clean field is reliable.
4. Sample by Heun (2nd-order) ODE integration of the learned velocity
   field from t=0 (noise) to t=1 (data). Invert the rank transform to
   recover physical y.

WPH is implemented in JAX (copied from `diffusiontsz/`) so the entire
training graph — including the WPH operator — sits inside `jax.jit` and
gradients flow cleanly through the inverse Gaussianization and through
the WPH feature computation.

## Components

- **Compton-y** (single channel) — runbook [`RUN`](RUN).
- **CIB** (joint SPT-3G 95/150/220 GHz; one U-Net emits all three bands from
  shared noise, with per-channel + cross-band WPH features so inter-band
  correlation is preserved) — runbook [`RUN_CIB`](RUN_CIB).

## Installation

```bash
git clone https://github.com/yomori/AgoraSynth.git
cd AgoraSynth
pip install -e ".[dev,maps,viz]"
```

`jax`/`jaxlib` install CPU wheels by default; for GPU follow the
[JAX install guide](https://jax.readthedocs.io/en/latest/installation.html).

## Development

```bash
pip install -e ".[dev]"
pre-commit install
JAX_PLATFORMS=cpu pytest        # CPU-only smoke tests
```

## License

MIT — see [LICENSE](LICENSE).
