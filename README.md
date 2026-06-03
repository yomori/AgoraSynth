# AgoraSynth

Amortized non-Gaussian Compton-y synthesis: rectified flow matching with a
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

See [`RUN`](RUN) for the operational runbook.
