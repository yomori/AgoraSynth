"""Differentiable few-step flow for HMC / NumPyro posterior sampling.

For posterior sampling the generative map must be ONE differentiable
``z -> map`` function evaluated (with gradient) once per leapfrog step. We use
an UNROLLED few-step Euler integrator -- NOT ``lax.fori_loop``, whose reverse
-mode AD OOMs/errors -- with an optional ``custom_vjp`` ``grad_steps`` knob:

  * the forward VALUE is always the exact ``nfe``-step Euler result;
  * the backward differentiates through only the LAST ``grad_steps`` steps,
    treating the earlier steps' Jacobian as identity (``dx_split/dz ~= I``).

``grad_steps == nfe`` -> exact reverse-mode gradient. ``grad_steps < nfe`` ->
a cheaper, lower-memory, *approximate* gradient. This is valid inside HMC/NUTS
because the accept step / trajectory energies use the exact potential VALUE
(the ``custom_vjp`` primal) and leapfrog is reversible & volume-preserving for
ANY force field -- so the stationary distribution stays EXACT; only proposal
efficiency (accept rate / ESS) changes. Rationale: for a (reflowed) rectified
flow the paths are nearly straight, so ``dx/dz ~= I + small`` and the
last-step VJP captures almost all of the true Jacobian.

Euler (not Heun) is used on purpose: reflow straightens the paths, so Euler is
the standard, cheapest choice and a higher-order corrector buys little.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp


def make_few_step_flow(
    apply_fn: Callable,
    params: Any,
    *,
    nfe: int = 4,
    grad_steps: int = 1,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Return ``flow(z) -> x`` : ``nfe``-step Euler integration in gaussianized
    space, with a ``custom_vjp`` that backprops through the last ``grad_steps``
    steps only (exact value; approximate gradient when ``grad_steps < nfe``).

    ``z`` and the returned ``x`` are ``(B, H, W, C)`` (gaussianized log-field).
    """
    if not (1 <= grad_steps <= nfe):
        raise ValueError(f"need 1 <= grad_steps <= nfe; got grad_steps={grad_steps}, nfe={nfe}")
    dt = jnp.float32(1.0 / nfe)

    def step(x: jnp.ndarray, i: int) -> jnp.ndarray:
        t = jnp.full((x.shape[0],), i * dt, jnp.float32)
        return x + dt * apply_fn(params, x, t)

    def full(z: jnp.ndarray) -> jnp.ndarray:
        x = z
        for i in range(nfe):
            x = step(x, i)
        return x

    if grad_steps == nfe:
        return full  # plain reverse-mode AD == exact gradient

    n_const = nfe - grad_steps  # leading steps excluded from the backward

    @jax.custom_vjp
    def flow(z: jnp.ndarray) -> jnp.ndarray:
        return full(z)

    def flow_fwd(z: jnp.ndarray):
        x = z
        for i in range(n_const):           # leading steps: value only, no AD state kept
            x = step(x, i)
        x_split = x
        y = x_split
        for i in range(n_const, nfe):
            y = step(y, i)
        return y, (x_split,)               # save only the split state (~one field)

    def flow_bwd(res, g):
        (x_split,) = res

        def tail(x: jnp.ndarray) -> jnp.ndarray:
            y = x
            for i in range(n_const, nfe):
                y = step(y, i)
            return y

        _, vjp = jax.vjp(tail, x_split)
        # Approximate dL/dz <- dL/dx_split  (treats d x_split / d z as identity).
        return (vjp(g)[0],)

    flow.defvjp(flow_fwd, flow_bwd)
    return flow


def make_physical_flow(
    apply_fn: Callable,
    params: Any,
    quantile_grid,
    z_grid,
    y0,
    *,
    nfe: int = 4,
    grad_steps: int = 1,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """``z -> physical map``: few-step flow then the per-band inverse rank
    transform (differentiable ``jnp.interp``). The ``custom_vjp`` lives on the
    flow; the inversion is exact AD on top (cheap)."""
    from agorasynth.data import gaussianized_to_physical_multi

    flow = make_few_step_flow(apply_fn, params, nfe=nfe, grad_steps=grad_steps)
    qg = jnp.asarray(quantile_grid, jnp.float32)
    zg = jnp.asarray(z_grid, jnp.float32)
    y0a = jnp.asarray(y0, jnp.float32)

    def physical(z: jnp.ndarray) -> jnp.ndarray:
        x = flow(z)                                   # (B,H,W,C) gaussianized
        return gaussianized_to_physical_multi(x, qg, zg, y0a)

    return physical
