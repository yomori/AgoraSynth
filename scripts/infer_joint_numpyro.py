"""Joint tSZ + CIB posterior sampling with NumPyro NUTS over few-step flows.

Latents z_tsz (1,H,W,1) and z_cib (1,H,W,3) ~ N(0,I) are pushed through their
few-step Euler flows (agorasynth.inference) and compared to data. NUTS
differentiates through the flows once per leapfrog; the `--grad-steps` knob
(1..nfe) trades proposal fidelity for speed+memory while keeping the target
EXACT (the flow VALUE is always exact -- NUTS's accept/energy use it).

The SELF-TEST (default, mock data) runs in the model's native, well-conditioned
GAUSSIANIZED space (fields ~O(1)) so NUTS actually mixes -- it validates the
flow + custom_vjp + NUTS machinery (gradient exactness, per-leapfrog cost,
peak memory, accept rate). For REAL inference, use `physical_observation()`
below: physical maps combined into the 95/150/220 bands with the true SED,
beam, and PER-PIXEL instrument noise. WARNING: a single fixed sigma on physical
CIB intensities (which span ~1e3-1e6, i.e. 3 dex) is badly ill-conditioned for
HMC -- use the real per-pixel noise; the bright-source dynamic range is the
main sampling challenge, and a brightness-aware reparameterization may help.
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpyro  # noqa: E402
import numpyro.distributions as dist  # noqa: E402
from numpyro.infer import MCMC, NUTS  # noqa: E402

from agorasynth.inference import make_few_step_flow, make_physical_flow  # noqa: E402
from agorasynth.unet import UNet  # noqa: E402

# Placeholder tSZ -> per-band scaling (REPLACE with the real SZ spectral
# function g(nu) in your units). 220 GHz is near the tSZ null.
TSZ_BAND_COEFF = jnp.asarray([-1.5, -1.0, 0.1], jnp.float32)


def physical_observation(z_cib, z_tsz, cib_model, tsz_model, *, nfe=4, grad_steps=1):
    """REAL-DATA observation operator (template). ``cib_model``/``tsz_model`` are
    ``(apply_fn, params, quantile_grid, z_grid, y0)`` tuples. Builds physical-space
    few-step flows and combines them into the 95/150/220 bands. Replace
    TSZ_BAND_COEFF with the true SED, add the beam, and use the real per-pixel
    noise in the likelihood (NOT a single fixed sigma)."""
    cib_flow = make_physical_flow(*cib_model, nfe=nfe, grad_steps=grad_steps)
    tsz_flow = make_physical_flow(*tsz_model, nfe=nfe, grad_steps=grad_steps)
    return cib_flow(z_cib) + TSZ_BAND_COEFF[None, None, None, :] * tsz_flow(z_tsz)


def _load(ckpt: Path):
    with open(ckpt, "rb") as f:
        ck = pickle.load(f)
    cfg = ck["config"]
    model = UNet(channels=tuple(cfg["channels"]), t_dim=cfg["t_dim"],
                 bottleneck_blocks=cfg["bottleneck_blocks"], out_channels=int(cfg["out_channels"]))
    return model.apply, ck["params"], int(cfg["out_channels"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cib-ckpt", type=Path, default=Path("checkpoints/fmonly_val_cib.pkl"))
    ap.add_argument("--tsz-ckpt", type=Path, default=Path("checkpoints/fm_rectified.pkl"))
    ap.add_argument("--H", type=int, default=512)
    ap.add_argument("--W", type=int, default=512)
    ap.add_argument("--nfe", type=int, default=4)
    ap.add_argument("--grad-steps", type=int, default=1)
    ap.add_argument("--n-warmup", type=int, default=20)
    ap.add_argument("--n-samples", type=int, default=20)
    ap.add_argument("--max-tree-depth", type=int, default=7)
    ap.add_argument("--sigma-frac", type=float, default=0.3,
                    help="Mock noise std as a fraction of the field std (gaussianized self-test).")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    H, W, nfe, gs = args.H, args.W, args.nfe, args.grad_steps
    dev = jax.devices()[0]
    print(f"dev={dev.device_kind}  {H}x{W} ({H*W/1e6:.2f} Mpix)  nfe={nfe} grad_steps={gs}")

    cib_apply, cib_p, cib_c = _load(args.cib_ckpt)
    tsz_apply, tsz_p, tsz_c = _load(args.tsz_ckpt)
    assert cib_c == 3 and tsz_c == 1, f"expected CIB out=3, tSZ out=1; got {cib_c},{tsz_c}"

    # Gaussianized-space flows (O(1), well-conditioned) for the machinery self-test.
    cib_flow = make_few_step_flow(cib_apply, cib_p, nfe=nfe, grad_steps=gs)
    tsz_flow = make_few_step_flow(tsz_apply, tsz_p, nfe=nfe, grad_steps=gs)

    # ---- mock data: a fixed draw + noise scaled to the field std ----
    k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(args.seed), 4)
    zc_true = jax.random.normal(k1, (1, H, W, 3))
    zt_true = jax.random.normal(k2, (1, H, W, 1))
    xc_true, xt_true = cib_flow(zc_true), tsz_flow(zt_true)
    sig_c = args.sigma_frac * float(jnp.std(xc_true))
    sig_t = args.sigma_frac * float(jnp.std(xt_true))
    data_c = xc_true + sig_c * jax.random.normal(k3, (1, H, W, 3))
    data_t = xt_true + sig_t * jax.random.normal(k4, (1, H, W, 1))

    # ---------- custom_vjp gradient self-check (value exact? grad cos vs exact) ----------
    z0c, z0t = zc_true * 0.7, zt_true * 0.7
    def potential(zc, zt):
        return (0.5 * jnp.sum((cib_flow(zc) - data_c) ** 2) / sig_c ** 2
                + 0.5 * jnp.sum((tsz_flow(zt) - data_t) ** 2) / sig_t ** 2)
    val_g, grad_g = jax.value_and_grad(potential, argnums=(0, 1))(z0c, z0t)
    cib_ex = make_few_step_flow(cib_apply, cib_p, nfe=nfe, grad_steps=nfe)
    tsz_ex = make_few_step_flow(tsz_apply, tsz_p, nfe=nfe, grad_steps=nfe)
    def potential_ex(zc, zt):
        return (0.5 * jnp.sum((cib_ex(zc) - data_c) ** 2) / sig_c ** 2
                + 0.5 * jnp.sum((tsz_ex(zt) - data_t) ** 2) / sig_t ** 2)
    val_e, grad_e = jax.value_and_grad(potential_ex, argnums=(0, 1))(z0c, z0t)
    gc = jnp.concatenate([grad_g[0].ravel(), grad_g[1].ravel()])
    ge = jnp.concatenate([grad_e[0].ravel(), grad_e[1].ravel()])
    cos = float(gc @ ge / (jnp.linalg.norm(gc) * jnp.linalg.norm(ge) + 1e-30))
    gratio = float(jnp.linalg.norm(gc) / (jnp.linalg.norm(ge) + 1e-30))
    print(f"  [self-check] value exact match: {bool(jnp.allclose(val_g, val_e))} "
          f"(|Δ|={float(abs(val_g - val_e)):.2e}) | "
          f"grad cosine(gs={gs},exact)={cos:.4f} |g|ratio={gratio:.3f}")

    # ---------- NUTS (gaussianized self-test) ----------
    def model(data_c, data_t):
        zc = numpyro.sample("z_cib", dist.Normal(0., 1.).expand([1, H, W, 3]).to_event(4))
        zt = numpyro.sample("z_tsz", dist.Normal(0., 1.).expand([1, H, W, 1]).to_event(4))
        numpyro.sample("obs_cib", dist.Normal(cib_flow(zc), sig_c).to_event(4), obs=data_c)
        numpyro.sample("obs_tsz", dist.Normal(tsz_flow(zt), sig_t).to_event(4), obs=data_t)

    mcmc = MCMC(NUTS(model, max_tree_depth=args.max_tree_depth),
                num_warmup=args.n_warmup, num_samples=args.n_samples,
                num_chains=1, progress_bar=False)
    t0 = time.time()
    mcmc.run(jax.random.PRNGKey(args.seed + 1), data_c=data_c, data_t=data_t,
             extra_fields=("num_steps", "accept_prob", "diverging"))
    dt = time.time() - t0
    ef = mcmc.get_extra_fields()
    steps = np.asarray(ef["num_steps"])
    acc = np.asarray(ef["accept_prob"])
    div = int(np.asarray(ef["diverging"]).sum())
    peak = (dev.memory_stats() or {}).get("peak_bytes_in_use", 0) / 1e9
    tot_lf = int(steps.sum())
    med = int(np.median(steps))
    print(f"  NUTS: {args.n_warmup}+{args.n_samples} in {dt:.1f}s | "
          f"leapfrog/sample med={med} max={steps.max()} | "
          f"accept={acc.mean():.2f} | diverging={div}")
    print(f"  -> ~{dt / max(args.n_samples, 1):.2f} s/sample, "
          f"~{1000 * dt / max(tot_lf, 1):.0f} ms/leapfrog (gross), peak={peak:.1f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
