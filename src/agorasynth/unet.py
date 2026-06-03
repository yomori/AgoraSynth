"""Time-conditional U-Net (Flax) for flow-matching velocity prediction.

NHWC layout (Flax convention). Input ``x`` has shape ``(B, H, W, C_in)`` and
``t`` has shape ``(B,)`` with values in [0, 1]; output has the same spatial
shape and ``C_in`` channels (predicting the velocity ``dx/dt``).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal embedding of t in [0, 1] -> (B, dim)."""

    dim: int

    @nn.compact
    def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
        if self.dim % 2 != 0:
            raise ValueError(f"dim must be even, got {self.dim}")
        half = self.dim // 2
        freqs = jnp.exp(
            -math.log(10_000.0) * jnp.arange(half, dtype=jnp.float32) / max(half - 1, 1)
        )
        # Scale t to roughly the range used in DDPM/EDM sinusoidal embeddings.
        t_scaled = jnp.asarray(t, dtype=jnp.float32) * 1000.0
        args = t_scaled[:, None] * freqs[None, :]
        return jnp.concatenate([jnp.sin(args), jnp.cos(args)], axis=-1)


class TimeMLP(nn.Module):
    """Two-layer MLP on top of the sinusoidal embedding."""

    dim: int

    @nn.compact
    def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
        h = SinusoidalTimeEmbedding(self.dim)(t)
        h = nn.Dense(self.dim * 4)(h)
        h = nn.silu(h)
        h = nn.Dense(self.dim)(h)
        return h


class ResBlock(nn.Module):
    """Pre-activation ResBlock with t-conditioning as a per-channel bias."""

    out_ch: int
    n_groups: int = 8

    @nn.compact
    def __call__(self, x: jnp.ndarray, t_emb: jnp.ndarray) -> jnp.ndarray:
        in_ch = x.shape[-1]
        h = nn.GroupNorm(num_groups=min(self.n_groups, in_ch))(x)
        h = nn.silu(h)
        h = nn.Conv(self.out_ch, (3, 3), padding="SAME")(h)
        # Inject time embedding as channel-wise bias.
        t_proj = nn.Dense(self.out_ch)(nn.silu(t_emb))
        h = h + t_proj[:, None, None, :]
        h = nn.GroupNorm(num_groups=min(self.n_groups, self.out_ch))(h)
        h = nn.silu(h)
        h = nn.Conv(self.out_ch, (3, 3), padding="SAME")(h)
        if in_ch != self.out_ch:
            x = nn.Conv(self.out_ch, (1, 1), padding="SAME")(x)
        return x + h


class UNet(nn.Module):
    """Time-conditional U-Net.

    Parameters
    ----------
    channels
        Channels at each encoder level. Number of levels = ``len(channels)``;
        spatial divisor is ``2 ** len(channels)``.
    t_dim
        Width of the t embedding and conditioning features.
    bottleneck_blocks
        Number of ResBlocks at the lowest resolution.
    out_channels
        Output channels (1 for single-channel y patches).
    """

    channels: Sequence[int] = (32, 64, 128)
    t_dim: int = 128
    bottleneck_blocks: int = 2
    out_channels: int = 1

    @property
    def divisor(self) -> int:
        return 2 ** len(self.channels)

    @nn.compact
    def __call__(self, x: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        h_orig, w_orig = x.shape[1], x.shape[2]
        h_pad = (-h_orig) % self.divisor
        w_pad = (-w_orig) % self.divisor
        if h_pad or w_pad:
            x = jnp.pad(
                x, ((0, 0), (0, h_pad), (0, w_pad), (0, 0)), mode="reflect"
            )

        t_emb = TimeMLP(self.t_dim)(t)
        h = nn.Conv(self.channels[0], (3, 3), padding="SAME")(x)

        skips = []
        for ch in self.channels:
            h = ResBlock(ch)(h, t_emb)
            skips.append(h)
            h = nn.Conv(ch, (3, 3), strides=(2, 2), padding="SAME")(h)

        for _ in range(self.bottleneck_blocks):
            h = ResBlock(self.channels[-1])(h, t_emb)

        for skip, ch in zip(reversed(skips), reversed(self.channels), strict=False):
            new_shape = (h.shape[0], h.shape[1] * 2, h.shape[2] * 2, h.shape[3])
            h = jax.image.resize(h, new_shape, method="nearest")
            h = nn.Conv(ch, (3, 3), padding="SAME")(h)
            h = jnp.concatenate([h, skip], axis=-1)
            h = ResBlock(ch)(h, t_emb)

        out = nn.Conv(self.out_channels, (3, 3), padding="SAME")(h)
        if h_pad or w_pad:
            out = out[:, :h_orig, :w_orig, :]
        return out
