"""Lazy backend array-operation facade shared by built-in and custom restraints.

Geometry is implemented once against this small interface.  Only framework spelling
differences live here, and torch/jax remain lazy imports so importing :mod:`rgi_utils`
still requires NumPy only.
"""

from __future__ import annotations

EPS = 1e-12


class _AxisOps:
    """Operations shared by NumPy and JAX (both use ``axis=``)."""

    def __init__(self, xp, int_dtype, wrap, stop_grad):
        self.xp = xp
        self._int = int_dtype
        self._wrap = wrap
        self._stop_grad = stop_grad

    def gather(self, coords, idx):
        return coords[..., idx, :]

    def asint(self, value):
        return self.xp.asarray(value, dtype=self._int)

    def const(self, value):
        return self._wrap(value)

    def const_like(self, value, like):
        return self.xp.asarray(value, dtype=like.dtype)

    def scalar_like(self, value, like):
        return self.xp.asarray(value, dtype=like.dtype)

    def astype_like(self, value, like):
        return value.astype(like.dtype)

    def mean_atoms(self, value):
        return self.xp.mean(value, axis=-2)

    def mean_last(self, value):
        return self.xp.mean(value, axis=-1)

    def mean(self, value, axis=None, keepdims=False):
        return self.xp.mean(value, axis=axis, keepdims=keepdims)

    def sum_atoms(self, value):
        return self.xp.sum(value, axis=-2)

    def sum(self, value, axis=None, keepdims=False):
        return self.xp.sum(value, axis=axis, keepdims=keepdims)

    def concat_atoms(self, blocks):
        return self.xp.concatenate(blocks, axis=-2)

    def vdot(self, first, second):
        return self.xp.sum(first * second, axis=-1)

    def vnorm(self, value):
        return self.xp.sqrt(self.xp.sum(value * value, axis=-1) + EPS)

    def cross(self, first, second):
        return self.xp.cross(first, second)

    def matmul(self, first, second):
        return first @ second

    def swapaxes_last2(self, value):
        return self.xp.swapaxes(value, -1, -2)

    def svd(self, value):
        return self.xp.linalg.svd(value)

    def eigh(self, value):
        return self.xp.linalg.eigh(value)

    def det(self, value):
        return self.xp.linalg.det(value)

    def sign(self, value):
        return self.xp.sign(value)

    def stack(self, arrays, axis=-1):
        return self.xp.stack(arrays, axis=axis)

    def ones_like(self, value):
        return self.xp.ones_like(value)

    def stop_gradient(self, value):
        return self._stop_grad(value)

    def arccos(self, value):
        return self.xp.arccos(value)

    def arctan2(self, first, second):
        return self.xp.arctan2(first, second)

    def sqrt(self, value):
        return self.xp.sqrt(value)

    def exp(self, value):
        return self.xp.exp(value)

    def log(self, value):
        return self.xp.log(value)

    def abs(self, value):
        return self.xp.abs(value)

    def sin(self, value):
        return self.xp.sin(value)

    def cos(self, value):
        return self.xp.cos(value)

    def clip(self, value, lo, hi):
        return self.xp.clip(value, lo, hi)

    def minimum(self, first, second):
        return self.xp.minimum(first, second)

    def maximum(self, first, second):
        return self.xp.maximum(first, second)

    def clamp_min(self, value, lo):
        return self.xp.maximum(value, lo)

    def clamp_max(self, value, hi):
        return self.xp.minimum(value, hi)

    def where(self, condition, first, second):
        return self.xp.where(condition, first, second)


class _TorchOps:
    """Torch implementation of the shared array-operation interface."""

    def __init__(self, torch, device=None):
        self.t = torch
        self._device = device

    def gather(self, coords, idx):
        return coords[..., idx, :]

    def asint(self, value):
        return self.t.as_tensor(value, dtype=self.t.long, device=self._device)

    def const(self, value):
        return value

    def const_like(self, value, like):
        return self.t.as_tensor(value, dtype=like.dtype, device=like.device)

    def scalar_like(self, value, like):
        return self.t.as_tensor(value, dtype=like.dtype, device=like.device)

    def astype_like(self, value, like):
        return value.to(like.dtype)

    def mean_atoms(self, value):
        return self.t.mean(value, dim=-2)

    def mean_last(self, value):
        return self.t.mean(value, dim=-1)

    def mean(self, value, axis=None, keepdims=False):
        return self.t.mean(value, dim=axis, keepdim=keepdims)

    def sum_atoms(self, value):
        return self.t.sum(value, dim=-2)

    def sum(self, value, axis=None, keepdims=False):
        return self.t.sum(value, dim=axis, keepdim=keepdims)

    def concat_atoms(self, blocks):
        return self.t.cat(blocks, dim=-2)

    def vdot(self, first, second):
        return self.t.sum(first * second, dim=-1)

    def vnorm(self, value):
        return self.t.sqrt(self.t.sum(value * value, dim=-1) + EPS)

    def cross(self, first, second):
        return self.t.linalg.cross(first, second, dim=-1)

    def matmul(self, first, second):
        return first @ second

    def swapaxes_last2(self, value):
        return self.t.swapaxes(value, -1, -2)

    def svd(self, value):
        return self.t.linalg.svd(value)

    def eigh(self, value):
        return self.t.linalg.eigh(value)

    def det(self, value):
        return self.t.linalg.det(value)

    def sign(self, value):
        return self.t.sign(value)

    def stack(self, arrays, axis=-1):
        return self.t.stack(arrays, dim=axis)

    def ones_like(self, value):
        return self.t.ones_like(value)

    def stop_gradient(self, value):
        return value.detach()

    def arccos(self, value):
        return self.t.arccos(value)

    def arctan2(self, first, second):
        return self.t.atan2(first, second)

    def sqrt(self, value):
        return self.t.sqrt(value)

    def exp(self, value):
        return self.t.exp(value)

    def log(self, value):
        return self.t.log(value)

    def abs(self, value):
        return self.t.abs(value)

    def sin(self, value):
        return self.t.sin(value)

    def cos(self, value):
        return self.t.cos(value)

    def clip(self, value, lo, hi):
        return self.t.clamp(value, lo, hi)

    def _pair(self, first, second):
        ref = (
            first
            if self.t.is_tensor(first)
            else second
            if self.t.is_tensor(second)
            else None
        )
        device = ref.device if ref is not None else None
        return self.t.as_tensor(first, device=device), self.t.as_tensor(
            second, device=device
        )

    def minimum(self, first, second):
        return self.t.minimum(*self._pair(first, second))

    def maximum(self, first, second):
        return self.t.maximum(*self._pair(first, second))

    def clamp_min(self, value, lo):
        return self.t.clamp(value, min=lo)

    def clamp_max(self, value, hi):
        return self.t.clamp(value, max=hi)

    def where(self, condition, first, second):
        if isinstance(condition, bool):
            return first if condition else second
        return self.t.where(condition, first, second)


def get_ops(backend: str, device=None):
    """Return the lazy array-operation facade for one backend."""
    if backend == "numpy":
        import numpy as np

        return _AxisOps(np, np.int64, lambda value: value, lambda value: value)
    if backend == "torch":
        import torch

        return _TorchOps(torch, device)
    if backend == "jax":
        import jax
        import jax.numpy as jnp

        return _AxisOps(jnp, jnp.int32, jnp.asarray, jax.lax.stop_gradient)
    raise ValueError(f"unknown backend {backend!r} (numpy / torch / jax)")
