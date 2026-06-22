"""Backend array-op facade (``ops``) for the custom-restraint vocabulary.

The vocabulary (geometry + math + penalty) is written ONCE against this facade, so the
same code runs on numpy / torch / jax and parity is structural (literally one
implementation). The only per-backend differences live here: the reduction-axis keyword
(``axis=`` for numpy/jax vs ``dim=`` for torch), the cross/clamp spellings, and the int
dtype used for gather indices. torch / jax are imported lazily, so importing this module
(and ``rgi_utils``) stays numpy-only.

Atom-group selections are exact index lists (no padding across restraints — each custom
restraint is its own closure), so the facade needs no mask handling: a centroid is just
the mean over the gathered atoms.
"""

from __future__ import annotations

EPS = 1e-12


class _AxisOps:
    """numpy / jax share one implementation (both use ``axis=`` and the same fn names);
    only the module (``xp``), the int dtype, and scalar wrapping differ."""

    def __init__(self, xp, int_dtype, wrap):
        self.xp = xp
        self._int = int_dtype
        self._wrap = wrap

    # --- indexing ---
    def gather(self, coords, idx):
        return coords[..., idx, :]  # (..., k, 3); fancy-index works on all backends

    def asint(self, a):
        return self.xp.asarray(a, dtype=self._int)

    def const(self, x):
        return self._wrap(x)

    # --- reductions over the atom axis (-2) / xyz axis (-1) ---
    def mean_atoms(self, pos):
        return self.xp.mean(pos, axis=-2)

    def mean_last(self, x):
        return self.xp.mean(x, axis=-1)

    def sum_atoms(self, pos):
        return self.xp.sum(pos, axis=-2)

    def vdot(self, a, b):
        return self.xp.sum(a * b, axis=-1)

    def vnorm(self, v):
        return self.xp.sqrt(self.xp.sum(v * v, axis=-1) + EPS)

    def cross(self, a, b):
        return self.xp.cross(a, b)

    # --- elementwise math (the DSL / ctx vocabulary) ---
    def arccos(self, x):
        return self.xp.arccos(x)

    def arctan2(self, y, x):
        return self.xp.arctan2(y, x)

    def sqrt(self, x):
        return self.xp.sqrt(x)

    def exp(self, x):
        return self.xp.exp(x)

    def log(self, x):
        return self.xp.log(x)

    def abs(self, x):
        return self.xp.abs(x)

    def sin(self, x):
        return self.xp.sin(x)

    def cos(self, x):
        return self.xp.cos(x)

    def clip(self, x, lo, hi):
        return self.xp.clip(x, lo, hi)

    def minimum(self, a, b):
        return self.xp.minimum(a, b)

    def maximum(self, a, b):
        return self.xp.maximum(a, b)

    def clamp_min(self, x, lo):
        return self.xp.maximum(x, lo)

    def clamp_max(self, x, hi):
        return self.xp.minimum(x, hi)

    def where(self, c, a, b):
        return self.xp.where(c, a, b)

    def sum(self, x):
        return self.xp.sum(x)


class _TorchOps:
    """torch uses ``dim=`` and a few different spellings (clamp / linalg.cross / atan2).
    ``device`` places the baked selection-index tensors on the coords' device."""

    def __init__(self, torch, device=None):
        self.t = torch
        self._device = device

    def gather(self, coords, idx):
        return coords[..., idx, :]

    def asint(self, a):
        return self.t.as_tensor(a, dtype=self.t.long, device=self._device)

    def const(self, x):
        return x  # a python float composes with torch ops

    def mean_atoms(self, pos):
        return self.t.mean(pos, dim=-2)

    def mean_last(self, x):
        return self.t.mean(x, dim=-1)

    def sum_atoms(self, pos):
        return self.t.sum(pos, dim=-2)

    def vdot(self, a, b):
        return self.t.sum(a * b, dim=-1)

    def vnorm(self, v):
        return self.t.sqrt(self.t.sum(v * v, dim=-1) + EPS)

    def cross(self, a, b):
        return self.t.linalg.cross(a, b, dim=-1)

    def arccos(self, x):
        return self.t.arccos(x)

    def arctan2(self, y, x):
        return self.t.atan2(y, x)

    def sqrt(self, x):
        return self.t.sqrt(x)

    def exp(self, x):
        return self.t.exp(x)

    def log(self, x):
        return self.t.log(x)

    def abs(self, x):
        return self.t.abs(x)

    def sin(self, x):
        return self.t.sin(x)

    def cos(self, x):
        return self.t.cos(x)

    def clip(self, x, lo, hi):
        return self.t.clamp(x, lo, hi)

    def _pair(self, a, b):
        # place both operands on the tensor operand's device (a python scalar would
        # otherwise land on CPU and mismatch a CUDA tensor in torch.minimum/maximum).
        ref = a if self.t.is_tensor(a) else (b if self.t.is_tensor(b) else None)
        dev = ref.device if ref is not None else None
        return self.t.as_tensor(a, device=dev), self.t.as_tensor(b, device=dev)

    def minimum(self, a, b):
        return self.t.minimum(*self._pair(a, b))

    def maximum(self, a, b):
        return self.t.maximum(*self._pair(a, b))

    def clamp_min(self, x, lo):
        return self.t.clamp(x, min=lo)

    def clamp_max(self, x, hi):
        return self.t.clamp(x, max=hi)

    def where(self, c, a, b):
        return self.t.where(c, a, b)

    def sum(self, x):
        return self.t.sum(x)


def get_ops(backend: str, device=None):
    """Return the ops facade for ``backend`` ('numpy' | 'torch' | 'jax'); torch / jax are
    imported lazily so the numpy path (and ``import rgi_utils``) needs neither. ``device``
    is honoured only by torch (where the index tensors must sit on the coords' device)."""
    if backend == "numpy":
        import numpy as np

        return _AxisOps(np, np.int64, lambda x: x)
    if backend == "torch":
        import torch

        return _TorchOps(torch, device)
    if backend == "jax":
        import jax.numpy as jnp

        return _AxisOps(jnp, jnp.int32, jnp.asarray)
    raise ValueError(f"unknown backend {backend!r} (numpy / torch / jax)")
