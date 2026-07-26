"""Scan-friendly wrapper around a pure restraint minimizer.

JAX tools run the restraint minimizer INSIDE the diffusion ``lax.scan`` via a pure
``(flat_coords, sigma, step) -> flat_coords`` closure (``CombinedRestraints.get_minimizer``),
rather than calling ``CombinedRestraints.minimize`` per step like the eager torch tools.
``ScanMinimizer`` bundles that closure with its ``CombinedRestraints`` and exposes the
duck-typed interface the network consumes — ``is_active()`` / ``minimize_gpu(positions,
sigma, step)`` (plus ``n_active`` / ``finalize`` for host-side logging) — handling the only
per-call shaping the model needs: flatten the coordinate tensor to ``(-1, 3)`` for the
minimizer and restore the original shape. The flat ``(n_atom, 3)`` layout is the one the
active-site indices address, so any tool whose per-step coordinate tensor carries extra
axes (e.g. AF3's ``(num_tokens, max_atoms_per_token, 3)``) reshapes through here.

Framework-free: ``positions`` is duck-typed (anything with ``.shape`` / ``.reshape`` — a
jax or numpy array), so this module imports neither jax nor any tool, keeping
``import rgi_utils`` numpy-only.
"""

from __future__ import annotations


class ScanMinimizer:
    """Bundle a ``CombinedRestraints`` with its pure minimizer for in-scan use.

    Network code consumes this purely by duck typing: ``is_active()`` and
    ``minimize_gpu(positions, sigma, step)`` (plus ``n_active`` / ``finalize`` for
    logging).
    """

    def __init__(self, rgi, minimizer) -> None:
        self._rgi = rgi  # rgi_utils CombinedRestraints (for is_active / spec / stats)
        self._minimizer = minimizer  # pure (flat_coords, sigma) -> flat_coords | None

    def is_active(self) -> bool:
        return self._rgi.is_active() and self._minimizer is not None

    @property
    def n_active(self) -> int:
        """Number of optimised atoms (host-side logging); 0 if none."""
        spec = getattr(self._rgi, "spec", None)
        return int(spec.n_active) if spec is not None else 0

    def minimize_gpu(self, positions, sigma, step=0):
        """One restraint-minimization step on active atoms inside the diffusion scan.

        ``positions`` is the per-step coordinate tensor; it is flattened to ``(-1, 3)``
        for the pure minimizer (whose active-site indices address a flat atom axis) and
        restored to its original shape. ``sigma`` is the noise level and ``step`` the
        diffusion step index (both gate the restraints; ``step`` is a traced scalar inside
        the scan). The minimizer is pure, so this stays JIT-compiled inside ``lax.scan`` /
        ``lax.vmap``. Reshaping is shape-agnostic (it reads no tool-specific dims), so it
        works for any layout. ``step`` defaults to 0 so a tool that threads no step counter
        (i.e. uses only sigma windows) keeps working unchanged."""
        if self._minimizer is None:
            return positions
        shape = positions.shape
        flat = positions.reshape(-1, 3)
        return self._minimizer(flat, sigma, step).reshape(shape)

    def finalize(self, positions, istep: int = 0) -> None:
        """Log per-term restraint energy of a final structure (host-side, after the
        scan; the in-scan minimizer cannot log). ``positions``: any shape ending in 3
        (or already-flat ``(-1, 3)``)."""
        if self._minimizer is None:
            return
        import numpy as np

        flat = np.asarray(positions).reshape(-1, 3)
        self._rgi.finalize(flat, istep)
