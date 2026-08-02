"""Restraint energy backends.

Shared geometry and energy kernels are implemented against the lazy array facade;
the ``numpy_energy``, ``torch_energy``, and ``jax_energy`` modules are thin adapters that
preserve the same two public entry points:

    prepare_spec(spec, ...) -> prepared   # convert RestraintSpec -> backend arrays
    total_energy(positions, prepared) -> scalar

``positions`` has shape ``(..., n_active, 3)`` — any leading batch dimensions are
allowed and summed over, so the same function works for a single structure or a
batch. Indices in ``prepared`` are local indices into ``active_sites``.

Backends are imported lazily by ``combined.py`` so that ``import rgi_utils`` works
without torch or jax installed.
"""
