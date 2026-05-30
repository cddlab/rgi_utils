"""GPU-resident restraint optimizers.

``torch_optim`` and ``jax_optim`` both minimize the restraint energy on the
active-site coordinates, in-place for torch and functionally for jax. They are
imported lazily by ``combined.py`` so the package does not require both backends.
"""
