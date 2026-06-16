"""GPU-resident restraint optimizers.

``torch_optim`` and ``jax_optim`` both minimize the restraint energy on the
active-site coordinates, in-place for torch and functionally for jax. They are
imported lazily by ``combined.py`` so the package does not require both backends.

``scan_runner.ScanMinimizer`` wraps a pure minimizer + ``CombinedRestraints`` for use
inside a JAX diffusion scan (framework-free — imports neither backend; import it
directly to keep this package's import graph backend-free).
"""
