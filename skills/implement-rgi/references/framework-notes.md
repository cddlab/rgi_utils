# Framework notes

## PyTorch (eager loop) — boltz, protenix

- **autograd under `inference_mode`**: predict loops often run inside
  `torch.inference_mode()`, where autograd backward fails with "element does not
  require grad / no grad_fn". rgi_utils' torch optimizer handles this for you: it
  wraps the minimization in `torch.inference_mode(False) + torch.enable_grad()`
  and clones the active coords into a normal leaf (`empty_like + copy_`) so
  backward works. You don't write this — but know it's why passing an
  inference-mode coordinate tensor to `minimize` is fine.
- **adapter placement**: `rgi_utils/<tool>/adapter.py` (receives a plain
  dict/array, imports no framework code).
- **VdW flavours**: the dynamic ligand-protein VdW (a per-step radius search) is
  torch-only; the intramolecular VdW (`vdw: {mode: intramolecular}`) is static
  and works in every backend. Pick intramolecular unless you specifically want
  ligand-protein declashing.

## JAX (JIT / `lax.scan`) — AlphaFold3

- **no Python callbacks in the scan**: build the spec outside the scan (numpy, at
  build time) and inject `get_minimizer()`'s pure closure inside. Never call the
  Python `minimize` per step in a compiled loop.
- **line search is load-bearing**: the jax optimizer uses jaxopt
  `NonlinearCG`/`LBFGS` with **`linesearch="backtracking"`** (Armijo). This is
  not cosmetic. The default `zoom`/`hager-zhang` searches can accept a huge first
  step that collapses atoms onto each other, where the eps-regularized distance
  `sqrt(0 + eps)` makes the gradient vanish — a false stationary point the solver
  then "converges" to. Backtracking rejects that step. A plain fixed-step
  gradient descent diverges (NaN) on degenerate geometry, which is why a real
  line-searched solver is required, not hand-rolled GD.
- **coordinate reshape**: AF3 coords are `(num_tokens, max_atoms, 3)`; reshape to
  `(-1, 3)` before the minimizer and back after. That reshape is the only glue.
- **non-finite guard**: the jax minimizer returns the input coords unchanged if a
  step produces NaN/Inf, so one bad geometry can't poison the whole trajectory.

## Why the old AF3 path was slow (and the fix)

The original AF3 restraint code minimized with `jaxopt.ScipyMinimize` (scipy,
*outside* JIT, with O(n) finite-difference gradients) called via
`jax.pure_callback` — a CPU round-trip on every denoise step, which was
extremely slow. Replacing it with rgi_utils' in-JIT `make_minimizer` (analytic
`jax.grad`, running entirely on the accelerator) is the "slow → fast" fix. When
you integrate a JAX tool, use `get_minimizer()`; do not reach for scipy or
`pure_callback`.

## Backend selection recap

`config.gpu: true` → torch; `config.backend: jax` → jax (set this for a JAX
tool); otherwise numpy (CPU reference / fallback). The energies are the same
maths across all three (checked by `tests/test_backend_parity.py`), so a tool
gets identical restraint behaviour regardless of backend — only speed differs.
