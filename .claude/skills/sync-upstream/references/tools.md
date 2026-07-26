# Per-tool upstream table, hook points, and gotchas

## The upstream table (verified — do not guess or re-derive)

Every `cddlab/*_restr` repo is an **independent repo, not a GitHub fork**. `gh repo view
--json parent` returns null for all seven, so there is no way to rediscover these URLs from
GitHub metadata — that is why they are written down here. Verified 2026-07-17 by confirming
each fork's `origin/main` HEAD commit exists in the listed upstream repo.

| tool directory | upstream remote URL |
| --- | --- |
| `boltz_restr` | `https://github.com/jwohlwend/boltz.git` |
| `protenix_restr` | `https://github.com/bytedance/Protenix.git` |
| `alphafold3_restr` | `https://github.com/google-deepmind/alphafold3.git` |
| `chai-lab_restr` | `https://github.com/chaidiscovery/chai-lab.git` |
| `openfold-3_restr` | `https://github.com/aqlaboratory/openfold-3.git` |
| `esm_restr` | `https://github.com/Biohub/esm.git` |
| `transformers_restr` | `https://github.com/Biohub/transformers.git` |

`scripts/sync_status.sh` carries this same table and adds any missing remote for you. All
seven checkouts had the remote configured as of 2026-07-17, but the setup stays in the
script because it is idempotent and a fresh clone starts with none of them.

Two entries are counter-intuitive and worth stating outright:

- **`transformers_restr`'s upstream is `Biohub/transformers`, NOT `huggingface/transformers`.**
  The fork's history does reach back to huggingface's initial commit, so the repo *looks*
  like plain transformers — but `esmfold2` (the model RGI hooks into) exists only in the
  Biohub fork. Pointing `upstream` at huggingface would pull a firehose of unrelated
  commits and would never deliver esmfold2 updates.
- **`esm_restr`'s upstream is `Biohub/esm`** — the repo formerly known as
  `evolutionaryscale/esm`. The old URL still redirects, so both fetch fine; prefer the
  current name so the remote doesn't depend on a redirect.

## Where RGI touches each tool

These are the files that carry RGI wiring, i.e. the places a conflict will land and the
places a silent drop can happen. `scripts/rgi_probe.sh` recomputes this live; the list here
is for orienting before you start.

| tool | RGI-touched source files |
| --- | --- |
| `boltz_restr` | `model/modules/diffusion.py`, `diffusionv2.py` (loop hooks); `data/parse/schema.py`, `data/types.py`, `data/feature/featurizer{,v2}.py`, `data/module/inference{,v2}.py`, `data/parse/mmcif*.py` (config + per-ligand opt-in); `main.py`; `pyproject.toml` |
| `protenix_restr` | `model/generator.py`, `model/protenix.py` (loop hooks); `tfg/engine.py` (native TFG composition); `data/inference/json_parser.py`, `json_to_feature.py`; `runner/inference.py`, `runner/dumper.py`; `requirements.txt` |
| `chai-lab_restr` | `chai_lab/chai1.py` (loop hooks + sidecar YAML load); `data/dataset/all_atom_feature_context.py`; `requirements.in` |
| `alphafold3_restr` | `model/network/diffusion_head.py` (loop hook inside the scan); `model/restraints/` (new dir: `adapter.py`, `combined_restraints.py`); `common/folding_input.py`; `run_alphafold.py` (**see the AF3 warning below**); `pyproject.toml`, `uv.lock` |
| `openfold-3_restr` | `core/model/structure/diffusion_module.py` (loop hook); `projects/of3_all_atom/model.py`; `core/data/primitives/structure/query.py`, `projects/of3_all_atom/config/inference_query_format.py` (config); `core/utils/tensor_utils.py`; `pixi.toml` |
| `esm_restr` | `esm/models/esmfold2/prepare_input.py`, `processor.py`, `conformers.py`; `esm/utils/structure/input_builder.py`; `pyproject.toml`, `pixi.lock` |
| `transformers_restr` | `src/transformers/models/esmfold2/modeling_esmfold2_common.py` (**the loop hook**), `modeling_esmfold2.py`, `modeling_esmfold2_experimental.py` |

Note that the two ESMFold2 repos are coupled: the model and diffusion loop live in
`transformers_restr`, the user API in `esm_restr`. An upstream sync of one may require the
other. If `Biohub/transformers` changes `DiffusionStructureHead.sample`, that is the exact
function RGI hooks — expect a conflict and re-derive the hook rather than force-fitting it.

## Per-tool gotchas

### boltz — `main` has genuinely diverged (this is intentional)

`boltz_restr`'s `main` is the only one that is not a plain ancestor of upstream. It once
carried an old, pre-`rgi_utils` in-tree RGI implementation, which was undone with a revert
commit (`5dd09f1`) rather than a history rewrite, because `main` is a published release.
The net effect: `main`'s *tree* equals upstream, but its *history* has a detour.

Consequences for a sync:

- `git merge upstream/main` on boltz's `main` is a **real merge, not a fast-forward** — and
  it is conflict-free precisely because the tree already matches the merge base. Do not be
  alarmed by the merge commit; do be alarmed by a conflict.
- Do **not** `git rebase` or `--force` to "clean it up". The divergence is the deliberate,
  history-preserving choice.
- Local `main` sits ahead of `origin/main` with unpushed commits (the revert `5dd09f1`, and
  an upstream merge `72e21c0`, have both been held back from publication by the user's
  choice). `sync_status.sh` flags this. Tell the user before pushing — your push publishes
  those too.
- What "not pushing boltz's `main`" actually withholds is **only the branch pointer**, not
  the commits. `main` is fully merged into `rgi-integration`, and `origin/rgi-integration`
  is already published, so `5dd09f1` and `72e21c0` are *already reachable on the remote*
  (verify with `git merge-base --is-ancestor 5dd09f1 origin/rgi-integration`). The
  consequence: `origin/main` still points at the old-RGI `72ae28e`, so the
  "`git diff main..rgi-integration` is exactly the RGI patch" contract holds **locally but
  not on GitHub**. Don't describe boltz as contract-clean on the remote until `origin/main`
  is pushed.

Upstream also has a habit of dropping RGI plumbing here without conflicting: merge `f99e260`
silently removed the per-ligand `conformer_restraint` opt-in across `types.py` / `schema.py`
/ `featurizer{,v2}.py` / `mmcif`, and the only symptom was a spec with `n_active=0`. This is
the single strongest argument for the Step 3 / Step 6 probe.

### AF3 — `run_alphafold.py` WAS a whole-file reformat; it was cleaned up 2026-07-17

**Do not `git checkout --theirs run_alphafold.py` any more.** That advice was correct once
and is now actively destructive; read this section before acting on any memory of it.

History: `run_alphafold.py` on `rgi-integration` used to be a wholesale 4-space reformat of
upstream's 2-space Google style, so `git diff main..rgi-integration` showed ~840 changed
lines for a handful of real RGI additions. When upstream's `pathlib` → `epath` migration
landed on top, the textual merge exploded into 21 conflict hunks / 757 lines of pure
formatting noise. It was resolved the way this file used to prescribe: take upstream's file
wholesale, then hand re-apply the five RGI additions in upstream's style.

That worked, and it **removed the reformat permanently** — the file's RGI patch went from
`840+/692-` to `83+/4-`. So the condition that justified `--theirs` no longer exists:
`run_alphafold.py` is now an ordinary hook file whose diff is only RGI. Taking upstream's
copy today would simply delete those five additions, and unlike last time there would be no
formatting churn hiding them to recover from. Resolve it hunk-by-hunk like any other file,
per the keep-both principle in `conflicts.md`.

For reference, the five RGI additions in `run_alphafold.py` are: the `build_restraints`
import; `ModelRunner._build_model_with_restraints` (a JIT-compiled forward pass with the
restraints closed over, cached by restraints identity so each seed doesn't recompile the
model); the `restraints=` parameter on `run_inference` plus the model_fn selection; building
`restraints` once from the first featurised example in `predict_structure`; and the
best-effort `finalize` energy log after the first seed's inference.

AF3's other structural difference: RGI lives partly in its own new package
(`src/alphafold3/model/restraints/`), which upstream cannot conflict with by construction.
The real hook is in `diffusion_head.py`, where the pure `restr.get_minimizer()` closure is
injected inside the `lax.scan` — see `alphafold3_restr/AGENTS.md` (or its `CLAUDE.md`
alias) and the implement-rgi skill's `framework-notes.md` for why the spec must be built
outside the scan.

### Lockfiles — regenerate, never hand-merge

`esm_restr`'s `pixi.lock` differs from `main` by ~2700 lines; `alphafold3_restr` carries a
`uv.lock`. Conflict markers inside a lockfile are unresolvable by reading — the file is
resolver output, not source.

```bash
# esm / openfold-3 (pixi)
git checkout --theirs pixi.lock && pixi install -e <env>    # then commit the regenerated lock
# alphafold3 (uv)
git checkout --theirs uv.lock && uv lock
```

Take upstream's lock as the base, then let the resolver re-add the RGI dependency
(`rgi_utils`, declared in `pixi.toml` / `pyproject.toml` — make sure that declaration
survived the merge first, since it is the actual source of truth). Lock regeneration
downloads packages, so on this cluster it belongs in an `sbatch` job, not the login node.

**Locks are a blind spot in `rgi_probe.sh`** — it excludes them from the marker inventory,
because thousands of lines of resolver output would drown the signal. So when a merge
touches a lock, check it by hand:

```bash
git -C <tool> grep -c -i 'rgi' origin/rgi-integration -- pixi.lock   # before
git -C <tool> grep -c -i 'rgi' rgi-integration -- pixi.lock          # after
```

As of 2026-07-17 both counts are **0** in every tool: `rgi_utils` is a git dependency and
has no materialised entry in `pixi.lock` / `uv.lock`, so the RGI declaration lives only in
`pixi.toml` / `pyproject.toml` (which the probe *does* cover). A nonzero-to-zero drop would
be a real regression; 0-to-0 is the expected steady state.

Git will sometimes **auto-merge a lock with no conflict** (this happened to esm's
`pixi.lock` and af3's `uv.lock` on 2026-07-17). That result is not resolver output and may
be internally inconsistent — but unlike dropped RGI plumbing it fails **loudly** at install
time rather than silently at inference, so it is worth reporting as a residual risk rather
than blocking the sync on an `sbatch` regeneration.

### `README.md` and `.gitignore` — RGI-owned on `rgi-integration`

Every tool's `rgi-integration` adds an RGI section to `README.md` and RGI entries to
`.gitignore`. These are pure additions, so conflicts here are mechanical: keep both sides —
upstream's new content plus the RGI block. Never resolve by taking one side wholesale.

### protenix — `tfg/engine.py` composes with native guidance

protenix's RGI hook coexists with protenix's own Training-Free Guidance: the composition
order is TFG first, then RGI. If upstream restructures `tfg/engine.py`, preserve that
ordering when re-installing the hook — swapping it changes what the restraint acts on.

### chai / openfold-3 — the smallest RGI patches

Nothing structural is special about them; chai's RGI patch is the smallest of the seven
(5 files) and openfold-3's is 9.

One practical trap, hit on the 2026-07-17 sync: **an upstream sync can raise the tool's own
toolchain floor**, and that surfaces only when you try to run it, well after the merge
verified clean. openfold-3's 124-commit catch-up bumped `requires-pixi` in `pixi.toml` to
`>=0.72.0`, so the workspace's pinned `.pixi-bin/pixi` (0.70.0) refused to start with
`× this project requires pixi '>=0.72.0'`. That is not an RGI regression — the merge was
clean and the probe was clean. Re-read `requires-pixi` after any openfold-3 sync and refresh
the binary if it moved:

```bash
curl -fsSL -o .pixi-bin/pixi \
  https://github.com/prefix-dev/pixi/releases/download/v0.72.0/pixi-x86_64-unknown-linux-musl
chmod +x .pixi-bin/pixi
```

(Do the download in an `sbatch` job, like every other fetch on this cluster.)
