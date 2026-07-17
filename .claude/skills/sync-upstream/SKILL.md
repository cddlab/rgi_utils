---
name: sync-upstream
description: >-
  Pull real-upstream updates into an RGI tool fork's `main` (a pristine mirror of the
  predictor's true upstream — jwohlwend/boltz, bytedance/Protenix, chaidiscovery/chai-lab,
  google-deepmind/alphafold3, aqlaboratory/openfold-3, Biohub/esm, Biohub/transformers),
  then propagate that update into its `rgi-integration` branch by merging, resolving the
  conflicts that land on the RGI hook points, and verifying the RGI wiring survived. Use
  this skill whenever someone wants to sync / update / catch up a `*_restr` fork with its
  fork source — "pull in the upstream updates", "merge the fork source's changes into main
  and propagate them to rgi-integration", "catch boltz up with the latest", "update the
  fork", "merge upstream into rgi-integration", "I want to keep up with the original repo"
  — even when they don't name the branches or the upstream repo.
  It carries the verified upstream URL table (4 of the 7 forks have no `upstream` remote
  yet, and transformers' upstream is Biohub, NOT huggingface), so do not guess remotes. A
  conflict-free merge is NOT proof of success — upstream rewrites silently drop RGI
  plumbing — so always run the wiring verification before reporting done.
---

# Sync real-upstream updates into an RGI fork, then into `rgi-integration`

Every RGI tool (`boltz_restr`, `protenix_restr`, `chai-lab_restr`, `alphafold3_restr`,
`openfold-3_restr`, `esm_restr`, `transformers_restr`) is a checkout of a `cddlab/*_restr`
repo with two branches that carry a deliberate contract:

- **`main`** = a pristine mirror of the predictor's *real upstream*. No RGI code.
- **`rgi-integration`** = `main` + the `rgi_utils` integration. The canonical RGI branch.

That split is what makes `git diff main..rgi-integration` *exactly* the RGI patch, and it
is what keeps upstream syncs near-conflict-free. Every step below exists to preserve it.
The flow is always a two-hop merge chain:

```
upstream/main  ──merge──►  main  ──merge──►  rgi-integration
```

## Hard rules (violating these destroys the contract)

- **Never merge `rgi-integration` into `main`.** That puts RGI code on `main`, and both
  benefits above die at once. (boltz already drifted this way historically; it was
  repaired with a revert.)
- **Merge, never rebase, into `rgi-integration`.** Rebasing replays RGI commits onto files
  upstream already changed and manufactures conflicts that a merge never sees.
- **Never force-push.** Both branches are published. If history looks wrong, stop and
  report — do not rewrite.
- **Never hand-edit lockfiles through conflict markers** (`pixi.lock`, `uv.lock`,
  `poetry.lock`). Regenerate them with the resolver instead.

## Step 1 — Scope and status

Default scope is **the tool(s) the user named**. If they named none, run the status probe
across all seven and report which ones actually have upstream commits pending, then ask
which to sync rather than syncing everything.

```bash
rgi_utils/.claude/skills/sync-upstream/scripts/sync_status.sh            # all seven
rgi_utils/.claude/skills/sync-upstream/scripts/sync_status.sh boltz_restr chai-lab_restr
```

The script is idempotent: it adds the `upstream` remote where missing (from the verified
table in `references/tools.md`), fetches, and prints per tool how far `main` is behind
`upstream/main`, whether local `main`/`rgi-integration` diverge from `origin`, and whether
the worktree is dirty.

**Stop and ask the user before touching anything if the probe reports** a dirty worktree,
or a local branch ahead of `origin` (unpushed work — e.g. boltz's `main` has carried an
unpushed revert + merge for a while). Merging on top of unpushed local state is fine, but
the user needs to know it's there, because your push at the end would publish it too.

## Step 2 — `main` ← `upstream/main`

```bash
cd <tool>
git checkout main
git merge upstream/main
```

For six of the seven this fast-forwards — `main` has no commits of its own, so there is
nothing to conflict. **boltz is the exception**: its `main` carries a revert commit, so it
has genuinely diverged and this is a real merge. It still resolves cleanly, because the
revert made `main`'s *tree* equal to the merge base — but if git does report conflicts
here on boltz, that is a signal something else is wrong; read `references/tools.md` before
resolving.

A conflict on `main` in any *other* tool means `main` is not pristine after all. Stop and
report that finding rather than resolving it — the fix is a separate decision (see the
boltz revert precedent in the workspace `CLAUDE.md`), not part of a routine sync.

## Step 3 — Record the RGI baseline (do this *before* merging into `rgi-integration`)

```bash
git checkout rgi-integration
rgi_utils/.claude/skills/sync-upstream/scripts/rgi_probe.sh <tool> > <scratch>/rgi_before.txt
```

`rgi_probe.sh` inventories where the RGI wiring currently lives: per file, how many lines
carry an RGI marker (`rgi_utils`, `CombinedRestraints`, `restraints_config`,
`restraints.minimize`, the `conformer_restraint` opt-in, and the `restraints=` pass-through
kwarg), plus the RGI patch's file list. Take the snapshot *before* the merge — afterwards
the pre-merge counts are gone, and they are what Step 7 compares against.

## Step 4 — `rgi-integration` ← `main`

```bash
git merge main
```

A clean merge commits itself, so you can jump straight to Step 7 — but **do not skip Step
7**. A clean merge is exactly the case where RGI plumbing disappears without telling you.

## Step 5 — Resolve conflicts

Conflicts cluster in a handful of predictable places, because that is precisely where RGI
touches the tool: the diffusion loop, the input schema, and the dependency files.

The resolution principle is always the same, and it is not "pick a side": **take upstream's
change in full, then re-install the RGI hook on top of the new code.** RGI is an additive
patch, so a correct resolution keeps both — upstream's new logic *and* the restraint hook,
adapted to whatever upstream renamed or restructured.

"Adapted correctly" has a spec, not a vibe. The RGI hook contract (where `minimize` fires,
which sigma it gates on, what `setup`/`finalize` must bracket) lives in
`rgi_utils/.claude/skills/implement-rgi/references/lifecycle-and-hooks.md`, and the traps
are catalogued in `.../references/pitfalls.md`. Read them when a hook site has moved
enough that you're re-deriving the hook rather than re-indenting it. The invariants in the
workspace `CLAUDE.md` ("Non-obvious invariants that must hold across all six tools") are
the acceptance criteria — especially that `minimize` gates on the **pre-step / pre-churn**
schedule sigma, which is easy to lose when upstream reshuffles a sampling loop.

Read `references/conflicts.md` for the per-hunk playbook: lockfiles, `README`/`.gitignore`
(RGI-owned, keep ours + fold in upstream's additions), the loop hooks, and AF3's
`run_alphafold.py`, which is a ~1500-line whole-file reformat against upstream and must not
be line-merged — see `references/tools.md`.

## Step 6 — Commit the resolved merge (only if Step 5 ran)

```bash
git add <each file you resolved>          # NOT `git add -A`
git commit -m "Merge main (upstream <repo>) into rgi-integration"
```

**Do not use `git add -A`.** A conflicted merge already staged everything that auto-merged,
so only the files you hand-resolved need adding — while `-A` additionally sweeps in every
untracked file in the worktree. These checkouts are full of untracked scratch (`.coverage`,
`.pixi-bin/`, bench outputs, `checkpoint/`), which is exactly what the Step 1 dirty-worktree
warning was telling you about, and `-A` commits it straight onto a published branch. If you
already did it, `git rm -r --cached <paths>` + `git commit --amend` removes them from the
commit while leaving them on disk. The Step 7 probe catches this — scratch paths appear in
the RGI patch surface — but it is much cheaper not to stage them.

Do this *before* verifying, not after. Every Step 7 check asks git about the
`rgi-integration` branch, and until you commit, that branch still points at the pre-merge
commit: the probe would faithfully re-measure the old tree and report a reassuring "nothing
changed" while your resolution sits unexamined in the index. It is a convincing false pass.

## Step 7 — Verify the RGI wiring survived (mandatory)

A clean merge is **not** evidence of success. When upstream rewrites a region that RGI
hooks into, git resolves it in upstream's favour without a conflict and the RGI lines
vanish silently. This has already happened once: boltz's upstream merge `f99e260` dropped
the per-ligand `conformer_restraint` opt-in plumbing with no conflict, and the only symptom
was `n_active=0` — a restraint that parses fine and does nothing.

```bash
rgi_utils/.claude/skills/sync-upstream/scripts/rgi_probe.sh <tool> > <scratch>/rgi_after.txt
diff <scratch>/rgi_before.txt <scratch>/rgi_after.txt
```

Read the diff with this question in mind: **did any RGI marker line count go down, or did
any RGI file leave the patch surface?** Either means plumbing was dropped. Increases and new
files are usually fine (upstream grew, or you re-installed a hook). Investigate every
decrease against `git log -p main..rgi-integration -- <file>` and restore what was lost.

The only line that is *expected* to differ is the `merge-base:` marker — it moves precisely
because the merge landed. A clean sync's diff shows that and nothing else.

Also confirm mechanically:

```bash
git -C <tool> diff --check                                  # no leftover conflict markers
git -C <tool> grep -n '<<<<<<<\|>>>>>>>' rgi-integration    # ditto, committed content
git -C <tool> diff --stat main..rgi-integration             # still the clean RGI patch
```

The diffstat is a second read on the contract: it should list only RGI files. If upstream
files you never touched appear in it, `main` and `rgi-integration` have drifted apart —
something got merged in the wrong direction.

Then, if the tool's environment is available, run the cheap engine check (it needs no GPU):

```bash
rgi_utils/.venv/bin/python -m pytest -m "not gpu" tests/test_backend_parity.py -q
```

Full E2E confirmation is a GPU job and **must go through `sbatch`, never the login node**
(see the workspace `CLAUDE.md` for the recipe and the per-tool partition constraints —
protenix must run on sm_89). Don't launch one unprompted; instead tell the user which E2E
fixture would confirm the hook still fires, and offer to submit it.

The cheapest fixture that proves the hook actually fires is the **fumarate plane run**
(`sbatch_{af3,chai,of3,esm}_plane.sh` + `input_*_plane.*`). One decisive line:

```
built spec: n_active=8 bonds=7 angles=8 chirals=0 plane=2 cistrans=1 ...
[rgi_utils] finalize (step N): ... plane=0.00000 ... total=0.00000
```

`n_active=8` / `plane=2` / `cistrans=1` is the documented fumarate expectation, and it is
**identical across tools and backends** — verified 2026-07-17 on af3 (jax), chai and
openfold-3 (torch) after this sync. `n_active=0` is the silent-drop signature: the config
parses, the run succeeds, and nothing is restrained. Note `finalize plane=0.00000` only
means "satisfied" when the spec reports `plane>0` — with `plane=0` the same line is a no-op,
so always read the two together.

If a partition is busy, move the job rather than wait: af3/chai/openfold-3 all run fine on
`q1`/`q3` (sm_89), and `sbatch -p q3 -o <new>.out <script>` overrides the script's own
`#SBATCH` lines without editing the fixture.

## Step 8 — Report, then push after confirmation

Show the user, per tool: how many upstream commits came in, what conflicted and how you
resolved it, and the Step 7 verification result. Then ask before pushing — these branches
are published and the push is outward-facing.

```bash
git push origin main                 # plain push, NEVER --force
git push origin rgi-integration
```

Push `main` first: `rgi-integration` contains `main`, so the reverse order publishes a
merge whose parent isn't on the remote yet.

If verification found dropped plumbing that you could not confidently restore, say so
plainly and do not push. A silently no-op restraint is worse than an unsynced fork,
because it produces plausible structures that quietly ignore the restraint.

## Reference files

- `references/tools.md` — the verified upstream URL table for all seven forks, each tool's
  RGI hook points, and the per-tool gotchas (boltz's diverged `main`, AF3's reformatted
  `run_alphafold.py`, transformers' Biohub upstream, esm's lockfile).
- `references/conflicts.md` — the conflict playbook by hunk type, with worked resolutions.
