#!/usr/bin/env bash
# Inventory the RGI wiring of a tool's current worktree, as a stable, diffable snapshot.
#
# Why this exists: a conflict-free merge is not proof the RGI integration survived. When
# upstream rewrites a region RGI hooks into, git takes upstream's side without ever raising
# a conflict and the RGI lines vanish silently — the symptom is `n_active=0`, a restraint
# that parses fine and does nothing. (Precedent: boltz upstream merge f99e260 dropped the
# per-ligand conformer_restraint opt-in this way.)
#
# Run it on `rgi-integration` BEFORE the merge and again AFTER, then diff the two outputs.
# Any marker count that DROPS, or any file that leaves the main..rgi-integration diffstat,
# means plumbing was lost.
#
# Usage:
#   rgi_probe.sh <tool>  > /tmp/rgi_before.txt     # before `git merge main`
#   rgi_probe.sh <tool>  > /tmp/rgi_after.txt      # after
#   diff /tmp/rgi_before.txt /tmp/rgi_after.txt
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"

tool="${1:-}"
if [ -z "$tool" ]; then
    echo "usage: rgi_probe.sh <tool>   (e.g. boltz_restr)" >&2
    exit 2
fi
d="$ROOT/$tool"
[ -d "$d/.git" ] || { echo "not a git checkout: $d" >&2; exit 2; }

# The strings that mark RGI wiring. Two of these look redundant but are not:
#  - `conformer_restraint`: the per-ligand opt-in plumbing, which an upstream merge has
#    actually dropped before (boltz f99e260). It mentions neither rgi_utils nor
#    CombinedRestraints, so the obvious markers miss it entirely.
#  - `restraints=` / `is_active`: the pass-through plumbing. Whole files of the RGI patch
#    (e.g. transformers' modeling_esmfold2.py) consist only of threading a `restraints=`
#    kwarg down the call chain. Drop that and the hook below it never fires, while every
#    other marker still greps clean.
MARKERS='rgi_utils|CombinedRestraints|restraints_config|restraints\.minimize|restr\.minimize|conformer_restraint|build_.*_adapter|restraints=|restraints is not None|restraints\.is_active'

echo "# rgi_probe: $tool"
echo "# branch: $(git -C "$d" rev-parse --abbrev-ref HEAD)"
echo
echo "## RGI marker lines per file (a DROP here = plumbing lost)"
# Sorted and count-per-file so the output is diffable; lockfiles/docs excluded as noise.
git -C "$d" grep -c -I -E "$MARKERS" -- . \
    2>/dev/null \
    | grep -v -E '(^|/)(pixi\.lock|uv\.lock|poetry\.lock|README\.md|\.gitignore):' \
    | sort \
    | sed 's/^/  /'
echo
echo "## total marker lines"
git -C "$d" grep -h -I -E "$MARKERS" -- . 2>/dev/null | wc -l | sed 's/^/  /'
echo
echo "## RGI patch surface: files added by rgi-integration (a FILE LEAVING here = lost)"
# Diff from the MERGE BASE, not from `main` directly. Once `main` has moved ahead and
# rgi-integration hasn't caught up yet, `main..rgi-integration` also reports upstream's new
# work as reverse-deltas — unrelated files show up looking like RGI deletions. The merge
# base is the last point the two branches agreed, so this stays the clean RGI patch at every
# stage of the sync, and the before/after snapshots stay comparable.
base="$(git -C "$d" merge-base main rgi-integration 2>/dev/null)"
echo "  (merge-base: ${base:0:8} — this line SHOULD change once the merge lands; ignore it)"
git -C "$d" diff --name-only "$base"..rgi-integration 2>/dev/null | sort | sed 's/^/  /'
echo
echo "## conflict markers left behind (must be empty)"
git -C "$d" grep -n -E '^(<<<<<<< |>>>>>>> |=======$)' -- . 2>/dev/null | head -20 | sed 's/^/  /'

# `git grep` exits 1 on "no matches", which is the success case for the block above.
# The probe reports by printing, not by status, so don't let that leak out as a failure.
exit 0
