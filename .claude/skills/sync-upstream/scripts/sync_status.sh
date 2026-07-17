#!/usr/bin/env bash
# Idempotent upstream probe for the RGI tool forks.
#
# For each requested tool: add the `upstream` remote if absent, fetch it, and report how
# far `main` is from `upstream/main`, whether the local branches diverge from `origin`,
# and whether the worktree is dirty. Read-only with respect to branches — it never merges,
# checks out, or pushes.
#
# Usage:  sync_status.sh [tool ...]      (no args = all seven)
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# scripts -> sync-upstream -> skills -> .claude -> rgi_utils -> workspace root
ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"

# Verified 2026-07-17: each fork's origin/main HEAD was confirmed to exist in the upstream
# repo listed here. The cddlab repos are NOT GitHub forks, so `gh repo view --json parent`
# reports null and cannot be used to rediscover these.
upstream_url() {
    case "$1" in
        boltz_restr)        echo "https://github.com/jwohlwend/boltz.git" ;;
        protenix_restr)     echo "https://github.com/bytedance/Protenix.git" ;;
        chai-lab_restr)     echo "https://github.com/chaidiscovery/chai-lab.git" ;;
        alphafold3_restr)   echo "https://github.com/google-deepmind/alphafold3.git" ;;
        openfold-3_restr)   echo "https://github.com/aqlaboratory/openfold-3.git" ;;
        esm_restr)          echo "https://github.com/Biohub/esm.git" ;;
        # NOT huggingface/transformers: esmfold2 only exists in the Biohub fork.
        transformers_restr) echo "https://github.com/Biohub/transformers.git" ;;
        *)                  echo "" ;;
    esac
}

ALL_TOOLS="boltz_restr protenix_restr chai-lab_restr alphafold3_restr openfold-3_restr esm_restr transformers_restr"
TOOLS="${*:-$ALL_TOOLS}"

for tool in $TOOLS; do
    d="$ROOT/$tool"
    echo "=================================================================="
    echo "=== $tool"
    if [ ! -d "$d/.git" ]; then
        echo "    SKIP: $d is not a git checkout"
        continue
    fi
    url="$(upstream_url "$tool")"
    if [ -z "$url" ]; then
        echo "    SKIP: unknown tool (not in the upstream table)"
        continue
    fi

    # --- idempotent remote setup -----------------------------------------------
    # Compare normalized: an existing remote may spell the same repo without the `.git`
    # suffix, in a different case, or over ssh. Only a genuinely different repo is a
    # mismatch worth reporting.
    norm() { echo "$1" | sed -E 's#^git@github\.com:#https://github.com/#; s#\.git$##' | tr 'A-Z' 'a-z'; }
    have="$(git -C "$d" remote get-url upstream 2>/dev/null)"
    if [ -z "$have" ]; then
        echo "    upstream remote: ABSENT -> adding $url"
        git -C "$d" remote add upstream "$url" || continue
    elif [ "$(norm "$have")" != "$(norm "$url")" ]; then
        # Do not silently rewrite: a mismatch may be intentional, or may be a wrong guess.
        echo "    !! upstream remote MISMATCH"
        echo "       configured: $have"
        echo "       expected  : $url"
        echo "       -> not touching it; confirm with the user which is right."
    fi

    echo "    fetching upstream ..."
    git -C "$d" fetch --quiet upstream 2>&1 | sed 's/^/       /'
    git -C "$d" fetch --quiet origin 2>&1 | sed 's/^/       /'

    # --- state ------------------------------------------------------------------
    dirty="$(git -C "$d" status --porcelain | head -5)"
    if [ -n "$dirty" ]; then
        echo "    !! WORKTREE DIRTY (merging on top of this is risky — ask the user):"
        git -C "$d" status --porcelain | head -10 | sed 's/^/       /'
    fi
    echo "    current branch: $(git -C "$d" rev-parse --abbrev-ref HEAD)"

    # ahead/behind, printed as "<local-only> <remote-only>"
    ab() { git -C "$d" rev-list --left-right --count "$1...$2" 2>/dev/null || echo "?	?"; }

    read -r m_ahead m_behind <<<"$(ab main upstream/main)"
    echo "    main vs upstream/main   : main-only=$m_ahead  PENDING-UPSTREAM=$m_behind"
    read -r lo_a lo_b <<<"$(ab main origin/main)"
    echo "    main vs origin/main     : unpushed=$lo_a  behind=$lo_b"
    read -r ri_a ri_b <<<"$(ab rgi-integration origin/rgi-integration)"
    echo "    rgi-int vs origin/rgi-* : unpushed=$ri_a  behind=$ri_b"
    read -r rm_a rm_b <<<"$(ab rgi-integration main)"
    echo "    rgi-integration vs main : rgi-only=$rm_a  NOT-YET-MERGED-FROM-MAIN=$rm_b"

    if [ "$m_behind" = "0" ]; then
        echo "    => up to date with upstream."
    else
        echo "    => $m_behind upstream commit(s) to pull in. Newest:"
        git -C "$d" log --oneline -5 main..upstream/main | sed 's/^/       /'
    fi
    if [ "$lo_a" != "0" ]; then
        echo "    !! local main has $lo_a UNPUSHED commit(s) — a push at the end publishes these too:"
        git -C "$d" log --oneline -5 origin/main..main | sed 's/^/       /'
    fi
done
echo "=================================================================="
