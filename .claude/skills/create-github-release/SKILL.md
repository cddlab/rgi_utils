---
name: create-github-release
description: >-
  Safely create and push a SemVer Git tag such as v1.0.0, then publish the
  matching GitHub Release with gh. Use when the user asks to tag a version,
  cut or publish a release, or create a GitHub Release. Always inspect and
  report the repository's current version, proposed tag, and target commit,
  then obtain the user's explicit final approval before creating any tag or
  Release.
---

# Create a GitHub Release

Create one immutable version tag and its matching GitHub Release. Treat the
version declared by the repository, the Git tag, and the Release tag as one
version.

## Hard rules

- Always perform the mandatory approval gate below. The user's initial request
  to create a release does not count as final approval.
- Before approval, do not create a local or remote tag and do not create a
  draft or published Release.
- Never guess the version, silently change a manifest version, move an existing
  tag, overwrite a Release, force-push, or delete a local or remote tag.
- Invalidate approval and ask again if the version, target commit, branch,
  release-note mode, or repository state changes after approval.
- Follow the repository's `AGENTS.md` and release documentation. Run expensive
  or GPU checks only in the environment they prescribe.

## 1. Inspect the release state

Keep this phase read-only with respect to tags and Releases.

1. Read the repository instructions and any release workflow.
2. Inspect the worktree, branch, target commit, remotes, and upstream:

   ```bash
   git status --short --branch
   git branch --show-current
   git rev-parse HEAD
   git remote -v
   git rev-parse --abbrev-ref --symbolic-full-name '@{upstream}'
   ```

3. Locate version declarations with `rg`. Prefer a release-specific source
   documented by the repository; otherwise inspect common manifests such as
   `pyproject.toml`, `package.json`, `Cargo.toml`, and package `__version__`
   declarations. Do not treat the latest tag as the manifest version.
4. Inspect existing local and remote tags and GitHub Releases:

   ```bash
   git tag --sort=-version:refname
   git ls-remote --tags origin
   gh auth status
   gh repo view --json nameWithOwner,defaultBranchRef
   gh release list --limit 20
   ```

5. Normalize a manifest version `X.Y.Z` to tag `vX.Y.Z`. Accept a SemVer
   prerelease only when the user explicitly requests it. Reject ambiguous,
   malformed, or conflicting version declarations and ask the user which
   source is authoritative.
6. Require a clean worktree and a pushed target commit. If the manifest version
   differs from the requested tag, stop and ask whether the manifest should be
   updated; never perform the bump implicitly.

## 2. Verify before approval

Run the repository-prescribed lint and non-GPU tests. Do not publish a release
with failing checks unless the user explicitly accepts the named failures after
seeing them.

Confirm mechanically that:

- the exact proposed tag is absent locally and on `origin`;
- no GitHub Release already uses that tag;
- the remote branch points at the proposed target commit;
- the target commit and version declaration contain the intended release
  contents.

Choose generated release notes by default. If the user supplied notes or asked
for a prerelease, include that choice in the approval summary.

## 3. Mandatory approval gate

Present all of the following in the user's language:

- repository;
- version found in the authoritative file and that file's path;
- latest existing version tag, or `none`;
- proposed new tag;
- full target commit SHA and branch;
- whether the target is pushed and the worktree is clean;
- verification results;
- Release title, generated/custom notes mode, and stable/prerelease status.

Then ask explicitly: **"The current version is `<version>` and the release tag
will be `<tag>`. May I create and push this tag and publish the GitHub Release
with this version?"**

Wait for an unambiguous yes. Do not combine this question with tag or Release
creation in the same step. A response that changes the version is not approval;
repeat inspection and ask again with the new version.

## 4. Recheck and publish

After approval, recalculate the version, `HEAD`, branch, worktree state, and
remote branch SHA. Continue only if they exactly match the approved summary.

Use an annotated tag and push only that tag:

```bash
git tag -a "<tag>" "<approved-sha>" -m "Release <tag>"
git push origin "refs/tags/<tag>"
gh release create "<tag>" --verify-tag --title "<tag>" --generate-notes
```

For an explicitly approved prerelease, add `--prerelease`. For approved custom
notes, replace `--generate-notes` with `--notes-file <path>`.

Run each mutating command separately and inspect its result before continuing.
If local tag creation succeeds but the push fails, keep the tag and report the
failure. If the tag push succeeds but Release creation fails, report that the
tag is already public and retry only the Release step after resolving the
error. Never delete or retarget the tag as automatic cleanup.

## 5. Verify and report

Verify the exact remote tag and Release:

```bash
git ls-remote --tags origin "refs/tags/<tag>" "refs/tags/<tag>^{}"
gh release view "<tag>" --json url,tagName,name,isDraft,isPrerelease,targetCommitish
```

Report the version, tag, tagged commit, Release URL, and verification results.
