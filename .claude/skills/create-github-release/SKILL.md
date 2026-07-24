---
name: create-github-release
description: >-
  Safely create and push SemVer Git tags such as v1.0.0, then publish matching
  GitHub Releases with curated notes using gh. Use when the user asks to tag a
  version, cut or publish a release, or create a GitHub Release, including a
  coordinated rgi_utils and sibling *_restr release. Discover eligible *_restr
  repositories and explicitly ask which to include. Always inspect and report
  each repository's current version, proposed tag, and target commit, then
  obtain the user's explicit final approval before creating any tag or Release.
  If a current version already has both a tag and a GitHub Release, ask whether
  to increment the version.
---

# Create a GitHub Release

Create one immutable version tag and its matching GitHub Release, or a
user-selected coordinated set of them. Treat the version declared by each
repository, its Git tag, and its Release tag as one version.

## Hard rules

- Always perform the mandatory approval gate below. The user's initial request
  to create a release does not count as final approval.
- Before approval, do not create a local or remote tag and do not create a
  draft or published Release.
- For an RGI release, always ask whether to include detected sibling Git
  repositories whose names end in `_restr`. Do not infer the answer from the
  initial release request.
- If the current version already has both its tag and GitHub Release, stop and
  ask whether to increment the version.
- Inspect and approve every repository independently. Never assume companion
  repositories use the primary repository's version, tag, branch, or commit.
- Write complete, evidence-based release notes and show the full draft before
  requesting approval.
- Never guess the version, silently change a manifest version, move an existing
  tag, overwrite a Release, force-push, or delete a local or remote tag.
- Invalidate approval and ask again if the version, target commit, branch,
  release-note content, or repository state changes after approval.
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

## 2. Choose companion `*_restr` repositories

When releasing `rgi_utils`, inspect its immediate sibling directories whose
names end in `_restr`. Do not use `find`. Retain only directories that resolve
to distinct Git worktrees, and resolve each repository's exact GitHub
`owner/name` from its remote.

Show the detected repository list and ask in the user's language, equivalent
to: **"Should I also create releases for these `*_restr` repositories? Reply
with all, specific repository names, or none."**

Wait for the answer before preparing releases. This companion-selection answer
is not the final publication approval. Treat the primary repository and the
selected companions as the release set. Apply every inspection, existing-state
check, verification, release-note, approval, recheck, and publication rule in
this skill to each member independently. If a selected repository has no
authoritative version, has an ambiguous release state, or needs a version bump,
ask for that repository's exact disposition; allow the user to remove it from
the release set.

Skip this section when there are no sibling Git repositories ending in
`_restr`.

## 3. Handle an existing current version

If both the exact tag for the authoritative current version and its GitHub
Release already exist, report the version, tag, tagged commit, and Release URL.
Then ask in the user's language, equivalent to:
**"A tag and GitHub Release already exist for the current version `<version>`.
Do you want to increment the version?"**

Wait for the answer. Do not infer whether to make a major, minor, or patch
increment. If the user wants an increment but did not give the exact new
version, ask which version to use. Update version declarations only after that
choice, then restart inspection from Step 1. The later mandatory approval gate
still applies to the new version; approval to increment is not approval to tag
or publish it.

If only the tag or only the Release exists, report the partial state and ask
whether to repair the missing artifact or increment the version. Never
overwrite or retarget the existing artifact.

## 4. Verify before approval

Run the repository-prescribed lint and non-GPU tests. Do not publish a release
with failing checks unless the user explicitly accepts the named failures after
seeing them.

Confirm mechanically that:

- the exact proposed tag is absent locally and on `origin`;
- no GitHub Release already uses that tag;
- the remote branch points at the proposed target commit;
- the target commit and version declaration contain the intended release
  contents.

## 5. Write release notes

Before the approval gate, write a complete Markdown release-note draft for each
repository to a separate temporary path outside every worktree, such as
`/tmp/<repo>-<tag>-release-notes.md`. Determine the previous published Release
or version tag and inspect the complete range through the target commit:

- commits and relevant diffs between the previous tag and target commit;
- merged PR titles, descriptions, links, and contributors when available;
- user-facing documentation and migration instructions affected by the range.

For a first release, derive the summary from the README, public documentation,
and repository history. GitHub-generated notes may be used as a source and
checklist, but never publish them without reviewing and rewriting them.

Write in the repository documentation language. Start with a short release
summary, then group user-visible changes under meaningful headings such as
`Breaking Changes`, `Features`, `Fixes`, and `Documentation`. Omit empty
sections. Add upgrade or migration instructions when needed, credit
contributors, link relevant PRs or issues, and add a full-changelog comparison
link when a previous tag exists. Do not invent claims, dump raw commit subjects,
or fill the notes with implementation details that do not affect users.

Incorporate any notes supplied by the user, then verify every claim against the
release range. Show the full draft in the mandatory approval summary. Any
subsequent edit to the notes invalidates approval and requires approval again.

## 6. Mandatory approval gate

Present one approval summary for the complete release set in the user's
language. For every repository, include:

- repository;
- version found in the authoritative file and that file's path;
- latest existing version tag, or `none`;
- proposed new tag;
- full target commit SHA and branch;
- whether the target is pushed and the worktree is clean;
- verification results;
- Release title, the full release-note draft, and stable/prerelease status.

Then ask explicitly in the user's language, equivalent to:
**"May I create and push all tags listed above and publish all corresponding
GitHub Releases with the shown versions and release notes?"**

Wait for an unambiguous yes. Do not combine this question with tag or Release
creation in the same step. A response that changes the version is not approval;
repeat inspection and ask again with the new version. Approval for only part of
the release set authorizes only that explicitly named subset; rebuild and show
the summary for the reduced set before publishing.

## 7. Recheck and publish

After approval, recalculate the version, `HEAD`, branch, worktree state, remote
branch SHA, and release-note content for every approved repository before
creating any tag. Continue only if the complete set exactly matches the
approved summary.

Use explicit repository paths and GitHub `owner/name` values. For each
repository, create an annotated tag and push only that tag:

```bash
git -C "<repo-path>" tag -a "<tag>" "<approved-sha>" -m "Release <tag>"
git -C "<repo-path>" push origin "refs/tags/<tag>"
gh release create "<tag>" -R "<owner/name>" --verify-tag --title "<tag>" --notes-file "<notes-file>"
```

For an explicitly approved prerelease, add `--prerelease`. Use the exact
approved notes file without modifying it after approval.

Run each mutating command separately and inspect its result before continuing.
If local tag creation succeeds but the push fails, keep the tag and report the
failure. If the tag push succeeds but Release creation fails, report that the
tag is already public and retry only the Release step after resolving the
error. Never delete or retarget the tag as automatic cleanup. Coordinated
publication is not atomic: if any repository fails, stop before mutating the
next repository, report which repositories completed, and ask whether to
continue the remaining approved set after the failure is resolved.

## 8. Verify and report

Verify every exact remote tag and Release:

```bash
git -C "<repo-path>" ls-remote --tags origin "refs/tags/<tag>" "refs/tags/<tag>^{}"
gh release view "<tag>" -R "<owner/name>" --json url,tagName,name,isDraft,isPrerelease,targetCommitish
```

For every repository, report the version, tag, tagged commit, Release URL, and
verification results.
