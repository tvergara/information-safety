---
description: Push local paper/ changes to Overleaf
allowed-tools: Bash(git:*), Read, Glob, Grep
---

## Instructions

Push local paper file changes to the Overleaf git remote.

### Step 1: Fetch latest

```bash
git fetch overleaf
```

### Step 2: Create a temporary worktree

```bash
git worktree add /tmp/overleaf-push overleaf/master
```

**CRITICAL:** Do NOT use `git stash` or `git checkout` to switch branches — this will lose uncommitted paper/ changes.

### Step 3: Copy changed files

Copy the paper files that have been modified from `paper/` to the worktree root:

```bash
cp paper/main.tex /tmp/overleaf-push/main.tex
cp paper/references.bib /tmp/overleaf-push/references.bib
cp paper/macros.tex /tmp/overleaf-push/macros.tex
cp paper/neurips_2025.sty /tmp/overleaf-push/neurips_2025.sty
```

Also copy any new image files that should be on Overleaf. Only copy files that have actually changed — check with `diff` first if unsure.

### Step 4: Review the diff

```bash
cd /tmp/overleaf-push && git diff --stat
```

Show the user the summary and ask for confirmation before pushing.

### Step 5: Commit and push

```bash
cd /tmp/overleaf-push
git add .
PRE_COMMIT_ALLOW_NO_CONFIG=1 git commit -m "description of changes"
git push overleaf HEAD:master
```

Note: `PRE_COMMIT_ALLOW_NO_CONFIG=1` is needed because the Overleaf repo has no `.pre-commit-config.yaml`.

### Step 6: Clean up

```bash
git worktree remove /tmp/overleaf-push
```

### Step 7: Report

Tell the user the push succeeded and what was updated.

## Key rules

- Paper files live in `paper/` locally but at ROOT on Overleaf
- ALWAYS use a git worktree — never switch branches on master
- NEVER use `git stash` during this process — uncommitted paper changes will be lost
- NEVER use `git merge` or `--allow-unrelated-histories`
- The `cd` into `/tmp/overleaf-push` will be reset by the shell — use full paths or chain commands
