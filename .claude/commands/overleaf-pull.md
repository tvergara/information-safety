---
description: Pull latest changes from Overleaf into paper/
allowed-tools: Bash(git:*), Read, Glob, Grep, Edit
---

## Instructions

Pull the latest paper files from the Overleaf git remote into `paper/`.

### Step 1: Fetch

```bash
git fetch overleaf
```

### Step 2: List files on Overleaf

```bash
git ls-tree --name-only overleaf/master
```

### Step 3: Extract text files

For each `.tex`, `.bib`, and `.sty` file, extract it:

```bash
git show overleaf/master:FILENAME > paper/FILENAME
```

**Exception:** `macros.tex` may be empty on Overleaf. If so, do NOT overwrite the local version — it contains comment macro definitions (`\marius{}`, `\response{}`, `\tom{}`).

### Step 4: Extract image/binary files

For each `.png`, `.jpg`, `.pdf` (not `main.pdf`) file on Overleaf:

```bash
git show overleaf/master:FILENAME > paper/FILENAME
```

**Unicode filenames:** Some files may have unicode characters in their names (e.g. narrow no-break space). If `git show` fails on a filename, use `git ls-tree -z` to get raw bytes, then extract with `printf` for the filename. Rename to a clean ASCII name locally and update the `\includegraphics` reference in the `.tex` file.

### Step 5: Verify compilation

```bash
cd paper && ~/.local/bin/tectonic main.tex
```

If compilation fails, diagnose and fix (missing packages, bad filenames, undefined macros, etc.).

### Step 6: Report

Tell the user what changed (new files, modified files) and whether compilation succeeded.

## Key rules

- Paper files live in `paper/` locally but at ROOT on Overleaf
- Do NOT use `git merge` or `--allow-unrelated-histories` — the histories are unrelated and a merge would try to delete all code files
- Do NOT stash or checkout branches — paper/ files are often uncommitted and will be lost
