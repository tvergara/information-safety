---
description: Critically analyze recent commits and refactor to remove unnecessary code
---

## Instructions

This is an autonomous refactoring process. Run it end-to-end without user involvement.

### Step 1: Baseline

Count lines of code in all .py files touched by the last ~15 commits:

```
git diff --name-only HEAD~15..HEAD -- '*.py' | xargs wc -l
```

Save this as the "before" count (exclude test files from the removal goal).

### Step 2: Critical analysis

Spawn a Task subagent (general-purpose, model opus) to perform a deep critical analysis.

The subagent should:

1. Run `git log --oneline -15` to see the commits

2. Run `git diff --name-only HEAD~15..HEAD -- '*.py' '*.yaml'` to identify all files

3. Read EVERY file involved (full contents, not just diffs)

4. Produce a detailed critique organized by file, identifying:

    **(a) Defensive code** — unnecessary None checks, redundant assertions, overly cautious
    error handling, dead branches that can never trigger. We do NOT want defensive code.

    **(b) Dead/useless code** — metrics nobody reads, variables assigned but unused,
    functions that exist "just in case", config options that are always the same value,
    code that was added speculatively but never integrated.

    **(c) Untested logic** — branches or functions in source files that have no
    corresponding test coverage. List specific functions/branches.

    **(d) Duplicated code** — copy-pasted blocks, near-identical functions,
    logic that could be unified. Include line numbers.

    **(e) Over-engineering** — abstractions for one-time operations, unnecessary
    indirection, premature generalization, config options that add complexity
    for no real benefit.

    **(f) Stale code** — anything that was relevant for earlier iterations but
    is now dead weight given the current state of the codebase.

The subagent should output a numbered list of concrete, actionable refactoring
items, each with file path, line numbers, and a clear description of what to
change. Be ruthless — the goal is maximum simplification.

### Step 3: Attack the issues

Work through the critique items one by one:

- For each item, make the change
- Preserve functionality (unless it's genuinely dead/useless)
- After each batch of related changes, run the relevant tests to verify:
    `pytest <test_file> -v --tb=short`
- Track lines added and removed as you go
- Test files are allowed to GROW — only source code lines matter for the removal goal
- Do NOT add docstrings, comments, or type annotations to code you didn't change

### Step 4: Fix all failing tests and type errors

Run the full test suite and type checker across the **entire repository** (not just files touched by recent commits). Fix every failure:

```
. .venv/bin/activate && module load cuda/12.4.1 && pytest tests/ -v
```

```
ruff check --fix
pyright
```

**CRITICAL RULES:**

- **Zero failures.** The target is 0 failed, 0 skipped, 0 xfailed. Every skip/xfail is a test that isn't running — investigate and fix it. The only acceptable skips are environment-impossible ones (e.g., "shouldn't run on cluster" when you're on a cluster).
- **NEVER dismiss failures as "pre-existing."** If it fails, fix it. Period.
- **Install missing dependencies.** If a test fails because a package is missing, install it with `uv pip install`. Do NOT work around it with `TYPE_CHECKING`, `try/except`, or skip markers. Fix the actual problem.
- **Fix root causes, not symptoms.** If an `isinstance` check fails, investigate why. If an import fails, install the package. Do not add workarounds.
- If any test fails, fix the root cause (in source or test code as appropriate).
- If pyright reports type errors, fix them.
- If ruff reports issues, fix them.
- Re-run all three until everything passes cleanly.
- Do NOT suppress errors, add `type: ignore`, or skip tests to make them pass.

### Step 4b: Smoke-test infrastructure changes on GPU

**If you changed infrastructure code (launch.py, vllm0.py, or anything that spawns processes/servers), you MUST smoke-test it.** You have GPUs — use them.

- Check `squeue -u $USER` — if you already have an interactive GPU session, use it. **NEVER schedule new SLURM jobs if you already have GPUs.**
- For vLLM/launch changes: at minimum, verify the server starts and responds to a health check.
- For training loop changes: run a minimal training step.
- If a changed function has no test coverage, write a test for it before committing.

### Step 5: Commit

Use `/commit` to commit all changes.

### Step 6: Report

Output a summary to the user:

```
## Refactor Report

**Lines of code removed (net, source only):** X
**Files modified:** Y

### Top issues found:
1. ...
2. ...
3. ...

### Changes made:
- file.py: description of change (+A/-B lines)
- ...
```
