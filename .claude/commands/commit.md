---
description: Run pre-commit checks, code review, and create a git commit
allowed-tools: Bash(git:*), Bash(pre-commit:*), Bash(ruff:*), Bash(pyright:*)
---

## Context

- Git status: !`git status`
- Git diff (staged + unstaged): !`git diff HEAD`
- Branch: !`git branch --show-current`
- Recent commits: !`git log --oneline -5`

## Instructions

### Step 1: Pre-commit checks

Run `pre-commit run --all-files`. If it fails, fix issues and re-run until clean.

### Step 2: Code review

Spawn a Task subagent (general-purpose) as a strict code reviewer.
The reviewer receives the full diff from context above and evaluates:

- Is there unnecessary code being committed?
- Could this code be simpler? Could any abstraction, helper, or indirection be removed?
- Is there defensive code? (`or ""`, `or []`, `if x is not None` when x cannot be None, try/except around code that cannot fail, fallback defaults that mask bugs)
- Is there anything here that doesn't belong (debug prints, .env files, credentials)?
- Are there unnecessary comments, docstrings on obvious code, or over-engineering?

The reviewer is STRICT. Every suggestion must be addressed — you are NOT allowed
to ignore reviewer feedback. Output: "LGTM" or a numbered list of required changes.

### Step 3: Address feedback

If the reviewer flags ANY issues, fix ALL of them and re-run steps 1-2.
You must NOT skip or dismiss any reviewer suggestion.

### Step 4: Commit

- Stage relevant files (not .env, credentials, etc.)
- Commit message format: `claude(): short description in lowercase`
    - Always prefix with `claude()` to indicate the commit was made by Claude
    - Keep the description short and lowercase
    - Example: `claude(): add sequential block placement for dynamic machine`
- End with: `Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>`
- Use HEREDOC for the commit message:

```
git commit -m "$(cat <<'EOF'
claude(): short description here

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

- Do NOT push to the remote unless the user explicitly asks
