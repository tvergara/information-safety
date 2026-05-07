# CLAUDE.md

## CRITICAL: Ask Questions, Don't Assume

**When implementing anything, ask the user about ambiguous decisions instead of making silent assumptions.** The user cannot predict what choices you'll face during implementation. A quick question is always cheaper than redoing work.

Examples of things to ask about: data format choices, model selection, parameter values, architectural tradeoffs, naming conventions, edge case handling. Err heavily on the side of asking.

## CRITICAL: Never Inspect Data

**NEVER read, print, display, or inspect the contents of any data files (datasets, CSVs, JSONs, prompts, completions, etc.) in this project.** This is a safety research project that involves finetuning models on adversarial/unsafe data. Inspecting this data will trigger content policy violations and kill the session.

- **Do NOT** `cat`, `head`, `tail`, `Read`, or otherwise view data files
- **Do NOT** print dataset samples in tests — assert on structure/shape/length, not content
- **Do NOT** log or display prompt/completion text during training or evaluation
- **Do** trust that the data pipeline works based on metadata (lengths, counts, column names, dtypes)
- **Do** use opaque fixtures in tests (mock data loaders, assert shapes not values)

If you need to debug a data issue, inspect **metadata only** (number of rows, column names, dtypes, sequence lengths) — never the actual text content.

## CRITICAL: Test-Driven Development

**ALWAYS write tests BEFORE implementing any new feature or algorithm. DO NOT ASK — JUST DO IT.**

1. Write test cases that define the expected behavior
2. Run the tests to confirm they fail (red)
3. Implement the feature
4. Run the tests to confirm they pass (green)
5. Refactor if needed

Test file naming: `<module>_test.py` in `tests/` mirroring the source structure.

**Model testing policy:**

- **NEVER run real models in tests** — always use mocks/fixtures
- **NEVER require GPU/VRAM for tests** — mock any test that needs GPU
- Use `unittest.mock` to mock model loading and inference
- Environment-specific skips (`SLURM_JOB_ID`, cluster configs) are acceptable

## Spec-Driven Development for Non-Trivial Tasks

**For any task that touches multiple files, involves architectural decisions, or has non-obvious requirements, use the spec-driven workflow. Do NOT ask — recognize when it's needed and initiate it.**

**When to trigger:** Multi-file changes, new features, refactors, anything where "just start coding" would risk wasted effort or misalignment.

**When NOT to trigger:** Single-file bug fixes, typos, config tweaks, simple one-function additions with clear requirements.

### Workflow

1. **Announce** that this task warrants a spec. Briefly explain why.
2. **Explore** the codebase to understand current state (use Glob/Grep/Read).
3. **Draft the spec** to `.claude/specs/{descriptive-name}.md` using the template below.
4. **Drill acceptance criteria.** Present each criterion to the user and ask:
    - *"Is this the right bar? Too strict? Too loose?"*
    - *"What edge cases am I missing?"*
    - *"How will you verify this is done?"*
        Do NOT finalize the spec until the user has explicitly approved the acceptance criteria. This is the most important step — push back if criteria are vague.
5. **Delegate** to a fresh subagent via `Task()` with the spec file path. The subagent prompt should be: *"Read the spec at `.claude/specs/{name}.md` and implement it. Follow all rules in CLAUDE.md."* — nothing else. Clean context.
6. **Verify** the subagent's work: run tests, linter, type checker. If it fails, either fix directly or re-delegate with a narrower spec.
7. **Commit** after successful verification.

### Spec Template

```markdown
# Spec: {title}

## Goal
One sentence: what does this accomplish and why.

## Context
- Relevant files and their roles
- Current behavior (what exists today)
- How this fits into the broader system

## Requirements
Numbered list of concrete changes. Each requirement should be independently verifiable.

## Constraints
- What must NOT change
- Performance/memory bounds
- Compatibility requirements
- Edge cases to handle

## Test Plan
- What tests to write (file paths)
- What to mock
- Key assertions

## Acceptance Criteria
Checkboxed list. Each item must be:
- **Observable** — can be verified by running a command or reading output
- **Specific** — no ambiguity about pass/fail
- **Complete** — if all boxes are checked, the task is done

Example:
- [ ] `pytest tests/foo_test.py` passes
- [ ] `ruff check` reports no errors in changed files
- [ ] Function `bar()` returns X when given Y
- [ ] No regressions in `pytest tests/`
```

## Self-Verification

**IMPORTANT: Always verify your own work.** After making changes:

1. Run relevant tests (`pytest tests/path/to_test.py`)
2. Run linter (`ruff check --fix`)
3. Run type checker (`pyright`)
4. If tests or type checks fail, fix the root cause — do not suppress errors

**NEVER dismiss ANY failure as "pre-existing."** All code in this repo was generated by Claude. There is no such thing as someone else's problem. If you encounter a test failure, type error, lint error, or any other issue — even in files you didn't touch this session — fix it immediately. Do not work around it, note it, or defer it.

**NEVER skip or `--ignore` failing tests.** When a test fails, diagnose and fix the root cause — even if the failure predates your changes. Skipping a failing test is the same as hiding a bug. If the test is genuinely obsolete, delete it and explain why. If it tests behavior that changed, update the test to match the new behavior. The full test suite (`pytest tests/`) must pass clean before you declare your work done.

## Self-Improvement: Updating CLAUDE.md

**When you encounter a repeated error, a surprising project convention, or a non-obvious gotcha that could have been avoided with better instructions in this file, suggest adding a rule to CLAUDE.md.** Examples:

- An import pattern that keeps tripping you up
- A Hydra config convention that isn't obvious from the code
- A test pattern that requires specific mocking approaches
- A dependency quirk or environment setup issue

Phrase suggestions as: *"I hit \[problem\]. Should I add a rule to CLAUDE.md to prevent this in the future?"*

This keeps the file evolving as a living document that captures hard-won knowledge.

## Committing Code

Use `/commit` to commit changes. This runs pre-commit checks, a code review, and creates the commit automatically.

**NEVER skip the `/commit` skill.** Do not manually `git commit` to bypass the code review step — even for "obvious" or "small" fixes. Every commit must pass the review subagent. No exceptions.

## Code Style

- **Always use type annotations.** All function parameters, return types, and non-obvious variable types must be annotated. Enforced by `pyright` (basic mode) and ruff's `ANN` rules.
- **Avoid inline comments.** Code should be self-explanatory through clear naming and structure.
- Docstrings for public functions/classes are fine (documentation, not comments).
- Only add inline comments when the logic is truly non-obvious.
- Max line length: 99 characters.
- **NEVER write defensive code.** No `or ""`, no `or []`, no `if x is not None` when `x` cannot be `None`, no try/except around code that cannot fail. Trust the types. If a value can legitimately be `None`, fix the upstream code or type signature — don't paper over it at the call site.

## Development Setup

```bash
uv sync && . .venv/bin/activate
```

## Common Commands

```bash
# Tests
pytest                                    # All tests (includes doctests)
pytest tests/                             # Specific directory
pytest -n auto                            # Parallel
pytest --cov=information_safety           # With coverage

# Code quality
ruff check --fix
pyright
pre-commit run --all-files

# Experiments
python information_safety/main.py
python information_safety/main.py algorithm=llm_finetuning datamodule=your_datamodule
python information_safety/main.py debug=true
python information_safety/main.py cluster=mila
```

## Key Patterns

### Experiment Pipeline (CRITICAL)

**All experiments MUST write a result row to the centralized results file** (`/network/scratch/b/brownet/information-safety/results/final-results.jsonl`). This is the single source of truth for all experimental results. The row is written with `asr: null` initially, and the eval sweep job (`slurm/eval-sweep-results.sh`) backfills ASR later using the HarmBench classifier.

Every experiment must also save its generations (model responses) to `/network/scratch/b/brownet/information-safety/generations/{eval_run_id}/` as `input_data.jsonl` and `responses.jsonl`. The `eval_run_id` links the result row to its generations.

**Two experiment types, two code paths:**

1. **Training-based attacks** (finetuning, LoRA, data strategies): Use `AttackWithStrategy` Lightning module via `information_safety/main.py`. This handles training, generation, and result row writing automatically through Hydra configs.

2. **Prompt-based attacks** (zero-shot, few-shot, roleplay, jailbreak templates): Use `AttackWithStrategy` with a prompt strategy (e.g., `algorithm/strategy=roleplay`) via `information_safety/main.py`. Prompt strategies override `validation_step` to inject the attack template before generation.

**Do NOT** create separate results files, custom eval scripts, or one-off analysis notebooks that bypass this pipeline. If you're proposing a new experiment type, it must plug into this same results file and eval flow.

### Adding a New Algorithm

1. Write tests first in `tests/algorithms/your_algorithm_test.py`
2. Create `information_safety/algorithms/your_algorithm.py` (subclass `lightning.LightningModule`, accept `datamodule`)
3. Create config YAML in `information_safety/configs/algorithm/your_algorithm.yaml`
4. If custom training logic is needed, register with `@train_and_evaluate.register(YourAlgorithm)` in `experiment.py`

### Project Structure

- Configs: `information_safety/configs/` (Hydra, with `config.py` dataclass and YAML groups)
- Entry point: `information_safety/main.py` → `experiment.py` (`train_and_evaluate` uses `singledispatch`)
- Logs/SLURM output: `/network/scratch/b/brownet/information-safety/logs/`

**Hydra YAML: no duplicate top-level keys.** YAML silently overwrites duplicate keys. When adding new fields to an existing section, merge into the existing block — never create a second block with the same key.

### GPU Access for Debugging

**Before submitting a SLURM job for debugging, check if you already have an interactive GPU session.** Run `squeue -u brownet` to see active jobs. If there's an interactive session, use it directly — much faster iteration than submitting batch jobs. If no GPU is available, ask the user to allocate one.

If running outside Mila (for example, Narval), use your current user instead of a hardcoded username: `squeue -u $USER`.

### SLURM Jobs

**You can submit and monitor SLURM jobs directly.** Use `sbatch`, `squeue`, and read `slurm-*.out` files to debug.

**Never pipe long-running processes through `head`/`tail`.** The SIGPIPE from `head` will kill the process. Use background tasks and read output files instead.

**CRITICAL: Always benchmark before setting time limits.** Never guess job duration — submit a small benchmark job first (e.g., 5 samples on `main` partition), read the timing results, then extrapolate to set the real job's `--time`.

**Partitions:**

- **`main`**: Use for short/benchmark jobs — schedules faster. Default choice.
- **`long`**: Use for longer jobs that need more memory, GPU, or extended time limits.

### Disk Quota

**CRITICAL: HOME quota is ~100GB and often near-full.** Run `disk-quota` to check usage. If you hit `Disk quota exceeded`:

1. **STOP and ask the user** — do NOT try to clean up files yourself.
2. **Never write logs, results, checkpoints, or large outputs to HOME** (`/home/mila/b/brownet/`). Use SCRATCH (`/network/scratch/b/brownet/`) instead — it has 5TB.
3. SLURM output files (`-o` flag) should always point to SCRATCH: `-o /network/scratch/b/brownet/slurm-logs/slurm-%j.out`
4. Experiment results, model checkpoints, wandb data, etc. all go to SCRATCH.
5. Only source code, configs, and small files belong in HOME.

For Narval and other Compute Canada clusters, prefer portable paths:

1. Use `$SCRATCH` for large outputs instead of hardcoded Mila paths.
2. Put SLURM logs under `$SCRATCH/slurm-logs/`.
3. Use `quota -s` (or the cluster equivalent) to check storage usage.

### Cluster Environment

**Always load CUDA before running GPU code outside SLURM.** On the Mila cluster, run `module load cuda/12.4.1` before any command that imports DeepSpeed, vLLM, or other CUDA-dependent libraries. SLURM jobs handle this automatically, but interactive sessions and direct `python` invocations do not.

On Narval, module names can differ. Before loading CUDA, run `module avail cuda` and load an installed version that matches your environment.

## Cluster Policy

**Full-run experiments must go to Tamia or Nibi, not Mila.**

A "full run" is anything that invokes `python information_safety/main.py` to produce a real result row in the centralized `final-results.jsonl`. These belong on the Compute Canada clusters (Tamia or Nibi).

**Mila is still allowed for:**

- `pytest`, `ruff`, `pyright`, `pre-commit`
- Debug runs with 1–2 examples (e.g., `debug=true` or `algorithm.dataset_handler.max_examples=2`)
- `slurm/eval-sweep-results.sh` (HarmBench classifier scoring of existing generations — only when no eval jobs are queued/running, since its rewrite clobbers concurrent appends)

**SSH ControlMaster flow.** Tamia and Nibi require keyboard-interactive 2FA via the Compute Canada mobile app. Non-interactive submission only works while a ControlMaster socket is live in `~/.ssh/cm-%r@%h:%p`. Before any Tamia/Nibi work, run:

```bash
bash slurm/check-cluster-ssh.sh tamia nibi
```

If either cluster reports closed, the ControlMaster must be re-established. Claude Code's `!` prefix does **not** allocate a PTY, so running `ssh <cluster> true` directly in the session will fail with "Permission denied (keyboard-interactive)". The fix is to spawn Python, which opens a PTY internally via `pty.fork()` — the outer shell does not need to be a terminal. Replace both occurrences of `CLUSTER` below with `tamia` or `nibi`:

```bash
# Replace CLUSTER with tamia or nibi (two places: cluster= and the log path).
! nohup python3 -c "
import pty, os, time, select

cluster = 'CLUSTER'
pid, fd = pty.fork()
if pid == 0:
    os.execlp('ssh', 'ssh', '-N', cluster)
else:
    output = b''
    deadline = time.time() + 60
    while time.time() < deadline:
        r, _, _ = select.select([fd], [], [], 1)
        if r:
            try:
                chunk = os.read(fd, 4096)
                output += chunk
                if b'Duo' in output or b'push' in output.lower() or b'passcode' in output.lower():
                    os.write(fd, b'1\n')
                    time.sleep(15)
                    break
            except OSError:
                break
    os.waitpid(pid, os.WNOHANG)
" > /tmp/ssh_cm_CLUSTER.log 2>&1 &
```

Then:

1. Tell the user to open the Compute Canada mobile app and **approve the Duo push notification immediately**.
2. Wait ~15 seconds, then run `bash slurm/check-cluster-ssh.sh CLUSTER` to confirm the socket is live.
3. Check `/tmp/ssh_cm_CLUSTER.log` if it fails — look for auth errors vs timeout.

The `ssh -N` child process is left running as a nohup orphan, keeping the ControlMaster socket (`~/.ssh/cm-tvergara@CLUSTER.alliancecan.ca:22`) live. All subsequent SSH/rsync/sbatch calls reuse it silently.

**Helper scripts:**

- `slurm/check-cluster-ssh.sh <cluster> [<cluster> ...]` — probes the ControlMaster sockets and exits non-zero if any are closed.
- `slurm/bootstrap-remote-cluster.sh <tamia|nibi>` — idempotent setup on the remote cluster: clones (or pulls) the repo, runs `git submodule update --init --recursive`, installs `uv` if missing, and runs `uv sync`. Re-runs become `git pull` + `uv sync`. **Use this (not raw `git pull` via robot) for any pull on Tamia/Nibi** — it unlocks `information_safety/configs`, pulls, restores any zero-byte YAMLs, and re-locks the directory read-only as a Lustre-truncation guardrail.
- `slurm/sync-and-merge-from.sh <tamia|nibi>` — pull results, merge by `eval_run_id` (preserves backfilled ASR), symlink generations into the canonical location. Refuses to run if `eval-sweep` is already queued/running. Does not auto-submit eval — prints the `sbatch` command to run when ready.
- `slurm/sync-results-from.sh <tamia|nibi>` — raw rsync only (used by the wrapper above; rarely run on its own).

**Hydra cluster configs:** `cluster=tamia` and `cluster=nibi` inherit from `mila.yaml` (same shape as `narval.yaml`), set the cluster hostname, and disable internet on compute nodes.

### Submitting jobs via the robot/automation node (preferred)

**Use the robot node for all non-interactive work — `sbatch`, `squeue`, `rsync`, file ops.** It does not require Duo, so it does not depend on a live ControlMaster. The ControlMaster flow above (`ssh tamia` / `ssh nibi`) is only needed for interactive shells, editing files in-place, running `python` directly, pulling the repo (must go through `bootstrap-remote-cluster.sh`, see below), or other commands the wrapper rejects.

**Verified status (as of 2026-05-04):**

- ✅ `robot.tamia.ecpia.ca` — working.
- ⚠️ `robot.nibi.alliancecan.ca` — host is in the ssh config but the robot grant has not been confirmed. `ssh robot.nibi.alliancecan.ca "squeue --me"` currently fails. Smoke-test before relying on it; if it fails, ask support to extend the automation grant to Nibi.

**Connection:** `~/.ssh/config` has a single `Host` stanza enumerating `robot.tamia.ecpia.ca`, `robot.nibi.alliancecan.ca`, `robot.narval.alliancecan.ca`, `robot.fir.alliancecan.ca`, `robot.rorqual.alliancecan.ca` — pointing at `~/.ssh/id_ed25519_alliance_robot` with `PreferredAuthentications publickey,keyboard-interactive` and `RequestTTY no`. Just run:

```bash
ssh robot.tamia.ecpia.ca "<command>"
```

No flags, no Duo prompt. The `keyboard-interactive` entry in the preferred-auth list is **required** — Alliance sends a zero-prompt kbd-int probe (`num_prompts 0`) as a no-op bypass step. Disabling kbd-int breaks the connection with `Permission denied (keyboard-interactive)`. Don't.

**Wrapper restrictions** (`allowed_commands.sh`):

- **Single commands only.** Compound commands with `;`, `&&`, or `||` are rejected (`Command rejected by allowed_commands.sh: ...`). Run one ssh per command, or wrap multiple steps in an `sbatch` script.
- Allowed: file transfers (`scp`, `sftp`, `rsync`), file ops (`mv`, `cp`, `rm`), archiving (`gzip`, `tar`), `git`, Slurm (`squeue`, `sbatch`, `scancel`, `scontrol`, `sq`).
- **Not allowed:** interactive shells, `bash`, `python`, arbitrary executables. For those, use `ssh tamia` / `ssh nibi` (the ControlMaster-backed aliases above).

**Typical job-submission pattern.** The robot's `allowed_commands.sh` wrapper does **not** expand variables — it executes the literal argv. Pass an absolute path (`/scratch/t/tvergara/...` for Tamia, `/scratch/<initial>/<user>/...` for Nibi); `$SCRATCH`, `~`, and `$HOME` will be passed through unexpanded and the script will crash with "No such file or directory":

```bash
bash slurm/bootstrap-remote-cluster.sh tamia      # pulls + restores + locks configs
ssh robot.tamia.ecpia.ca "sbatch /home/t/tvergara/information-safety/slurm/run-job-pool.sh /scratch/t/tvergara/information-safety/job-pool/run-1"
ssh robot.tamia.ecpia.ca "squeue --me"
```

**Lustre HOME silently truncates working-tree files.** After a successful `git pull` on Tamia/Nibi, tracked YAMLs occasionally end up at zero bytes (no error, mtime advances normally). Empirically the truncation is rare and asymmetric — typically 4 of 6 recently-updated YAMLs in a single pull. Two layers of protection:

1. `slurm/bootstrap-remote-cluster.sh` runs `git checkout HEAD --` on any zero-byte YAML and then `chmod -R a-w information_safety/configs` to lock the directory read-only.
2. `slurm/run-job-pool.sh` repeats the restore at SLURM job startup as defense-in-depth (in case truncation happens between pull and worker launch).

Do **not** use `ssh robot ... "git -C ~/information-safety pull"` — it bypasses both layers and your next SLURM job will explode with `hydra.errors.ConfigCompositionException`.

**Source IP constraint.** The CCDB key is registered with `restrict,from="64.15.78.*",command="/cvmfs/soft.computecanada.ca/custom/bin/computecanada/allowed_commands/allowed_commands.sh"`. The `from=` clause pins the key to the Mila login outbound range. If Mila ever changes its outbound IP block, the key will silently stop working — `ssh -vvv` will show the key not even being offered. `curl ifconfig.me` from Mila login confirms current public IP.

### Job Pool (fat-allocation scheduler)

Tamia/Nibi force whole-node allocations (4xH100). To run our backlog of single-GPU experiments efficiently, use the job pool:

1. **Build the queue** (idempotent):

    ```bash
    python scripts/build_job_queue.py --queue-root "$SCRATCH/information-safety/job-pool/run-1"
    ```

    This reads `final-results.jsonl`, finds missing configs from `scripts/check_results.py`, dedups `DataStrategy` epochs (per `(model, max_examples, max_epochs)`), and writes one `pending/<id>.json` per job. Re-running on the same `--queue-root` is a no-op (deterministic ids).

    **DataStrategy is compute-matched.** The grid in `scripts/check_results.py` enumerates `(max_examples, max_epochs)` pairs whose product is fixed at 1024 examples-seen — currently `[(16, 64), (32, 32), (64, 16), (128, 8), (256, 4), (512, 2)]`. Each pair gets `max_epochs` rows (one per validation epoch), and the queue producer emits one job per `(model, max_examples, max_epochs)` triple with a `trainer.max_epochs=<N>` Hydra override. The default `max_epochs: 2` in `experiment/finetune-with-strategy.yaml` is only a fallback for non-pool callers.

2. **Submit the pool**:

    ```bash
    sbatch slurm/run-job-pool.sh "$SCRATCH/information-safety/job-pool/run-1"
    ```

    Allocates a whole node, spawns 4 workers (one per GPU), each drains the queue via atomic `os.rename` claims.

3. **Resume after time-limit**: re-`sbatch` the same `QUEUE_ROOT` — remaining `pending/*.json` files are picked up.

**Queue states:** `pending/` (unclaimed) → `claimed/<id>.<worker>.json` (running) → `done/<id>.json` or `failed/<id>.json`. Per-job stdout/stderr lives in `logs/<id>.<worker>.log`.

## Periodic Self-Reflection

**After completing a significant task (launching a job, finishing a feature, debugging a hard issue), pause and ask:**

1. Did anything fail that could have been prevented with a rule in this file?
2. Did I repeat any mistake more than once?
3. Is there a non-obvious project convention I discovered that future sessions should know?

If yes to any, update this file immediately. Do not wait for the user to ask.
