# Spec: Cluster Job Pool (fat-allocation scheduler for Tamia)

## Goal

Allow us to run our backlog of single-GPU experiments on Tamia, which forces
whole-node (4×H100) allocations. One SLURM job grabs the node and runs 4
single-GPU sub-jobs in parallel from a queue. Future-proofed so the same
machinery works on Nibi/Narval if those also enforce whole-node policies.

## Context

- **`scripts/check_results.py`** declares the full grid of desired
    `(experiment_name, dataset_name, model_name, max_examples, epoch)` configs
    and reports which are missing from `final-results.jsonl`.
- **`slurm/sweep.sh`** shows the canonical Hydra-override shape per
    `experiment_name`. Each `check_results.py` config maps cleanly to one of
    three command shapes:
    - `BaselineStrategy` / `RoleplayStrategy` → `experiment=prompt-attack` +
        `algorithm/strategy=baseline|roleplay`
    - `DataStrategy` → `experiment=finetune-with-strategy` +
        `algorithm/strategy=data` + `algorithm.dataset_handler.max_examples=<n>`
    - `Precomputed{GCG,AutoDAN,PAIR}Strategy` → `experiment=prompt-attack` +
        `algorithm/strategy=precomputed_<attack>` +
        `algorithm.strategy.suffix_file=<convention-based path>`
- **`final-results.jsonl`** is the single source of truth. Workers append to
    it via `O_APPEND`; appends under `PIPE_BUF` (~4KB) are POSIX-atomic, so no
    locking is required for concurrent writes from 4 workers on one node.
- **`DataStrategy` invocations write multiple rows** (one per epoch). The
    queue producer must dedup at `(experiment_name, dataset_name, model_name, max_examples)` so we don't enqueue two jobs for `(DataStrategy, llama2, max_ex=10)` just because both `epoch=0` and `epoch=1` are missing.
- **Cluster policy** (`CLAUDE.md`): full-run experiments must be submitted
    from Tamia or Nibi. This script lives in the repo and is invoked from the
    cluster.

## Requirements

### 1. Queue producer: `scripts/build_job_queue.py`

A new script that:

1. Imports `HARMBENCH_CONFIGS` and `WMDP_CONFIGS` from `check_results.py`.
2. Reads `final-results.jsonl`.
3. Dedups configs at the level
    `(experiment_name, dataset_name, model_name, max_examples)`.
    For `DataStrategy`, only the last epoch (`epoch=1`) needs to match — if
    the last-epoch row exists, the job is considered done.
4. For `Precomputed{GCG,AutoDAN,PAIR}Strategy` rows, resolves
    `suffix_file` via the convention
    `attacks/<attack_lower>-<model_slug>.jsonl`,
    where `<model_slug>` is the model path with `/` → `_`. If multiple files
    match a glob `attacks/<attack_lower>-<model_slug>-*.jsonl`, picks the
    newest by mtime. If none match, **skips that row** with a printed warning
    (rather than emitting a broken job).
5. Maps each remaining missing config to a Hydra command (see "Command
    Templates" below).
6. Emits a directory of one-file-per-job under `<queue_root>/pending/`,
    each file named `<id>.json` containing:
    ```json
    {"id": "<deterministic-id>", "command": ["python", "main.py", ...],
     "config": {original config dict}}
    ```
    The `id` is a stable hash of the config dict (so re-running the producer
    on the same backlog produces the same filenames; safe to re-run).

### 2. Worker: `scripts/job_pool_worker.py`

A new script invoked once per GPU. Each worker:

1. Reads `WORKER_INDEX` and `QUEUE_ROOT` from environment.
2. Sets `CUDA_VISIBLE_DEVICES=$WORKER_INDEX` and inherits other env.
3. Loops:
    - List `<queue_root>/pending/*.json` (sorted by mtime for determinism).
    - For each, attempt `os.rename` from
        `pending/<id>.json` → `claimed/<id>.<worker_index>.json`. On
        `FileNotFoundError`, another worker grabbed it; continue. On success,
        this worker owns it.
    - Run the command with `subprocess.run`, capturing stdout/stderr to
        `<queue_root>/logs/<id>.<worker_index>.log`.
    - On exit code 0: rename to `done/<id>.json`.
        On non-zero: rename to `failed/<id>.json` (with the log preserved).
    - Continue until `pending/` is empty.
4. Exits 0.

The atomic claim relies on `os.rename` being atomic on the same filesystem
(POSIX guarantees this for same-fs renames, including SLURM-managed scratch
NFS).

### 3. SLURM driver: `slurm/run-job-pool.sh`

A new SLURM script that:

1. Allocates a whole Tamia node: `#SBATCH --gres=gpu:h100:4 --nodes=1 --ntasks=1 --cpus-per-task=<N> --mem=<M> --time=<T>`. Account and
    partition parameterized via env vars (`ACCOUNT`, `PARTITION`).
2. Takes `QUEUE_ROOT` as `$1` (or env). Defaults to a path under scratch.
3. Activates the project venv and `module load cuda/...`.
4. Spawns 4 background workers with `WORKER_INDEX=0..3` and `wait`s for all.
5. Prints a summary at the end (`done` / `failed` counts).

### 4. Tests

- `tests/scripts/build_job_queue_test.py` — unit tests, no real cluster:
    - Given a synthetic `final-results.jsonl` and the real CONFIGS, asserts
        correct missing/present partitioning.
    - Asserts dedup at `(experiment, model, max_ex)` for DataStrategy.
    - Asserts Precomputed-\* `suffix_file` resolution (convention path,
        fallback to glob+mtime, skip-with-warning when nothing matches).
    - Asserts each emitted command is a non-empty `list[str]` and starts with
        `python information_safety/main.py`.
- `tests/scripts/job_pool_worker_test.py` — integration test with **mock**
    subprocess (no real model runs):
    - Spawn 4 workers against a `tmp_path` queue with N jobs (N >> 4).
    - Assert no double-claims (every `done` file appears exactly once across
        workers).
    - Assert each job's `<id>` ends up in either `done/` or `failed/`.
    - Mock `subprocess.run` to occasionally fail; assert failures land in
        `failed/`.

### 5. Documentation

- Update `CLAUDE.md` with a one-line pointer under "Cluster Policy" or a
    new "Job Pool" subsection: how to invoke `build_job_queue.py` then
    `sbatch slurm/run-job-pool.sh`, and what "queue states" mean.

## Command Templates

The mapping from config dict → Hydra args. Reference: `slurm/sweep.sh`.

```python
# Common to all
COMMON = [
    "experiment=prompt-attack",  # overridden for DataStrategy
    "algorithm/dataset_handler={dataset_handler}",  # advbench_harmbench or wmdp
    "algorithm.model.pretrained_model_name_or_path={model}",
    "algorithm.model.trust_remote_code={trust_remote_code}",
    "trainer.precision=bf16-mixed",
    "name={name}",
]
```

Per-strategy specializations:

| `experiment_name`            | `experiment=`            | extra Hydra args                                                                                                                                                                             |
| ---------------------------- | ------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `BaselineStrategy`           | `prompt-attack`          | `algorithm/strategy=baseline`                                                                                                                                                                |
| `RoleplayStrategy`           | `prompt-attack`          | `algorithm/strategy=roleplay`                                                                                                                                                                |
| `DataStrategy`               | `finetune-with-strategy` | `algorithm/strategy=data` `algorithm.strategy.r=16` `algorithm.strategy.lora_alpha=16` `algorithm.dataset_handler.train_data_path=<TRAIN_DATA>` `algorithm.dataset_handler.max_examples=<n>` |
| `PrecomputedGCGStrategy`     | `prompt-attack`          | `algorithm/strategy=precomputed_gcg` `algorithm.strategy.suffix_file=<path>`                                                                                                                 |
| `PrecomputedAutoDANStrategy` | `prompt-attack`          | `algorithm/strategy=precomputed_autodan` `algorithm.strategy.suffix_file=<path>`                                                                                                             |
| `PrecomputedPAIRStrategy`    | `prompt-attack`          | `algorithm/strategy=precomputed_pair` `algorithm.strategy.suffix_file=<path>`                                                                                                                |

`trust_remote_code` is `true` iff model slug ∈ `{safety-pair-safe, safety-pair-unsafe, smollm3-circuit-breaker, Olmo-3-7B-Instruct}` (mirrors
`slurm/sweep.sh`).

`dataset_handler` is `wmdp` for WMDP rows, `advbench_harmbench` for
HarmBench rows.

## Constraints

- **Must not break existing single-job submission paths.** `slurm/sweep.sh`,
    `slurm/train-safety-pair.sh`, etc. continue to work unchanged.
- **Must not write to HOME.** Queue/logs/results go to scratch only
    (`/network/scratch/b/brownet/...` on Mila;
    `$SCRATCH/...` on Tamia).
- **Cluster-portable scratch path**: use `$SCRATCH` if set, else fall back
    to the Mila scratch path. The driver script should resolve this once.
- **No mocked GPUs in tests.** Per CLAUDE.md, model invocations in tests
    must be mocked at the subprocess level.
- **Reentrant queue producer.** Re-running `build_job_queue.py` against an
    existing `<queue_root>/pending/` should not duplicate jobs (filename is
    the deterministic hash).
- **Workers must not crash the whole allocation if one job fails.** A
    Python exception in the subprocess goes into `failed/` and the worker
    picks the next item.
- **No defensive code.** Trust types. If a config dict is missing a
    required field, that's a bug in `check_results.py` and should crash loudly.

## Test Plan

| Test file                               | Mocks                                                     | Key assertions                                                                                          |
| --------------------------------------- | --------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| `tests/scripts/build_job_queue_test.py` | filesystem (`tmp_path`); fake `final-results.jsonl`       | correct partitioning; DataStrategy epoch dedup; Precomputed `suffix_file` resolution; deterministic IDs |
| `tests/scripts/job_pool_worker_test.py` | `subprocess.run` patched to a no-op or controlled failure | no double-claims; every job lands in done/failed; failures preserve logs                                |

No GPU, no real model loads. Tests run in \<5 seconds total.

## Acceptance Criteria

- [ ] `pytest tests/scripts/build_job_queue_test.py` passes.
- [ ] `pytest tests/scripts/job_pool_worker_test.py` passes.
- [ ] `pytest tests/` (full suite) passes with no regressions.
- [ ] `ruff check` reports no errors in changed files.
- [ ] `pyright` reports no errors in changed files.
- [ ] Running `python scripts/build_job_queue.py --queue-root /tmp/queue-smoke`
    against the live `final-results.jsonl` produces a `pending/` directory
    with at least one `*.json` file (smoke test, no SLURM submission).
- [ ] `pending/*.json` files contain a `command` list whose elements are
    strings, and the first three are `["python", "information_safety/main.py", "experiment=..."]`.
- [ ] Re-running `build_job_queue.py` against the same `--queue-root` does
    not increase the count of `pending/*.json` (deterministic IDs).
- [ ] `slurm/run-job-pool.sh` is shellcheck-clean.
- [ ] Workers pin to `CUDA_VISIBLE_DEVICES=$WORKER_INDEX` (assertable via
    a unit test that intercepts the env passed to `subprocess.run`).

## Out of Scope

- Multi-node allocations (we only need 1 node = 4 GPUs for now).
- Auto-resubmission when SLURM time limit is hit (manual: re-sbatch the
    same `QUEUE_ROOT`; remaining `pending/*.json` files are just picked up).
- WandB run-name disambiguation across workers — accepted that runs may
    conflict in the WandB UI; results in `final-results.jsonl` are the source
    of truth and aren't affected.
- Nibi-specific tuning. The script should run on Nibi unchanged if Nibi's
    partition accepts the same SBATCH directives, but we won't validate that
    in this spec.
