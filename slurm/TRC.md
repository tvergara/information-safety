# Using TRC for information-safety

`trc` is a fundamentally different beast from tamia/nibi. This note captures what we need to know to port our pipeline there. The authoritative `eai` reference lives on the trc Mac itself (see "Reading the skill" below) — this file is the project-specific layer on top.

## What "trc" actually is

- **The Mac at `24.225.203.65` (`siva.reddy@...`, in `~/.ssh/config` as `trc`) is just a submission host.** The compute is on ServiceNow's Element AI Toolkit cluster — specifically the **yul201 superpod** (H100s, Montréal).
- The submission CLI is `eai` (the binary at `/usr/local/bin/eai` on trc). Two profiles are configured:
    - `yul101` → `https://console.elementai.com` (old cluster, no H100s)
    - `yul201` → `https://toolkit-sp.yul201.service-now.com` (H100 superpod — what we want)
- Always prefix commands with `EAI_PROFILE=yul201` once we have our own login.
- **VPN required.** If a command fails with `dial tcp: lookup toolkit-sp.yul201.service-now.com: no such host`, the user is off the yul201 VPN. Setup is manual via the ServiceNow SharePoint page (linked in `docs/yul201_setup.md` on trc).

## Auth

- `trc` Mac is already logged in as `siva.reddy` for both profiles. Our SSH access uses Tomás's `id_rsa` (installed via `ssh-copy-id` on 2026-05-13).
- For our own account, the login flow is **interactive** (`eai login` opens a URL, prompts for a verification code) — Claude cannot do it. The user runs it once on whatever host we plan to submit from.
- `EAI_TOKEN` env var, if set, overrides everything and is a common source of 401s — unset it before re-logging in.

## Account scope (TBD — ask before submitting)

- Siva's running jobs are under `snow.research.mmteb`. That's his project, not ours.
- Before we submit anything, confirm with the user which account we use for information-safety. Candidates:
    - A new sub-account under `snow.research.*`
    - Piggyback on `snow.research.mmteb` (only if explicitly told to)
    - User's personal account (`snow.<user>`)

## The mental model — why this isn't sbatch

| Concept              | tamia/nibi (SLURM)                | trc (eai/Toolkit)                                                                                                                                             |
| -------------------- | --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Job                  | Shell script via `sbatch`         | Docker container via `eai job new`                                                                                                                            |
| Code                 | Mounted from `$HOME`              | Baked into the image **or** mounted from `eai data`                                                                                                           |
| Env                  | `module load` + `uv venv`         | Provided by the image                                                                                                                                         |
| Storage between jobs | `$SCRATCH` shared filesystem      | None — container is ephemeral; persist via `eai data push`                                                                                                    |
| SSH into a node      | `ssh node-XXX` after `salloc`     | `eai job exec <id> -- /bin/bash`                                                                                                                              |
| Status               | `squeue -u $USER`                 | `eai job ls`                                                                                                                                                  |
| Logs                 | tail `slurm-NNN.out` on shared FS | `eai job log <id> [--follow] [--tail N]`                                                                                                                      |
| Time limit           | `--time=HH:MM:SS`                 | `--max-run-time <seconds>`                                                                                                                                    |
| Job array            | `--array=0-99`                    | None native — write a queue-worker, or `--replicas` for distributed                                                                                           |
| Priority             | partition / QoS                   | `--interactive` (1/user, highest) → `--non-preemptable` (48h max) → `--preemptable` (extendable, evictable) → `--restartable`, with `--bid` for fine ordering |

## Job lifecycle

- States: `queued` / `queuing` → `running` → `succeeded` / `failed` / `cancelled` / `interrupted`. UUIDs everywhere.
- Each job has a per-job `accessUrl` (e.g. `https://<uuid>.job.toolkit-sp.yul201.service-now.com`) for jupyter/UI. Port-forward with `eai job port-forward <id> 2222`.
- Names must match `^[a-z0-9_]+$` and be unique — convention is `<name>_<YYYYMMDD_HHMMSS_utc>` (use `date -u +"%Y%m%d_%H%M%S_utc"`).

## Cheatsheet

All commands run via `ssh trc '<cmd>'` from Mila (we're using Siva's logged-in profile). Add `EAI_PROFILE=yul201` explicitly if we ever switch profiles.

```bash
# Submit (preferred: YAML spec; `submit` is an alias for `new`)
ssh trc 'cd ~/safety && eai job submit -f ./eai_code.yaml --enforce-name'

# --enforce-name is REQUIRED when reusing a YAML — old CANCELLED jobs squat
# the literal name. The flag auto-suffixes (e.g. safety_code_1 -> safety_code_11778...).

# Status
ssh trc 'eai job ls --limit 20'                              # last 24h of mine
ssh trc 'eai job ls --me --state all --limit 50'             # everything
ssh trc 'eai job get <id> --fields state --no-header'        # JUST the state, one line
# NOTE: use `--fields` (plural) with default table format. `--field state --format text`
# dumps the whole struct and is useless for polling.
ssh trc 'eai job info <id> [--job-spec]'                     # full YAML / reusable spec

# Logs
ssh trc 'eai job log <id>'                                   # snapshot
ssh trc 'eai job log --follow <id>'                          # tail -f (blocks until terminal)
ssh trc 'eai job log --tail 200 <id>'                        # last N lines

# Exec into a running container — works NON-INTERACTIVELY over ssh
# Wrap the inner command in `bash -c "..."` so we don't need a TTY.
ssh trc 'eai job exec <id> -- bash -c "ls /work && nproc"'

# Kill / retry
ssh trc 'eai job kill <id>'
ssh trc 'eai job retry <id>'
```

## Data — the durable layer

Containers are throwaway. **All durable state must round-trip through `eai data` resources.**

- Naming: `Org.Account[.SubAccount...].DataName`, all lowercase + underscores. E.g. `snow.research.<account>.information_safety_src`.
- Lifecycle: `eai data new <fullName> [./local]` creates (optionally with initial upload) → `eai data push <fullName>[@version] ./local[:remote/path]` for partial updates → `eai data pull <fullName>[@version] [./dest]` to download.
- Mount inside a job: `--data <fullName>[/subpath]:/container/path[:ro|:rw]`. **No nested mounts** (`/work` and `/work/sub` can't both be mount points; siblings are fine).
- Branches/versions: `eai data branch add <fullName>@<v> <branchname>` pins a version. Use branches (`@v1`) in job specs for reproducibility — never `@latest` for real runs.
- Inspection: `eai data content ls/tree/rm` (rm is irreversible, glob-supported).

Per the CLAUDE.md data policy: **never `cat` or `Read` any pulled data file contents** — only inspect metadata via `eai data size`, `eai data content ls`, etc.

## Docker images

- Stable vs volatile registry: `eai docker get-registry [--volatile]`. Stable = permanent. Volatile auto-deletes after 24h unless the tag has `-7days` / `-2months` suffix (max 90 days).
- Tagging: `docker tag local:tag <registry>/snow.<account>/<image>:<tag>` then `docker push`. Auth is auto-wired by `eai login`'s docker credential helper.
- For job submission: **always use a specific version tag**, never `:latest`.

## Reading the skill (canonical eai reference)

The Claude skill is installed on **trc only**, at:

```
/Users/siva.reddy/.claude/plugins/cache/agent-plugins-internal/toolkit/1.0.6/
├── skills/eai/SKILL.md          # top-level routing
├── skills/eai/docs/             # 20 topic-specific docs
│   ├── job.md                   # submission flow, monitoring, exec, port-forward
│   ├── data.md                  # push/pull/branch/content
│   ├── docker.md                # registries, tagging
│   ├── authentication_login.md  # 401/403 diagnosis
│   ├── yul201_setup.md          # VPN + LDAP setup pointers
│   └── ...                      # account, team, role, service, mfa, etc.
├── scripts/                     # helper scripts on PATH inside a trc session
│   ├── eai_plugin_watch_job_until_terminal_state.py
│   ├── eai_plugin_kill_job_and_wait_for_terminal_state.py
│   └── eai_client.py
```

To consult it from here:

```bash
ssh trc 'cat /Users/siva.reddy/.claude/plugins/cache/agent-plugins-internal/toolkit/1.0.6/skills/eai/docs/<topic>.md'
```

Or `eai <command> --help` from inside `ssh trc` — every subcommand supports it.

## Validated flow: the "code access" CPU job

Siva set up a sleep-loop CPU job on `~/safety/eai_code.yaml` on trc. Submitting it gives us a long-running container with `/work` mounted, into which we `eai job exec` to do anything that needs real cluster filesystem access (clone repos, build envs, push results, debug, etc.).

The YAML (don't modify Siva's copy — copy and adapt if we want our own):

```text
workdir: /home/toolkit
name: safety_code_1
data: [snow.research.mmteb.safety:/work:rw]
image: registry.toolkit-sp.yul201.service-now.com/snow.research.mmteb/mteb-lite:v1
resources: {cpu: 4, mem: 32}
interactive: false
preemptable: true
command: [/tk/bin/start.sh, bash, -c, while true; do sleep 3600; done]
```

Submission and use:

```bash
ssh trc 'cd ~/safety && eai job submit -f ./eai_code.yaml --enforce-name'
# -> takes ~10s to leave QUEUING, lands on a DGX node (Xeon Platinum 8480C, 2TiB host RAM)
# -> get the new UUID from the output

ssh trc 'eai job exec <id> -- bash -c "<any command>"'
```

### What the container looks like inside

| Aspect      | Value                                                                         |
| ----------- | ----------------------------------------------------------------------------- |
| Image       | `registry.toolkit-sp.yul201.service-now.com/snow.research.mmteb/mteb-lite:v1` |
| OS          | Ubuntu 20.04 LTS (Focal Fossa)                                                |
| User        | `toolkit` (HOME = `/home/toolkit`, **ephemeral**)                             |
| Python      | `/opt/conda/bin/python3` (3.10.13)                                            |
| Other tools | `git` at `/usr/bin/git`. **No `uv` on PATH** — but `/work/envs/uv/` exists    |
| Persistence | **Only `/work`** (mounted from `snow.research.mmteb.safety`)                  |
| Hostname    | The job UUID                                                                  |

### Lab conventions inside `/work` (from `/work/CLAUDE.md`)

- **Per-project venvs are the norm.** The lab's CLAUDE.md says always activate `/work/forecast_generalization/.venv` before `python` — that's *their* default; for us it means we should make our own at `/work/information-safety/.venv`.
- `/work/envs/` holds shared venvs (`safenv`, `scalenv`, `taenv`, `inferenv`, `rl`, `uv`, …). Don't depend on these — they're owned by other team members and can change.
- `/work/.claude/` has a lab-wide Claude config. Don't touch.
- Project layout pattern: `/work/<project_name>/` at the top level. Existing siblings include `attack-scaling`, `forecast_generalization`, `OLMo`, `rl-jailbreak`, `tiered_alignment`, `scaling_analysis` — so we'd add `/work/information-safety/`.

### Cheap polling pattern

```bash
# Get state as one bare word — works for scripting
ssh trc 'eai job get <id> --fields state --no-header'
# Returns: RUNNING (or QUEUING / SUCCEEDED / FAILED / CANCELLED / INTERRUPTED)
```

## Bootstrap: getting the repo onto /work

Done once on 2026-05-13. Re-run only if `/work/information-safety/` is gone.

**Layout decisions:**

- Repo path: `/work/information-safety/` (top-level, matches lab convention).
- Auth: fine-grained GitHub PAT (read-only, scoped to `tvergara/information-safety` + `tvergara/AdversariaLLM`), stored at `/work/.git-credentials` chmod 600. Git's `store` credential helper at the repo and submodule level picks it up for future `pull`/`fetch`.
- Venv: `/work/information-safety/.venv` via `uv sync`. The lab's `uv` binary is at `/work/envs/uv/uv` — **not** on PATH by default.
- Image: reusing `mteb-lite:v1` (the "code-access" CPU job's image); revisit if startup cost becomes painful.

**One-shot bootstrap** (piped over `eai job exec`'s stdin; the outer shell expands only `${PAT}` and `${JOB_ID}`):

```bash
PAT='<github_pat_...>'           # fine-grained, contents:read on both repos
JOB_ID='<safety_code_* UUID>'    # an alive CPU-access job

ssh trc "eai job exec ${JOB_ID} -- bash -s" <<EOF
set -e

umask 077
printf 'https://tvergara:%s@github.com\n' '${PAT}' > /work/.git-credentials
chmod 600 /work/.git-credentials

git -c credential.helper='store --file=/work/.git-credentials' \
    clone https://github.com/tvergara/information-safety.git /work/information-safety
git -C /work/information-safety config credential.helper 'store --file=/work/.git-credentials'

git -C /work/information-safety \
    -c credential.helper='store --file=/work/.git-credentials' \
    submodule update --init --recursive
git -C /work/information-safety/third_party/adversariallm \
    config credential.helper 'store --file=/work/.git-credentials'

cd /work/information-safety
/work/envs/uv/uv sync
EOF
```

**Why the credential helper is set per-repo (not `--global`):** `$HOME = /home/toolkit` is ephemeral. Anything written to `~/.gitconfig` disappears between containers. Setting `credential.helper` in `.git/config` means it lives on `/work` and survives.

**To pull updates later** (from anywhere with `ssh trc`):

```bash
ssh trc 'eai job exec <job_id> -- bash -c "cd /work/information-safety && git pull && git submodule update --init --recursive"'
```

**PAT hygiene:** rotate the token when bootstrap-time work is done. Replace `/work/.git-credentials` with `printf 'https://tvergara:<NEW_PAT>@github.com\n' > /work/.git-credentials && chmod 600 /work/.git-credentials`. Revoke the old PAT on GitHub.

## Pipeline gaps we need to close

To plug trc into our existing `python information_safety/main.py` → `final-results.jsonl` flow, we still need to decide:

1. **Where `eai` runs from.** Easiest is from trc itself (already logged in). Alternative: install on Mila and `eai login` as Tomás.
2. **The Docker image.** Either reuse `snow.shared/interactive-toolkit` (the yul201 default for interactive jobs) and `pip install` at job start, or build our own with `uv sync`'d env baked in. Latter is reproducible but heavier.
3. **Source layout.** Mount an `information_safety_src` data resource at `/work/information-safety`, then run the same Hydra entry point. Siva's `attack-scaling` jobs use this pattern — worth studying as a reference.
4. **Results round-trip.** No shared FS, so the job needs to `eai data push` results back at the end (or stream during run). Naïvely, `final-results.jsonl` lives in a `infosafety_results` data resource; the job appends + pushes on exit.
5. **Job-pool equivalent.** Tamia/nibi force 4×H100 allocations and we run our queue-worker pool inside. yul201 may not have the same allocation shape — once we know the resource model, decide whether to keep the `scripts/job_pool_worker.py` pattern or rely on eai's native job queue.
6. **Pre-submit guard.** `slurm/sync-and-merge-from.sh`'s eval-sweep race check has no analog yet — we'll need a trc-side equivalent (or just block locally while a trc merge is in flight).

Until these are decided, **do not submit information-safety jobs to trc**. Use it only for read-only exploration (`eai job ls`, etc.) so we don't pollute Siva's account.

## Open questions for the user

- Which `eai` account/team do we use? (`snow.research.???`)
- Is the VPN already set up for Tomás's machine, or does that need provisioning?
- Do we want to install `eai` on Mila and have our own login, or operate exclusively via `ssh trc`?
- Is there a separate quota/budget here that we should respect (interactive job slot, GPU-hours, etc.)?
