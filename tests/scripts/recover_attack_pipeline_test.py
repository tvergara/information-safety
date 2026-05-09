"""Tests for recover_attack_pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.recover_attack_pipeline import (
    attack_jobs_by_triple,
    merge_completed_attacks,
    recover,
    reextract_failed_attack_opt,
    repend_recoverable_failures,
)


def _write_payload(directory: Path, name: str, payload: dict) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(json.dumps(payload))
    return path


def _attack_payload(
    *,
    job_id: str,
    attack: str,
    model: str,
    dataset: str,
    shard: int,
    adv_save_dir: str | None = None,
    output_jsonl: str | None = None,
) -> dict:
    command = ["bash", "run_one_attack.sh", "--attack", attack]
    if adv_save_dir is not None:
        command += ["--adv-save-dir", adv_save_dir]
    if output_jsonl is not None:
        command += ["--output-jsonl", output_jsonl]
    return {
        "id": job_id,
        "command": command,
        "spec": {
            "attack": attack,
            "model": model,
            "dataset": dataset,
            "shard": shard,
            "start": shard * 10,
            "end": (shard + 1) * 10,
        },
        "attempts": 0,
    }


def _precomputed_payload(
    *, job_id: str, experiment: str, model: str, dataset: str
) -> dict:
    return {
        "id": job_id,
        "command": ["python", "main.py", "experiment=prompt-attack"],
        "config": {
            "experiment_name": experiment,
            "model_name": model,
            "dataset_name": dataset,
            "max_examples": None,
            "max_epochs": 1,
            "corpus_fraction": None,
            "corpus_subset": None,
        },
        "attempts": 1,
    }


def _write_shard_jsonl(path: Path, n_rows: int, behavior_prefix: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for i in range(n_rows):
            f.write(json.dumps({
                "behavior": f"{behavior_prefix}{i}",
                "test_case": f"tc-{behavior_prefix}{i}",
            }) + "\n")


class TestAttackJobsByTriple:
    def test_groups_payloads_by_attack_model_dataset(self, tmp_path: Path) -> None:
        _write_payload(
            tmp_path / "done", "a.json",
            _attack_payload(job_id="a", attack="gcg", model="m1", dataset="harmbench", shard=0),
        )
        _write_payload(
            tmp_path / "done", "b.json",
            _attack_payload(job_id="b", attack="gcg", model="m1", dataset="harmbench", shard=1),
        )
        _write_payload(
            tmp_path / "pending", "c.json",
            _attack_payload(job_id="c", attack="gcg", model="m2", dataset="harmbench", shard=0),
        )
        groups = attack_jobs_by_triple(tmp_path)
        assert set(groups.keys()) == {("gcg", "m1", "harmbench"), ("gcg", "m2", "harmbench")}
        assert groups[("gcg", "m1", "harmbench")]["done"] == {0, 1}
        assert groups[("gcg", "m2", "harmbench")]["pending"] == {0}

    def test_skips_payloads_without_attack_spec(self, tmp_path: Path) -> None:
        _write_payload(
            tmp_path / "done", "x.json",
            _precomputed_payload(
                job_id="x", experiment="DataStrategy", model="m1", dataset="harmbench"
            ),
        )
        groups = attack_jobs_by_triple(tmp_path)
        assert groups == {}


class TestMergeCompletedAttacks:
    def _setup_done_attack(
        self,
        tmp_path: Path,
        *,
        attack: str,
        model: str,
        dataset: str,
        shard_states: dict[int, str],
    ) -> Path:
        slug = model.replace("/", "_")
        for shard, state in shard_states.items():
            _write_payload(
                tmp_path / state, f"{attack}-{slug}-{dataset}-{shard}.json",
                _attack_payload(
                    job_id=f"{attack}-{slug}-{dataset}-{shard}",
                    attack=attack, model=model, dataset=dataset, shard=shard,
                ),
            )

        attacks_dir = tmp_path / "attacks"
        for shard in shard_states:
            _write_shard_jsonl(
                attacks_dir / f"{attack}-{slug}-{dataset}-shard{shard}.jsonl",
                n_rows=2, behavior_prefix=f"{attack}-{shard}-",
            )
        return attacks_dir

    def test_writes_merged_when_all_shards_done(self, tmp_path: Path) -> None:
        attacks_dir = self._setup_done_attack(
            tmp_path, attack="gcg", model="m1", dataset="harmbench",
            shard_states={0: "done", 1: "done"},
        )
        stats = merge_completed_attacks(tmp_path, attacks_dir)
        merged = attacks_dir / "gcg-m1-harmbench-merged.jsonl"
        assert merged.exists()
        assert stats["merged"] == 1
        with open(merged) as f:
            lines = [json.loads(line) for line in f]
        assert len(lines) == 4

    def test_skips_when_a_shard_is_still_pending(self, tmp_path: Path) -> None:
        attacks_dir = self._setup_done_attack(
            tmp_path, attack="gcg", model="m1", dataset="harmbench",
            shard_states={0: "done", 1: "pending"},
        )
        stats = merge_completed_attacks(tmp_path, attacks_dir)
        assert not (attacks_dir / "gcg-m1-harmbench-merged.jsonl").exists()
        assert stats["merged"] == 0

    def test_skips_when_a_shard_is_still_claimed(self, tmp_path: Path) -> None:
        attacks_dir = self._setup_done_attack(
            tmp_path, attack="gcg", model="m1", dataset="harmbench",
            shard_states={0: "done", 1: "claimed"},
        )
        stats = merge_completed_attacks(tmp_path, attacks_dir)
        assert not (attacks_dir / "gcg-m1-harmbench-merged.jsonl").exists()
        assert stats["merged"] == 0

    def test_skips_when_a_shard_is_failed(self, tmp_path: Path) -> None:
        attacks_dir = self._setup_done_attack(
            tmp_path, attack="gcg", model="m1", dataset="harmbench",
            shard_states={0: "done", 1: "failed"},
        )
        merge_completed_attacks(tmp_path, attacks_dir)
        assert not (attacks_dir / "gcg-m1-harmbench-merged.jsonl").exists()

    def test_skips_when_merged_already_exists(self, tmp_path: Path) -> None:
        attacks_dir = self._setup_done_attack(
            tmp_path, attack="gcg", model="m1", dataset="harmbench",
            shard_states={0: "done", 1: "done"},
        )
        merged = attacks_dir / "gcg-m1-harmbench-merged.jsonl"
        merged.write_text("preexisting\n")
        stats = merge_completed_attacks(tmp_path, attacks_dir)
        assert merged.read_text() == "preexisting\n"
        assert stats["skipped_existing"] == 1

    def test_skips_when_shard_file_missing_on_disk(self, tmp_path: Path) -> None:
        attacks_dir = self._setup_done_attack(
            tmp_path, attack="gcg", model="m1", dataset="harmbench",
            shard_states={0: "done", 1: "done"},
        )
        (attacks_dir / "gcg-m1-harmbench-shard1.jsonl").unlink()
        stats = merge_completed_attacks(tmp_path, attacks_dir)
        assert not (attacks_dir / "gcg-m1-harmbench-merged.jsonl").exists()
        assert stats["merged"] == 0
        assert stats["skipped_missing_shard_files"] == 1

    def test_handles_pair_attack(self, tmp_path: Path) -> None:
        attacks_dir = self._setup_done_attack(
            tmp_path, attack="pair", model="meta-llama/M1", dataset="harmbench",
            shard_states={0: "done"},
        )
        stats = merge_completed_attacks(tmp_path, attacks_dir)
        assert (attacks_dir / "pair-meta-llama_M1-harmbench-merged.jsonl").exists()
        assert stats["merged"] == 1

    def test_handles_slashes_in_model_name(self, tmp_path: Path) -> None:
        attacks_dir = self._setup_done_attack(
            tmp_path, attack="gcg", model="meta-llama/M1", dataset="strongreject",
            shard_states={0: "done", 1: "done"},
        )
        merge_completed_attacks(tmp_path, attacks_dir)
        assert (attacks_dir / "gcg-meta-llama_M1-strongreject-merged.jsonl").exists()


class TestRependRecoverableFailures:
    def test_moves_failed_precomputed_to_pending_when_merged_exists(
        self, tmp_path: Path
    ) -> None:
        attacks_dir = tmp_path / "attacks"
        attacks_dir.mkdir()
        (attacks_dir / "gcg-meta-llama_M1-harmbench-merged.jsonl").write_text("{}\n")

        _write_payload(
            tmp_path / "failed", "j1.json",
            _precomputed_payload(
                job_id="j1", experiment="PrecomputedGCGStrategy",
                model="meta-llama/M1", dataset="harmbench",
            ),
        )
        stats = repend_recoverable_failures(tmp_path, attacks_dir)
        assert (tmp_path / "pending" / "j1.json").exists()
        assert not (tmp_path / "failed" / "j1.json").exists()
        assert stats["repended"] == 1

    def test_leaves_failed_when_merged_missing(self, tmp_path: Path) -> None:
        attacks_dir = tmp_path / "attacks"
        attacks_dir.mkdir()

        _write_payload(
            tmp_path / "failed", "j2.json",
            _precomputed_payload(
                job_id="j2", experiment="PrecomputedGCGStrategy",
                model="meta-llama/M1", dataset="harmbench",
            ),
        )
        stats = repend_recoverable_failures(tmp_path, attacks_dir)
        assert (tmp_path / "failed" / "j2.json").exists()
        assert not (tmp_path / "pending" / "j2.json").exists()
        assert stats["repended"] == 0
        assert stats["skipped_no_suffix"] == 1

    def test_ignores_non_precomputed_failures(self, tmp_path: Path) -> None:
        attacks_dir = tmp_path / "attacks"
        attacks_dir.mkdir()
        (attacks_dir / "gcg-meta-llama_M1-harmbench-merged.jsonl").write_text("{}\n")

        _write_payload(
            tmp_path / "failed", "j3.json",
            _precomputed_payload(
                job_id="j3", experiment="DataStrategy",
                model="meta-llama/M1", dataset="harmbench",
            ),
        )
        stats = repend_recoverable_failures(tmp_path, attacks_dir)
        assert (tmp_path / "failed" / "j3.json").exists()
        assert stats["repended"] == 0

    def test_preserves_attempts(self, tmp_path: Path) -> None:
        attacks_dir = tmp_path / "attacks"
        attacks_dir.mkdir()
        (attacks_dir / "autodan-m1-harmbench-merged.jsonl").write_text("{}\n")

        payload = _precomputed_payload(
            job_id="j4", experiment="PrecomputedAutoDANStrategy",
            model="m1", dataset="harmbench",
        )
        payload["attempts"] = 5
        _write_payload(tmp_path / "failed", "j4.json", payload)

        repend_recoverable_failures(tmp_path, attacks_dir)
        repended = json.loads((tmp_path / "pending" / "j4.json").read_text())
        assert repended["attempts"] == 5

    def test_handles_pair_strategy(self, tmp_path: Path) -> None:
        attacks_dir = tmp_path / "attacks"
        attacks_dir.mkdir()
        (attacks_dir / "pair-m1-harmbench-merged.jsonl").write_text("{}\n")

        _write_payload(
            tmp_path / "failed", "j5.json",
            _precomputed_payload(
                job_id="j5", experiment="PrecomputedPAIRStrategy",
                model="m1", dataset="harmbench",
            ),
        )
        stats = repend_recoverable_failures(tmp_path, attacks_dir)
        assert (tmp_path / "pending" / "j5.json").exists()
        assert stats["repended"] == 1


class TestReextractFailedAttackOpt:
    def _setup_failed_with_run_json(
        self, tmp_path: Path, *, job_id: str, with_run_json: bool
    ) -> tuple[Path, Path]:
        adv_save_dir = tmp_path / "adv-outputs" / job_id
        output_jsonl = tmp_path / "attacks" / f"{job_id}.jsonl"
        if with_run_json:
            (adv_save_dir / "0").mkdir(parents=True, exist_ok=True)
            (adv_save_dir / "0" / "run.json").write_text("{}")
        else:
            adv_save_dir.mkdir(parents=True, exist_ok=True)
        (tmp_path / "done").mkdir(parents=True, exist_ok=True)
        _write_payload(
            tmp_path / "failed", f"{job_id}.json",
            _attack_payload(
                job_id=job_id, attack="pair", model="m1",
                dataset="harmbench", shard=0,
                adv_save_dir=str(adv_save_dir),
                output_jsonl=str(output_jsonl),
            ),
        )
        return adv_save_dir, output_jsonl

    def test_moves_to_done_when_extraction_succeeds(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _, output_jsonl = self._setup_failed_with_run_json(
            tmp_path, job_id="j1", with_run_json=True
        )

        def fake_extract(run_dir: str, output: str) -> None:
            Path(output).parent.mkdir(parents=True, exist_ok=True)
            Path(output).write_text("{}\n")

        monkeypatch.setattr(
            "scripts.recover_attack_pipeline.convert_run_dir", fake_extract
        )
        stats = reextract_failed_attack_opt(tmp_path)
        assert (tmp_path / "done" / "j1.json").exists()
        assert not (tmp_path / "failed" / "j1.json").exists()
        assert output_jsonl.exists()
        assert stats["reextracted"] == 1

    def test_skips_when_adv_save_dir_missing(self, tmp_path: Path) -> None:
        _write_payload(
            tmp_path / "failed", "j2.json",
            _attack_payload(
                job_id="j2", attack="pair", model="m1",
                dataset="harmbench", shard=0,
                adv_save_dir=str(tmp_path / "does-not-exist"),
                output_jsonl=str(tmp_path / "out.jsonl"),
            ),
        )
        stats = reextract_failed_attack_opt(tmp_path)
        assert (tmp_path / "failed" / "j2.json").exists()
        assert stats["skipped_no_run_dir"] == 1
        assert stats["reextracted"] == 0

    def test_skips_when_no_run_json_in_dir(self, tmp_path: Path) -> None:
        self._setup_failed_with_run_json(tmp_path, job_id="j3", with_run_json=False)
        stats = reextract_failed_attack_opt(tmp_path)
        assert (tmp_path / "failed" / "j3.json").exists()
        assert stats["skipped_no_run_dir"] == 1

    def test_leaves_in_failed_when_extraction_still_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._setup_failed_with_run_json(tmp_path, job_id="j4", with_run_json=True)

        def broken_extract(_r: str, _o: str) -> None:
            raise KeyError("num_params")

        monkeypatch.setattr(
            "scripts.recover_attack_pipeline.convert_run_dir", broken_extract
        )
        stats = reextract_failed_attack_opt(tmp_path)
        assert (tmp_path / "failed" / "j4.json").exists()
        assert not (tmp_path / "done" / "j4.json").exists()
        assert stats["extraction_still_fails"] == 1

    def test_ignores_strategy_failures(self, tmp_path: Path) -> None:
        _write_payload(
            tmp_path / "failed", "j5.json",
            _precomputed_payload(
                job_id="j5", experiment="PrecomputedGCGStrategy",
                model="m1", dataset="harmbench",
            ),
        )
        stats = reextract_failed_attack_opt(tmp_path)
        assert (tmp_path / "failed" / "j5.json").exists()
        assert stats["reextracted"] == 0

    def test_handles_missing_failed_dir(self, tmp_path: Path) -> None:
        stats = reextract_failed_attack_opt(tmp_path)
        assert stats["reextracted"] == 0


class TestRecover:
    def test_runs_both_phases(self, tmp_path: Path) -> None:
        attacks_dir = tmp_path / "attacks"
        attacks_dir.mkdir()
        slug = "m1"
        for shard in (0, 1):
            _write_shard_jsonl(
                attacks_dir / f"gcg-{slug}-harmbench-shard{shard}.jsonl",
                n_rows=2, behavior_prefix=f"s{shard}-",
            )
            _write_payload(
                tmp_path / "done", f"shard{shard}.json",
                _attack_payload(
                    job_id=f"shard{shard}", attack="gcg", model=slug,
                    dataset="harmbench", shard=shard,
                ),
            )

        _write_payload(
            tmp_path / "failed", "precomp.json",
            _precomputed_payload(
                job_id="precomp", experiment="PrecomputedGCGStrategy",
                model=slug, dataset="harmbench",
            ),
        )

        stats = recover(tmp_path, attacks_dir)

        assert (attacks_dir / "gcg-m1-harmbench-merged.jsonl").exists()
        assert (tmp_path / "pending" / "precomp.json").exists()
        assert stats["merged"] == 1
        assert stats["repended"] == 1


class TestMissingDirsTolerance:
    def test_attack_jobs_by_triple_handles_missing_subdirs(self, tmp_path: Path) -> None:
        groups = attack_jobs_by_triple(tmp_path)
        assert groups == {}

    def test_repend_handles_missing_failed_dir(self, tmp_path: Path) -> None:
        attacks_dir = tmp_path / "attacks"
        attacks_dir.mkdir()
        stats = repend_recoverable_failures(tmp_path, attacks_dir)
        assert stats["repended"] == 0

    def test_merge_handles_no_attack_jobs(self, tmp_path: Path) -> None:
        attacks_dir = tmp_path / "attacks"
        attacks_dir.mkdir()
        stats = merge_completed_attacks(tmp_path, attacks_dir)
        assert stats["merged"] == 0


class TestShardGapDetection:
    def test_does_not_merge_with_gap_in_shards(self, tmp_path: Path) -> None:
        attacks_dir = tmp_path / "attacks"
        attacks_dir.mkdir()
        for shard in (0, 2):
            _write_payload(
                tmp_path / "done", f"s{shard}.json",
                _attack_payload(
                    job_id=f"s{shard}", attack="gcg", model="m1",
                    dataset="harmbench", shard=shard,
                ),
            )
            _write_shard_jsonl(
                attacks_dir / f"gcg-m1-harmbench-shard{shard}.jsonl",
                n_rows=1, behavior_prefix=f"g{shard}-",
            )
        stats = merge_completed_attacks(tmp_path, attacks_dir)
        assert stats["merged"] == 0
        assert not (attacks_dir / "gcg-m1-harmbench-merged.jsonl").exists()
