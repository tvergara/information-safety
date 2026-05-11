from __future__ import annotations

import json
from pathlib import Path

from scripts.purge_strongreject_attack_artifacts import purge


def _write_spec(
    path: Path, *, attack: str, dataset: str, shard: int = 0
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "id": f"{attack}-{dataset}-{shard}",
        "command": ["bash", "noop.sh"],
        "spec": {
            "attack": attack,
            "model": "m1",
            "dataset": dataset,
            "shard": shard,
            "start": 0,
            "end": 1,
        },
        "attempts": 0,
    }
    path.write_text(json.dumps(payload))


def _setup_fake_world(tmp_path: Path) -> tuple[Path, Path, Path]:
    queue_root = tmp_path / "queue"
    attacks_dir = tmp_path / "attacks"
    adv_outputs = tmp_path / "adversariallm-outputs"

    for sub in ("pending", "claimed", "done", "failed"):
        (queue_root / sub).mkdir(parents=True, exist_ok=True)

    _write_spec(
        queue_root / "pending" / "a.json",
        attack="gcg", dataset="strongreject", shard=0,
    )
    _write_spec(
        queue_root / "claimed" / "b.pool.0.json",
        attack="autodan", dataset="strongreject", shard=1,
    )
    _write_spec(
        queue_root / "done" / "c.json",
        attack="gcg", dataset="strongreject", shard=2,
    )
    _write_spec(
        queue_root / "failed" / "d.json",
        attack="autodan", dataset="strongreject", shard=3,
    )

    _write_spec(
        queue_root / "pending" / "e.json",
        attack="gcg", dataset="harmbench", shard=0,
    )
    _write_spec(
        queue_root / "pending" / "f.json",
        attack="autodan", dataset="harmbench", shard=0,
    )
    _write_spec(
        queue_root / "pending" / "g.json",
        attack="pair", dataset="strongreject", shard=0,
    )
    _write_spec(
        queue_root / "done" / "h.json",
        attack="pair", dataset="harmbench", shard=0,
    )

    (queue_root / "done" / "non_attack.json").write_text(json.dumps({
        "id": "non_attack",
        "command": ["python", "main.py"],
        "config": {"experiment_name": "DataStrategy"},
        "attempts": 0,
    }))

    attacks_dir.mkdir(parents=True, exist_ok=True)
    sr_files = [
        "gcg-m1-strongreject-shard0.jsonl",
        "autodan-m1-strongreject-shard1.jsonl",
        "gcg-m1-strongreject-merged.jsonl",
        "autodan-m1-strongreject-merged.jsonl",
    ]
    keep_files = [
        "gcg-m1-harmbench-shard0.jsonl",
        "autodan-m1-harmbench-shard0.jsonl",
        "pair-m1-strongreject-shard0.jsonl",
        "pair-m1-strongreject-merged.jsonl",
        "pair-m1-harmbench-shard0.jsonl",
    ]
    for name in sr_files + keep_files:
        (attacks_dir / name).write_text("")

    adv_outputs.mkdir(parents=True, exist_ok=True)
    sr_dirs = [
        "gcg-m1-strongreject-shard0",
        "autodan-m1-strongreject-shard1",
    ]
    keep_dirs = [
        "gcg-m1-harmbench-shard0",
        "autodan-m1-harmbench-shard0",
        "pair-m1-strongreject-shard0",
        "pair-m1-harmbench-shard0",
    ]
    for name in sr_dirs + keep_dirs:
        (adv_outputs / name).mkdir(parents=True, exist_ok=True)
        (adv_outputs / name / "marker.txt").write_text("x")

    return queue_root, attacks_dir, adv_outputs


class TestPurge:
    def test_removes_only_strongreject_gcg_autodan(self, tmp_path: Path) -> None:
        queue_root, attacks_dir, adv_outputs = _setup_fake_world(tmp_path)

        result = purge(
            queue_root=queue_root,
            attacks_dir=attacks_dir,
            adv_outputs_dir=adv_outputs,
        )

        remaining_specs = []
        for sub in ("pending", "claimed", "done", "failed"):
            remaining_specs.extend(sorted((queue_root / sub).glob("*.json")))
        remaining_names = {p.name for p in remaining_specs}
        assert "non_attack.json" in remaining_names
        for path in remaining_specs:
            spec = json.loads(path.read_text()).get("spec")
            if spec is None:
                continue
            forbidden = (
                spec["dataset"] == "strongreject"
                and spec["attack"] in {"gcg", "autodan"}
            )
            assert not forbidden, spec

        attack_files = {p.name for p in attacks_dir.glob("*")}
        assert "gcg-m1-strongreject-shard0.jsonl" not in attack_files
        assert "autodan-m1-strongreject-shard1.jsonl" not in attack_files
        assert "gcg-m1-strongreject-merged.jsonl" not in attack_files
        assert "autodan-m1-strongreject-merged.jsonl" not in attack_files
        assert "gcg-m1-harmbench-shard0.jsonl" in attack_files
        assert "autodan-m1-harmbench-shard0.jsonl" in attack_files
        assert "pair-m1-strongreject-shard0.jsonl" in attack_files
        assert "pair-m1-strongreject-merged.jsonl" in attack_files
        assert "pair-m1-harmbench-shard0.jsonl" in attack_files

        out_dirs = {p.name for p in adv_outputs.iterdir()}
        assert "gcg-m1-strongreject-shard0" not in out_dirs
        assert "autodan-m1-strongreject-shard1" not in out_dirs
        assert "gcg-m1-harmbench-shard0" in out_dirs
        assert "autodan-m1-harmbench-shard0" in out_dirs
        assert "pair-m1-strongreject-shard0" in out_dirs
        assert "pair-m1-harmbench-shard0" in out_dirs

        assert result["specs_removed"] == 4
        assert result["jsonls_removed"] == 4
        assert result["dirs_removed"] == 2

    def test_idempotent_second_run(self, tmp_path: Path) -> None:
        queue_root, attacks_dir, adv_outputs = _setup_fake_world(tmp_path)

        purge(
            queue_root=queue_root,
            attacks_dir=attacks_dir,
            adv_outputs_dir=adv_outputs,
        )
        result = purge(
            queue_root=queue_root,
            attacks_dir=attacks_dir,
            adv_outputs_dir=adv_outputs,
        )

        assert result["specs_removed"] == 0
        assert result["jsonls_removed"] == 0
        assert result["dirs_removed"] == 0
