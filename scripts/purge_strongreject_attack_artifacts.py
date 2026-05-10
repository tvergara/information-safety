"""Remove strongreject GCG/AutoDAN queue specs, shard JSONLs, and adv-output dirs."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

_TARGET_ATTACKS = {"gcg", "autodan"}
_TARGET_DATASET = "strongreject"
_QUEUE_SUBDIRS = ("pending", "claimed", "done", "failed")


def purge(
    *, queue_root: Path, attacks_dir: Path, adv_outputs_dir: Path
) -> dict[str, int]:
    specs_removed = 0
    for sub in _QUEUE_SUBDIRS:
        for spec_path in (queue_root / sub).glob("*.json"):
            spec = json.loads(spec_path.read_text()).get("spec")
            if spec is None:
                continue
            if (
                spec["dataset"] == _TARGET_DATASET
                and spec["attack"] in _TARGET_ATTACKS
            ):
                spec_path.unlink()
                specs_removed += 1

    jsonls_removed = 0
    for attack in _TARGET_ATTACKS:
        for path in attacks_dir.glob(f"{attack}-*-{_TARGET_DATASET}-*.jsonl"):
            path.unlink()
            jsonls_removed += 1

    dirs_removed = 0
    for attack in _TARGET_ATTACKS:
        for path in adv_outputs_dir.glob(f"{attack}-*-{_TARGET_DATASET}-*"):
            if path.is_dir():
                shutil.rmtree(path)
                dirs_removed += 1

    return {
        "specs_removed": specs_removed,
        "jsonls_removed": jsonls_removed,
        "dirs_removed": dirs_removed,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue-root", type=Path, required=True)
    parser.add_argument("--attacks-dir", type=Path, required=True)
    parser.add_argument("--adv-outputs-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    counts = purge(
        queue_root=args.queue_root,
        attacks_dir=args.attacks_dir,
        adv_outputs_dir=args.adv_outputs_dir,
    )
    print(
        f"Removed {counts['specs_removed']} queue specs, "
        f"{counts['jsonls_removed']} shard jsonls, "
        f"{counts['dirs_removed']} adv-output dirs."
    )


if __name__ == "__main__":
    main()
