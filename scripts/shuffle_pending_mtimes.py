"""Randomize mtimes of pending job-pool files so workers drain in random order.

The pool worker sorts ``pending/*.json`` by mtime (oldest first). Re-touching
each file with a random mtime once before submitting mixes ordering without
changing the worker or the on-disk schema.
"""

from __future__ import annotations

import argparse
import os
import random
import time
from pathlib import Path


def shuffle(*, queue_root: Path, seed: int | None = None) -> None:
    pending_dir = queue_root / "pending"
    files = sorted(pending_dir.glob("*.json"))
    rng = random.Random(seed)
    now = time.time()
    offsets = list(range(len(files)))
    rng.shuffle(offsets)
    for path, offset in zip(files, offsets):
        new_mtime = now - offset
        os.utime(path, (new_mtime, new_mtime))
    print(f"shuffled mtimes of {len(files)} pending files in {pending_dir}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Randomize mtimes of pending job-pool files"
    )
    parser.add_argument("--queue-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)
    shuffle(queue_root=args.queue_root, seed=args.seed)


if __name__ == "__main__":
    main()
