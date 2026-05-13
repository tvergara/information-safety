from __future__ import annotations

import os
import time
from pathlib import Path

from scripts.shuffle_pending_mtimes import main, shuffle


def _make_pending(tmp_path: Path, count: int) -> Path:
    queue_root = tmp_path / "queue"
    pending = queue_root / "pending"
    pending.mkdir(parents=True)
    base = time.time() - 10_000
    for i in range(count):
        path = pending / f"{i:04d}.json"
        path.write_text("{}")
        os.utime(path, (base + i, base + i))
    return queue_root


def test_shuffle_preserves_all_files(tmp_path: Path) -> None:
    queue_root = _make_pending(tmp_path, count=10)
    before = sorted(p.name for p in (queue_root / "pending").iterdir())

    shuffle(queue_root=queue_root, seed=42)

    after = sorted(p.name for p in (queue_root / "pending").iterdir())
    assert before == after


def test_shuffle_changes_mtime_ordering(tmp_path: Path) -> None:
    queue_root = _make_pending(tmp_path, count=50)
    pending = queue_root / "pending"
    order_before = [p.name for p in sorted(pending.iterdir(), key=lambda p: p.stat().st_mtime)]

    shuffle(queue_root=queue_root, seed=42)

    order_after = [p.name for p in sorted(pending.iterdir(), key=lambda p: p.stat().st_mtime)]
    assert order_before != order_after


def test_shuffle_is_deterministic_with_seed(tmp_path: Path) -> None:
    queue_a = _make_pending(tmp_path / "a", count=20)
    queue_b = _make_pending(tmp_path / "b", count=20)

    shuffle(queue_root=queue_a, seed=123)
    shuffle(queue_root=queue_b, seed=123)

    order_a = [p.name for p in sorted((queue_a / "pending").iterdir(), key=lambda p: p.stat().st_mtime)]
    order_b = [p.name for p in sorted((queue_b / "pending").iterdir(), key=lambda p: p.stat().st_mtime)]
    assert order_a == order_b


def test_shuffle_assigns_distinct_mtimes(tmp_path: Path) -> None:
    queue_root = _make_pending(tmp_path, count=100)

    shuffle(queue_root=queue_root, seed=7)

    mtimes = [p.stat().st_mtime for p in (queue_root / "pending").iterdir()]
    assert len(set(mtimes)) == len(mtimes)


def test_shuffle_empty_queue_is_noop(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    (queue_root / "pending").mkdir(parents=True)

    shuffle(queue_root=queue_root, seed=0)

    assert list((queue_root / "pending").iterdir()) == []


def test_main_invokes_shuffle(tmp_path: Path) -> None:
    queue_root = _make_pending(tmp_path, count=5)
    pending = queue_root / "pending"
    order_before = [p.name for p in sorted(pending.iterdir(), key=lambda p: p.stat().st_mtime)]

    main(["--queue-root", str(queue_root), "--seed", "1"])

    order_after = [p.name for p in sorted(pending.iterdir(), key=lambda p: p.stat().st_mtime)]
    assert sorted(order_before) == sorted(order_after)
    assert order_before != order_after
