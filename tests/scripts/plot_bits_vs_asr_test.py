"""Tests for bits-vs-ASR scatter plotting."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.plot_bits_vs_asr import build_points, load_rows, plot_bits_vs_asr


class TestBuildPoints:
    def test_extracts_points_with_method_bits_asr(self) -> None:
        rows = [
            {"attack_method": "zero_shot", "bits": 12, "asr": 0.1},
            {"experiment_name": "gcg", "bits": 80, "asr": 0.6},
        ]
        points = build_points(rows)
        assert points == [("zero_shot", 12.0, 0.1), ("gcg", 80.0, 0.6)]

    def test_skips_incomplete_rows(self) -> None:
        rows = [{"attack_method": "a", "bits": 1}, {"attack_method": "b", "asr": 0.3}]
        assert build_points(rows) == []

    def test_falls_back_to_prompt_text_bits(self) -> None:
        rows = [{"attack_method": "few_shot", "prompt": "abcd", "asr": 0.25}]
        points = build_points(rows)
        assert points == [("few_shot", 32.0, 0.25)]


class TestPlotBitsVsAsr:
    def test_writes_scatter_png(self, tmp_path: Path) -> None:
        out_file = tmp_path / "bits_vs_asr.png"
        plot_bits_vs_asr([("zero_shot", 10.0, 0.2), ("gcg", 100.0, 0.8)], str(out_file))
        assert out_file.exists()

    def test_raises_for_empty_points(self, tmp_path: Path) -> None:
        out_file = tmp_path / "bits_vs_asr.png"
        with pytest.raises(ValueError, match="no valid points"):
            plot_bits_vs_asr([], str(out_file))


class TestLoadRows:
    def test_reads_jsonl(self, tmp_path: Path) -> None:
        file_path = tmp_path / "results.jsonl"
        with open(file_path, "w") as f:
            f.write(json.dumps({"attack_method": "zero_shot", "bits": 16, "asr": 0.2}) + "\n")

        rows = load_rows(str(file_path))
        assert len(rows) == 1
