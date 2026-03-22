"""Tests for plotting attack result summaries."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.plot_attack_results import aggregate_asr, load_rows, plot_asr


class TestLoadRows:
    def test_loads_jsonl_rows(self, tmp_path: Path) -> None:
        file_path = tmp_path / "results.jsonl"
        with open(file_path, "w") as f:
            f.write(json.dumps({"attack_method": "zero_shot", "asr": 0.1}) + "\n")
            f.write(json.dumps({"attack_method": "few_shot", "asr": 0.3}) + "\n")

        rows = load_rows(str(file_path))
        assert len(rows) == 2
        assert rows[0]["attack_method"] == "zero_shot"


class TestAggregateAsr:
    def test_averages_per_method(self) -> None:
        rows = [
            {"attack_method": "zero_shot", "asr": 0.1},
            {"attack_method": "zero_shot", "asr": 0.3},
            {"attack_method": "few_shot", "asr": 0.5},
        ]
        summary = aggregate_asr(rows)
        assert summary["zero_shot"] == pytest.approx(0.2)
        assert summary["few_shot"] == pytest.approx(0.5)

    def test_supports_experiment_name_fallback(self) -> None:
        rows = [{"experiment_name": "gcg", "asr": 0.6}]
        summary = aggregate_asr(rows)
        assert summary["gcg"] == pytest.approx(0.6)


class TestPlotAsr:
    def test_writes_output_png(self, tmp_path: Path) -> None:
        output_file = tmp_path / "plot.png"
        plot_asr({"zero_shot": 0.1, "few_shot": 0.3}, str(output_file))
        assert output_file.exists()

    def test_raises_on_empty_summary(self, tmp_path: Path) -> None:
        output_file = tmp_path / "plot.png"
        with pytest.raises(ValueError, match="no ASR entries"):
            plot_asr({}, str(output_file))
