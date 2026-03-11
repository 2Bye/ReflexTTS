"""Unit tests for benchmark scripts."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.run_benchmarks import (
    BenchmarkResult,
    compute_summary,
    load_texts,
    print_report,
)


class TestLoadTexts:
    """Tests for loading benchmark texts."""

    def test_loads_all(self) -> None:
        texts = load_texts()
        assert len(texts) == 50

    def test_filter_by_category(self) -> None:
        texts = load_texts(category="simple")
        assert len(texts) >= 3
        assert all(t["category"] == "simple" for t in texts)

    def test_filter_by_language(self) -> None:
        texts = load_texts(language="en")
        assert len(texts) >= 3
        assert all(
            str(t.get("language", "")).startswith("en") for t in texts
        )

    def test_filter_combined(self) -> None:
        texts = load_texts(category="simple", language="ru")
        assert len(texts) >= 2

    def test_texts_have_required_fields(self) -> None:
        texts = load_texts()
        for t in texts:
            assert "id" in t
            assert "text" in t
            assert "language" in t
            assert "category" in t


class TestComputeSummary:
    """Tests for aggregating benchmark results."""

    def test_all_passed(self) -> None:
        results = [
            BenchmarkResult(
                text_id=f"t{i}",
                text="hello",
                language="en",
                category="simple",
                difficulty="easy",
                status="completed",
                is_approved=True,
                wer=0.0,
                iterations=1,
                latency_seconds=1.0,
            )
            for i in range(5)
        ]
        summary = compute_summary(results)
        assert summary.total == 5
        assert summary.passed == 5
        assert summary.failed == 0
        assert summary.avg_wer == 0.0

    def test_mixed_results(self) -> None:
        results = [
            BenchmarkResult(
                text_id="t1",
                text="a",
                language="ru",
                category="simple",
                difficulty="easy",
                status="completed",
                is_approved=True,
                wer=0.0,
                iterations=1,
                latency_seconds=1.0,
            ),
            BenchmarkResult(
                text_id="t2",
                text="b",
                language="ru",
                category="hard",
                difficulty="hard",
                status="failed",
                is_approved=False,
                wer=0.5,
                latency_seconds=2.0,
            ),
            BenchmarkResult(
                text_id="t3",
                text="c",
                language="ru",
                category="simple",
                difficulty="easy",
                status="human_review",
                needs_human_review=True,
                wer=0.2,
                iterations=3,
                latency_seconds=3.0,
            ),
        ]
        summary = compute_summary(results)
        assert summary.total == 3
        assert summary.passed == 1
        assert summary.failed == 1
        assert summary.human_review == 1

    def test_empty_results(self) -> None:
        summary = compute_summary([])
        assert summary.total == 0
        assert summary.passed == 0


class TestReport:
    """Tests for report generation."""

    def test_report_is_markdown(self) -> None:
        results = [
            BenchmarkResult(
                text_id="t1",
                text="hello",
                language="en",
                category="simple",
                difficulty="easy",
                status="completed",
                is_approved=True,
                wer=0.01,
                iterations=1,
                latency_seconds=1.5,
            ),
        ]
        summary = compute_summary(results)
        report = print_report(results, summary)
        assert "# ReflexTTS Benchmark Report" in report
        assert "| t1 |" in report
        assert "0.010" in report


class TestBenchmarkTextsFile:
    """Tests for the benchmark texts JSON file."""

    def test_valid_json(self) -> None:
        path = Path("scripts/benchmark_texts.json")
        data = json.loads(path.read_text())
        assert isinstance(data, list)
        assert len(data) == 50

    def test_unique_ids(self) -> None:
        path = Path("scripts/benchmark_texts.json")
        data = json.loads(path.read_text())
        ids = [t["id"] for t in data]
        assert len(ids) == len(set(ids)), "Duplicate IDs found"

    def test_categories_covered(self) -> None:
        path = Path("scripts/benchmark_texts.json")
        data = json.loads(path.read_text())
        categories = {t["category"] for t in data}
        expected = {
            "simple", "dialog", "names", "numbers",
            "homographs", "emotion", "long",
        }
        assert expected.issubset(categories)
