#!/usr/bin/env python3
"""ReflexTTS Benchmark Runner.

Runs the full TTS pipeline on the benchmark text corpus
and collects quality metrics: WER, latency, RTF, loops.

Usage:
    python scripts/run_benchmarks.py
    python scripts/run_benchmarks.py --api-url http://localhost:8080
    python scripts/run_benchmarks.py --category homographs --language ru
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx

BENCHMARK_TEXTS = Path(__file__).parent / "benchmark_texts.json"
DEFAULT_API_URL = "http://localhost:8080"
POLL_INTERVAL = 0.5
MAX_WAIT_SECONDS = 120


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""

    text_id: str
    text: str
    language: str
    category: str
    difficulty: str
    duration_category: str = "medium"
    status: str = "pending"
    wer: float | None = None
    iterations: int = 0
    latency_seconds: float = 0.0
    audio_duration_seconds: float = 0.0
    rtf: float = 0.0  # Real-Time Factor (latency / audio_duration)
    is_approved: bool = False
    needs_human_review: bool = False
    error_message: str | None = None
    agent_log: list[dict[str, str]] = field(default_factory=list)


@dataclass
class BenchmarkSummary:
    """Aggregate summary of all benchmark results."""

    total: int = 0
    passed: int = 0
    failed: int = 0
    human_review: int = 0
    avg_wer: float = 0.0
    avg_latency: float = 0.0
    avg_rtf: float = 0.0
    avg_iterations: float = 0.0
    p50_latency: float = 0.0
    p95_latency: float = 0.0
    p99_latency: float = 0.0
    by_category: dict[str, dict[str, float]] = field(
        default_factory=dict
    )
    by_difficulty: dict[str, dict[str, float]] = field(
        default_factory=dict
    )


def load_texts(
    path: Path = BENCHMARK_TEXTS,
    category: str | None = None,
    language: str | None = None,
    duration: str | None = None,
) -> list[dict[str, object]]:
    """Load benchmark texts, optionally filtered."""
    with open(path) as f:
        texts: list[dict[str, object]] = json.load(f)

    if category:
        texts = [t for t in texts if t.get("category") == category]
    if language:
        texts = [
            t for t in texts
            if str(t.get("language", "")).startswith(language)
        ]
    if duration:
        texts = [
            t for t in texts
            if t.get("duration_category") == duration
        ]
    return texts


async def run_single(
    client: httpx.AsyncClient,
    api_url: str,
    text_item: dict[str, object],
    voice_id: str = "speaker_1",
) -> BenchmarkResult:
    """Run a single benchmark text through the API."""
    result = BenchmarkResult(
        text_id=str(text_item["id"]),
        text=str(text_item["text"]),
        language=str(text_item.get("language", "auto")),
        category=str(text_item.get("category", "unknown")),
        difficulty=str(text_item.get("difficulty", "unknown")),
        duration_category=str(text_item.get("duration_category", "medium")),
    )

    start_time = time.monotonic()

    try:
        # Start synthesis
        resp = await client.post(
            f"{api_url}/synthesize",
            json={"text": result.text, "voice_id": voice_id},
        )

        if resp.status_code != 202:
            result.status = "failed"
            result.error_message = f"HTTP {resp.status_code}: {resp.text}"
            return result

        session_id = resp.json()["session_id"]

        # Poll for completion
        for _ in range(int(MAX_WAIT_SECONDS / POLL_INTERVAL)):
            await asyncio.sleep(POLL_INTERVAL)

            status_resp = await client.get(
                f"{api_url}/session/{session_id}/status"
            )
            if status_resp.status_code != 200:
                continue

            status_data = status_resp.json()

            if status_data["status"] == "completed":
                result.status = "completed"
                result.wer = status_data.get("wer")
                result.iterations = status_data.get("iteration", 0)
                result.is_approved = status_data.get("is_approved", False)
                result.agent_log = status_data.get("agent_log", [])
                break

            if status_data["status"] == "failed":
                result.status = "failed"
                result.error_message = status_data.get("error_message")
                result.agent_log = status_data.get("agent_log", [])
                break

            if status_data.get("needs_human_review"):
                result.status = "human_review"
                result.needs_human_review = True
                result.wer = status_data.get("wer")
                result.iterations = status_data.get("iteration", 0)
                result.agent_log = status_data.get("agent_log", [])
                break
        else:
            result.status = "timeout"
            result.error_message = f"Timed out after {MAX_WAIT_SECONDS}s"

    except Exception as e:
        result.status = "error"
        result.error_message = str(e)

    result.latency_seconds = time.monotonic() - start_time

    # Estimate audio duration (rough: 150 chars/min speaking rate)
    char_count = len(result.text)
    result.audio_duration_seconds = max(1.0, char_count / 15.0)
    if result.audio_duration_seconds > 0:
        result.rtf = result.latency_seconds / result.audio_duration_seconds

    return result


def compute_summary(results: list[BenchmarkResult]) -> BenchmarkSummary:
    """Compute aggregate statistics from results."""
    summary = BenchmarkSummary(total=len(results))

    for r in results:
        if r.status == "completed" and r.is_approved:
            summary.passed += 1
        elif r.needs_human_review:
            summary.human_review += 1
        else:
            summary.failed += 1

    completed = [r for r in results if r.status == "completed"]

    if completed:
        wers = [r.wer for r in completed if r.wer is not None]
        latencies = sorted(r.latency_seconds for r in completed)
        rtfs = [r.rtf for r in completed]
        iters = [r.iterations for r in completed]

        summary.avg_wer = sum(wers) / len(wers) if wers else 0
        summary.avg_latency = sum(latencies) / len(latencies)
        summary.avg_rtf = sum(rtfs) / len(rtfs)
        summary.avg_iterations = sum(iters) / len(iters)

        n = len(latencies)
        summary.p50_latency = latencies[n // 2] if n > 0 else 0
        summary.p95_latency = latencies[int(n * 0.95)] if n > 0 else 0
        summary.p99_latency = latencies[int(n * 0.99)] if n > 0 else 0

    # Group by category
    for r in results:
        cat = r.category
        if cat not in summary.by_category:
            summary.by_category[cat] = {
                "total": 0, "passed": 0, "avg_wer": 0
            }
        summary.by_category[cat]["total"] += 1
        if r.status == "completed" and r.is_approved:
            summary.by_category[cat]["passed"] += 1
        if r.wer is not None:
            summary.by_category[cat]["avg_wer"] += r.wer

    for _cat, stats in summary.by_category.items():
        total = stats["total"]
        if total > 0:
            stats["avg_wer"] /= total

    return summary


def print_report(
    results: list[BenchmarkResult],
    summary: BenchmarkSummary,
) -> str:
    """Generate a markdown report."""
    lines = [
        "# ReflexTTS Benchmark Report\n",
        f"**Total:** {summary.total} | "
        f"**Passed:** {summary.passed} | "
        f"**Failed:** {summary.failed} | "
        f"**Human Review:** {summary.human_review}\n",
        "## Aggregate Metrics\n",
        "| Metric | Value | Target |",
        "|--------|-------|--------|",
        f"| Avg WER | {summary.avg_wer:.3f} | < 0.01 |",
        f"| Avg Latency | {summary.avg_latency:.2f}s | — |",
        f"| Avg RTF | {summary.avg_rtf:.2f} | < 4.0 |",
        f"| Avg Iterations | {summary.avg_iterations:.1f} | < 2.5 |",
        f"| P50 Latency | {summary.p50_latency:.2f}s | — |",
        f"| P95 Latency | {summary.p95_latency:.2f}s | — |",
        f"| P99 Latency | {summary.p99_latency:.2f}s | — |",
        "",
        "## By Category\n",
        "| Category | Total | Passed | Avg WER |",
        "|----------|-------|--------|---------|",
    ]

    for cat, stats in sorted(summary.by_category.items()):
        lines.append(
            f"| {cat} | {stats['total']:.0f} | "
            f"{stats['passed']:.0f} | {stats['avg_wer']:.3f} |"
        )

    lines.append("\n## Individual Results\n")
    lines.append(
        "| ID | Lang | Category | Duration | Status | WER "
        "| Loops | Latency |"
    )
    lines.append(
        "|-----|------|----------|----------|--------|-----"
        "|-------|---------|")

    for r in results:
        wer_str = f"{r.wer:.3f}" if r.wer is not None else "—"
        lines.append(
            f"| {r.text_id} | {r.language} | {r.category} | "
            f"{r.duration_category} | "
            f"{r.status} | {wer_str} | {r.iterations} | "
            f"{r.latency_seconds:.1f}s |"
        )

    report = "\n".join(lines)
    return report


async def main(args: argparse.Namespace) -> None:
    """Run the benchmark suite."""
    texts = load_texts(
        category=args.category,
        language=args.language,
        duration=args.duration,
    )

    print(f"Loaded {len(texts)} benchmark texts")
    print(f"API: {args.api_url}")
    print(f"Voice: {args.voice_id}")
    print("---")

    results: list[BenchmarkResult] = []

    async with httpx.AsyncClient(timeout=150.0) as client:
        for i, text_item in enumerate(texts):
            print(
                f"[{i+1}/{len(texts)}] {text_item['id']}: "
                f"{str(text_item['text'])[:60]}..."
            )
            result = await run_single(
                client, args.api_url, text_item, args.voice_id
            )
            results.append(result)
            status_icon = {
                "completed": "✅",
                "failed": "❌",
                "human_review": "⚠️",
                "timeout": "⏰",
                "error": "💥",
            }.get(result.status, "?")
            wer_str = (
                f"WER={result.wer:.3f}" if result.wer is not None else ""
            )
            print(
                f"  {status_icon} {result.status} {wer_str} "
                f"({result.latency_seconds:.1f}s, "
                f"{result.iterations} loops)"
            )

    summary = compute_summary(results)
    report = print_report(results, summary)

    # Save report
    output_path = Path(args.output)
    output_path.write_text(report)
    print(f"\nReport saved to {output_path}")

    # Save raw results
    raw_path = output_path.with_suffix(".json")
    raw_path.write_text(
        json.dumps([asdict(r) for r in results], indent=2, ensure_ascii=False)
    )
    print(f"Raw results saved to {raw_path}")

    # Print summary
    print(f"\n{'='*50}")
    print(f"TOTAL: {summary.total}")
    print(f"PASSED: {summary.passed}")
    print(f"FAILED: {summary.failed}")
    print(f"HUMAN_REVIEW: {summary.human_review}")
    print(f"AVG WER: {summary.avg_wer:.3f}")
    print(f"AVG RTF: {summary.avg_rtf:.2f}")
    print(f"AVG ITERATIONS: {summary.avg_iterations:.1f}")

    # Exit code
    if summary.failed > summary.total * 0.1:
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReflexTTS Benchmark Runner")
    parser.add_argument(
        "--api-url", default=DEFAULT_API_URL, help="API base URL"
    )
    parser.add_argument(
        "--voice-id", default="speaker_1", help="Voice ID"
    )
    parser.add_argument(
        "--category", default=None, help="Filter by category"
    )
    parser.add_argument(
        "--language", default=None, help="Filter by language"
    )
    parser.add_argument(
        "--duration",
        default=None,
        choices=["short", "medium", "long"],
        help="Filter by duration category",
    )
    parser.add_argument(
        "--output",
        default="docs/benchmark_report.md",
        help="Output report path",
    )
    parsed_args = parser.parse_args()
    asyncio.run(main(parsed_args))
