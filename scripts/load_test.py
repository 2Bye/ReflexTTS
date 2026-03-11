#!/usr/bin/env python3
"""Locust load test for ReflexTTS API.

Usage:
    pip install locust
    locust -f scripts/load_test.py --host http://localhost:8080
    # Open http://localhost:8089 for the Locust web UI

    # Headless mode:
    locust -f scripts/load_test.py --host http://localhost:8080 \
           -u 5 -r 1 -t 5m --headless
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

from locust import HttpUser, between, task

BENCHMARK_TEXTS = Path(__file__).parent / "benchmark_texts.json"


def _load_texts() -> list[dict[str, object]]:
    """Load benchmark texts for load testing."""
    if BENCHMARK_TEXTS.exists():
        with open(BENCHMARK_TEXTS) as f:
            return json.load(f)  # type: ignore[no-any-return]
    return [
        {"text": "Hello world", "id": "fallback_01"},
        {"text": "Testing TTS synthesis", "id": "fallback_02"},
    ]


_TEXTS = _load_texts()


class TTSUser(HttpUser):
    """Simulates a user making TTS synthesis requests."""

    wait_time = between(2, 8)  # Seconds between tasks

    @task(3)
    def synthesize_random(self) -> None:
        """Submit a random text for synthesis."""
        text_item = random.choice(_TEXTS)  # noqa: S311
        text = str(text_item.get("text", "Hello"))

        with self.client.post(
            "/synthesize",
            json={"text": text, "voice_id": "speaker_1"},
            catch_response=True,
            name="/synthesize",
        ) as response:
            if response.status_code == 202:
                data = response.json()
                session_id = data.get("session_id")
                if session_id:
                    response.success()
                    # Poll for completion
                    self._wait_for_completion(session_id)
                else:
                    response.failure("No session_id in response")
            elif response.status_code == 400:
                response.success()  # Expected for injection tests
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(1)
    def check_health(self) -> None:
        """Check the health endpoint."""
        self.client.get("/health")

    @task(1)
    def list_voices(self) -> None:
        """List available voices."""
        self.client.get("/voices")

    def _wait_for_completion(
        self, session_id: str, max_polls: int = 60
    ) -> None:
        """Poll session status until completion."""
        for _ in range(max_polls):
            time.sleep(0.5)
            with self.client.get(
                f"/session/{session_id}/status",
                name="/session/[id]/status",
                catch_response=True,
            ) as resp:
                if resp.status_code != 200:
                    resp.failure(f"Status poll failed: {resp.status_code}")
                    return

                status = resp.json().get("status")
                resp.success()

                if status in ("completed", "failed"):
                    if status == "completed":
                        # Try to download audio
                        self.client.get(
                            f"/session/{session_id}/audio",
                            name="/session/[id]/audio",
                        )
                    return

    def on_start(self) -> None:
        """Verify API is reachable on start."""
        resp = self.client.get("/health")
        if resp.status_code != 200:
            raise RuntimeError("API not reachable")
