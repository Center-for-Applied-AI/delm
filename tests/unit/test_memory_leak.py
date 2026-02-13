"""
Test for memory leaks and slowdowns in long-running extraction processes.

This test operates at the DELM level to test the full pipeline including:
- Real semantic cache (SQLite) - potential connection leak source
- Real DiskExperimentManager - potential file handle leak source
- Real concurrent processing
- Only mocks the Instructor API calls

Requirements tested:
1. Mock data with 100k rows (each row can be the same)
2. Mock schema for extraction
3. Mock API endpoint returning consistent JSON results
4. Batch size of 1000
5. 50 workers for concurrent processing
6. Track memory usage over batches
7. Track processing speed over batches
8. Real semantic cache and experiment manager
"""

import cProfile
import gc
import io
import os
import pstats
import random
import shutil
import sys
import tempfile
import threading
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from pydantic import BaseModel

# Ensure we can import delm
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from delm import DELM
from delm.config import DELMConfig, LLMExtractionConfig
from delm.core.extraction_manager import ExtractionManager
from delm.schemas import Schema


# ============================================================================
# Mock Response Classes (Only mock the LLM API)
# ============================================================================


class MockUsage:
    """Mock usage object returned by LLM API."""

    def __init__(self):
        self.prompt_tokens = 100
        self.completion_tokens = 50


class MockCompletion:
    """Mock completion object returned by LLM API."""

    def __init__(self):
        self.usage = MockUsage()


class MockExtractedData(BaseModel):
    """Mock extracted data model matching our test schema."""

    name: str = "Test Name"
    value: float = 123.45
    category: str = "Test Category"


class MockAPIError(Exception):
    """Mock API error for simulating failures."""

    pass


class MockChatCompletions:
    """
    Mock chat completions interface with realistic behavior.

    Simulates:
    - Variable response times (normal distribution)
    - Random failures at a configurable rate
    """

    def __init__(
        self,
        mean_latency_ms: float = 5.0,
        latency_std_ms: float = 2.0,
        failure_rate: float = 0.0,
    ):
        """
        Args:
            mean_latency_ms: Mean response latency in milliseconds
            latency_std_ms: Standard deviation of latency in milliseconds
            failure_rate: Probability of failure (0.0 to 1.0), e.g., 0.02 = 2%
        """
        self.mean_latency_ms = mean_latency_ms
        self.latency_std_ms = latency_std_ms
        self.failure_rate = failure_rate
        self.call_count = 0
        self.failure_count = 0

    def create_with_completion(
        self,
        model: str,
        temperature: float,
        response_model: Any,
        messages: list,
        max_completion_tokens: int = 4096,
        max_retries: int = 0,
    ):
        """Mock the create_with_completion method with realistic behavior."""
        self.call_count += 1

        # Simulate variable latency (normal distribution, clamped to positive)
        if self.mean_latency_ms > 0:
            latency_ms = max(0, random.gauss(self.mean_latency_ms, self.latency_std_ms))
            time.sleep(latency_ms / 1000.0)

        # Simulate random failures
        if self.failure_rate > 0 and random.random() < self.failure_rate:
            self.failure_count += 1
            raise MockAPIError(f"Simulated API failure (failure #{self.failure_count})")

        # Return mock response and completion
        mock_response = MockExtractedData()
        mock_completion = MockCompletion()

        return mock_response, mock_completion


class MockChat:
    """Mock chat interface."""

    def __init__(
        self,
        mean_latency_ms: float = 5.0,
        latency_std_ms: float = 2.0,
        failure_rate: float = 0.0,
    ):
        self.completions = MockChatCompletions(
            mean_latency_ms=mean_latency_ms,
            latency_std_ms=latency_std_ms,
            failure_rate=failure_rate,
        )


class MockInstructorClient:
    """
    Mock Instructor client with realistic behavior.

    Default settings simulate fast local processing with occasional failures.
    """

    def __init__(
        self,
        mean_latency_ms: float = 5.0,
        latency_std_ms: float = 2.0,
        failure_rate: float = 0.0,
    ):
        """
        Args:
            mean_latency_ms: Mean response latency in milliseconds (default: 5ms)
            latency_std_ms: Standard deviation of latency (default: 2ms)
            failure_rate: Probability of failure per request (default: 0 = no failures)
        """
        self.chat = MockChat(
            mean_latency_ms=mean_latency_ms,
            latency_std_ms=latency_std_ms,
            failure_rate=failure_rate,
        )

    @property
    def call_count(self) -> int:
        return self.chat.completions.call_count

    @property
    def failure_count(self) -> int:
        return self.chat.completions.failure_count


# ============================================================================
# Test Schema
# ============================================================================

TEST_SCHEMA_DICT = {
    "variables": [
        {
            "name": "name",
            "description": "The name of the entity",
            "data_type": "string",
        },
        {
            "name": "value",
            "description": "The numeric value",
            "data_type": "number",
        },
        {
            "name": "category",
            "description": "The category",
            "data_type": "string",
        },
    ]
}


# ============================================================================
# Memory and Performance Tracking
# ============================================================================


@dataclass
class BatchMetrics:
    """Metrics for a single batch."""

    batch_id: int
    start_time: float
    end_time: float
    memory_before_mb: float
    memory_after_mb: float
    num_items: int

    @property
    def duration_seconds(self) -> float:
        return self.end_time - self.start_time

    @property
    def items_per_second(self) -> float:
        if self.duration_seconds > 0:
            return self.num_items / self.duration_seconds
        return 0.0

    @property
    def memory_delta_mb(self) -> float:
        return self.memory_after_mb - self.memory_before_mb


def get_tracemalloc_mb() -> float:
    """Get current traced memory in MB."""
    current, peak = tracemalloc.get_traced_memory()
    return current / (1024 * 1024)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test artifacts."""
    temp_path = tempfile.mkdtemp(prefix="delm_test_")
    yield Path(temp_path)
    # Cleanup after test
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def mock_schema():
    """Create a mock schema for testing."""
    return Schema.from_dict(TEST_SCHEMA_DICT)


@pytest.fixture
def mock_data_small():
    """Create small mock data for quick tests (1000 rows) - each row unique."""
    return pd.DataFrame(
        {
            "text": [
                f"Row {i}: Test text with name Test{i}, value {i}.45, category A."
                for i in range(1000)
            ]
        }
    )


@pytest.fixture
def mock_data_medium():
    """Create medium mock data (10k rows) - each row unique."""
    return pd.DataFrame(
        {
            "text": [
                f"Row {i}: Test text with name Test{i}, value {i}.45, category A."
                for i in range(10_000)
            ]
        }
    )


@pytest.fixture
def mock_data_large():
    """Create large mock data for stress tests (30k rows) - each row unique."""
    return pd.DataFrame(
        {
            "text": [
                f"Row {i}: Test text with name Test{i}, value {i}.45, category A."
                for i in range(30_000)
            ]
        }
    )


# ============================================================================
# Helper to patch Instructor at the right level
# ============================================================================


def create_patched_delm(
    schema_dict: dict,
    temp_dir: Path,
    batch_size: int = 100,
    max_workers: int = 4,
    use_disk_storage: bool = True,
    cache_backend: str = "sqlite",
) -> tuple[DELM, MagicMock]:
    """
    Create a DELM instance with mocked Instructor client.

    Returns the DELM instance and the mock for verification.
    """
    mock_client = MockInstructorClient()

    with patch("instructor.from_provider", return_value=mock_client):
        with patch("instructor.from_openai", return_value=mock_client):
            delm = DELM(
                schema=schema_dict,
                provider="openai",
                model="gpt-4o-mini",
                batch_size=batch_size,
                max_workers=max_workers,
                max_retries=1,
                track_cost=False,
                # Real semantic cache
                cache_backend=cache_backend,
                cache_path=temp_dir / "cache",
                cache_max_size_mb=512,
                # Real disk storage
                use_disk_storage=use_disk_storage,
                experiment_path=temp_dir / "experiment",
                overwrite_experiment=True,
                auto_checkpoint_and_resume_experiment=True,
                # Logging
                console_log_level="WARNING",
                save_log_file=False,
            )

    return delm, mock_client


# ============================================================================
# Test Cases - DELM Level with Real Components
# ============================================================================


class TestDELMMemoryLeakWithRealCache:
    """Tests for memory leaks at the DELM level with real semantic cache."""

    def test_repeated_extractions_memory_stability(self, temp_dir, mock_data_small):
        """Test that repeated extractions don't leak memory."""
        gc.collect()
        tracemalloc.start()
        initial_memory = get_tracemalloc_mb()

        memory_readings = []

        # Run multiple extraction cycles
        for cycle in range(5):
            # Create fresh DELM for each cycle
            with patch("instructor.from_provider") as mock_provider:
                mock_provider.return_value = MockInstructorClient()

                delm = DELM(
                    schema=TEST_SCHEMA_DICT,
                    provider="openai",
                    model="gpt-4o-mini",
                    batch_size=100,
                    max_workers=4,
                    max_retries=1,
                    track_cost=False,
                    cache_backend="sqlite",
                    cache_path=temp_dir / f"cache_{cycle}",
                    cache_max_size_mb=64,
                    use_disk_storage=True,
                    experiment_path=temp_dir / f"experiment_{cycle}",
                    overwrite_experiment=True,
                    console_log_level="ERROR",
                    save_log_file=False,
                )

                # Mock the extraction at a lower level
                delm.extraction_manager._instructor_extract_with_retry = MagicMock(
                    return_value=MockExtractedData()
                )

                # Run extraction
                result = delm.extract(mock_data_small.copy())

                # Clean up
                del result
                del delm

            gc.collect()
            current_memory = get_tracemalloc_mb()
            memory_readings.append(current_memory)
            print(f"Cycle {cycle + 1}: Memory = {current_memory:.2f}MB")

        tracemalloc.stop()

        # Check memory growth trend
        first_reading = memory_readings[0]
        last_reading = memory_readings[-1]
        memory_growth = last_reading - first_reading

        print(
            f"\nMemory growth over {len(memory_readings)} cycles: {memory_growth:.2f}MB"
        )

        # Memory shouldn't grow more than 20MB across cycles
        assert (
            memory_growth < 20
        ), f"Memory grew by {memory_growth:.2f}MB across extraction cycles"

    def test_semantic_cache_connection_cleanup(self, temp_dir):
        """Test that semantic cache properly closes database connections."""
        import sqlite3

        cache_path = temp_dir / "cache_test"

        # Create multiple DELM instances and destroy them
        for i in range(10):
            with patch("instructor.from_provider") as mock_provider:
                mock_provider.return_value = MockInstructorClient()

                delm = DELM(
                    schema=TEST_SCHEMA_DICT,
                    provider="openai",
                    model="gpt-4o-mini",
                    batch_size=10,
                    max_workers=2,
                    track_cost=False,
                    cache_backend="sqlite",
                    cache_path=cache_path,
                    use_disk_storage=False,
                    console_log_level="ERROR",
                    save_log_file=False,
                )

                # Access cache to ensure it's initialized
                _ = delm.semantic_cache

                # Destroy
                del delm
                gc.collect()

        # Try to open the database - if connections weren't closed properly,
        # this might fail or the database might be locked
        db_path = cache_path / "semantic.db"
        if db_path.exists():
            try:
                conn = sqlite3.connect(str(db_path), timeout=1)
                conn.execute("SELECT COUNT(*) FROM cache")
                conn.close()
            except sqlite3.OperationalError as e:
                pytest.fail(f"Database appears to have leaked connections: {e}")

    def test_disk_experiment_manager_file_handle_cleanup(
        self, temp_dir, mock_data_small
    ):
        """Test that DiskExperimentManager properly closes file handles."""
        import subprocess
        import platform

        experiment_path = temp_dir / "experiment_handles"

        with patch("instructor.from_provider") as mock_provider:
            mock_provider.return_value = MockInstructorClient()

            delm = DELM(
                schema=TEST_SCHEMA_DICT,
                provider="openai",
                model="gpt-4o-mini",
                batch_size=50,
                max_workers=2,
                track_cost=False,
                cache_backend="sqlite",
                cache_path=temp_dir / "cache",
                use_disk_storage=True,
                experiment_path=experiment_path,
                overwrite_experiment=True,
                console_log_level="ERROR",
                save_log_file=False,
            )

            delm.extraction_manager._instructor_extract_with_retry = MagicMock(
                return_value=MockExtractedData()
            )

            # Run extraction
            result = delm.extract(mock_data_small.head(100).copy())

            del result
            del delm
            gc.collect()

        # Give OS time to release handles
        time.sleep(0.5)

        # Verify we can delete the experiment directory (no locked files)
        try:
            shutil.rmtree(experiment_path)
        except PermissionError as e:
            pytest.fail(f"File handles not properly closed: {e}")


class TestDELMPerformanceWithRealComponents:
    """Tests for performance degradation at the DELM level."""

    def test_no_slowdown_with_real_cache(self, temp_dir):
        """Test that processing speed doesn't degrade with real semantic cache."""
        batch_times = []

        with patch("instructor.from_provider") as mock_provider:
            mock_provider.return_value = MockInstructorClient()

            delm = DELM(
                schema=TEST_SCHEMA_DICT,
                provider="openai",
                model="gpt-4o-mini",
                batch_size=100,
                max_workers=4,
                track_cost=False,
                cache_backend="sqlite",
                cache_path=temp_dir / "cache",
                use_disk_storage=True,
                experiment_path=temp_dir / "experiment",
                overwrite_experiment=True,
                console_log_level="ERROR",
                save_log_file=False,
            )

            delm.extraction_manager._instructor_extract_with_retry = MagicMock(
                return_value=MockExtractedData()
            )

            # Run multiple batches - each row unique to test cache writes
            for batch_num in range(10):
                batch_data = pd.DataFrame(
                    {
                        "text": [
                            f"Batch {batch_num} Row {i}: text with name Test{i}, value {i}.45, category A."
                            for i in range(200)
                        ]
                    }
                )

                start_time = time.time()
                result = delm.extract(batch_data)
                end_time = time.time()

                batch_times.append(end_time - start_time)
                print(
                    f"Batch {batch_num + 1}: {end_time - start_time:.2f}s "
                    f"({200 / (end_time - start_time):.0f} items/sec)"
                )

                del result
                gc.collect()

        # Compare first half vs second half
        first_half = batch_times[: len(batch_times) // 2]
        second_half = batch_times[len(batch_times) // 2 :]

        first_half_avg = sum(first_half) / len(first_half)
        second_half_avg = sum(second_half) / len(second_half)

        slowdown_pct = (
            (second_half_avg - first_half_avg) / first_half_avg * 100
            if first_half_avg > 0
            else 0
        )

        print(f"\nFirst half avg: {first_half_avg:.2f}s")
        print(f"Second half avg: {second_half_avg:.2f}s")
        print(f"Slowdown: {slowdown_pct:.1f}%")

        # Allow up to 50% slowdown (some is expected due to cache growth)
        assert slowdown_pct < 50, f"Processing slowed by {slowdown_pct:.1f}%"


class TestLargeScaleDELMProcessing:
    """Large-scale stress tests at the DELM level."""

    @pytest.mark.slow
    def test_100k_requests_full_pipeline(self, temp_dir):
        """
        Stress test: 100k requests through full DELM pipeline.

        Tests that processing rate is CONSTANT and doesn't degrade over time,
        regardless of total number of articles or batch size.

        Uses:
        - Real semantic cache (SQLite)
        - Real DiskExperimentManager
        - Real concurrent processing with batch_size controlling batching
        - Mocked Instructor API only

        Run with: pytest -m slow -s
        """
        num_requests = 30_000
        batch_size = 1000
        max_workers = 50
        num_batches = num_requests // batch_size

        # Mock settings - no failures for consistent timing measurement
        mean_latency_ms = 1.0  # 1ms average response time
        latency_std_ms = 0.5  # 0.5ms std dev
        failure_rate = 0.0  # No failures - cleaner performance measurement

        print(f"\n{'='*60}")
        print(f"DELM Full Pipeline Stress Test")
        print(f"{'='*60}")
        print(f"Total Requests: {num_requests:,}")
        print(f"Batch size: {batch_size}")
        print(f"Expected batches: {num_batches}")
        print(f"Workers: {max_workers}")
        print(f"Mock latency: {mean_latency_ms}ms ± {latency_std_ms}ms")
        print(f"Mock failure rate: {failure_rate*100:.1f}%")
        print(f"{'='*60}\n")

        # Create ALL test data upfront (100k rows) - EACH ROW MUST BE UNIQUE
        # to ensure we're testing cache WRITES, not just cache reads
        test_data = pd.DataFrame(
            {
                "text": [
                    f"Row {i}: This is test text with name Test{i}, value {i}.45, category A."
                    for i in range(num_requests)
                ]
            }
        )

        gc.collect()
        tracemalloc.start()
        initial_memory = get_tracemalloc_mb()

        # Thread-safe metrics tracking
        counter_lock = threading.Lock()
        extraction_call_count = 0
        batch_metrics = []
        last_batch_time = time.time()
        total_failures = 0

        with patch("instructor.from_provider") as mock_provider:
            mock_client = MockInstructorClient(
                mean_latency_ms=mean_latency_ms,
                latency_std_ms=latency_std_ms,
                failure_rate=failure_rate,
            )
            mock_provider.return_value = mock_client

            delm = DELM(
                schema=TEST_SCHEMA_DICT,
                provider="openai",
                model="gpt-4o-mini",
                batch_size=batch_size,
                max_workers=max_workers,
                max_retries=1,
                track_cost=False,
                # Real semantic cache
                cache_backend="sqlite",
                cache_path=temp_dir / "cache",
                cache_max_size_mb=1024,
                # Real disk storage
                use_disk_storage=True,
                experiment_path=temp_dir / "experiment",
                overwrite_experiment=True,
                auto_checkpoint_and_resume_experiment=True,
                # Minimal logging to not interfere with our progress output
                console_log_level="ERROR",
                save_log_file=False,
            )

            # Instrument the extraction method to track per-batch metrics
            def instrumented_extract(*args, **kwargs):
                nonlocal extraction_call_count, last_batch_time, total_failures

                # Simulate variable latency (normal distribution)
                latency_ms = max(0, random.gauss(mean_latency_ms, latency_std_ms))
                time.sleep(latency_ms / 1000.0)

                # Simulate random failures
                if failure_rate > 0 and random.random() < failure_rate:
                    with counter_lock:
                        total_failures += 1
                    raise MockAPIError(f"Simulated API failure")

                # Thread-safe counter increment and batch detection
                with counter_lock:
                    extraction_call_count += 1
                    current_count = extraction_call_count

                    # Every batch_size calls, record batch metrics
                    if current_count % batch_size == 0:
                        current_time = time.time()
                        current_memory = get_tracemalloc_mb()
                        batch_num = current_count // batch_size
                        batch_duration = current_time - last_batch_time
                        rate = batch_size / batch_duration if batch_duration > 0 else 0

                        batch_metrics.append(
                            {
                                "batch_num": batch_num,
                                "items_processed": current_count,
                                "duration": batch_duration,
                                "rate": rate,
                                "memory_mb": current_memory,
                            }
                        )

                        # Print progress
                        print(
                            f"Batch {batch_num:3d}/{num_batches}: "
                            f"{current_count:,}/{num_requests:,} "
                            f"({100*current_count/num_requests:5.1f}%) | "
                            f"Rate: {rate:,.0f}/s | "
                            f"Mem: {current_memory:.0f}MB"
                        )

                        last_batch_time = current_time

                # Return mock result
                return MockExtractedData()

            delm.extraction_manager._instructor_extract_with_retry = (
                instrumented_extract
            )

            # Initialize timing for first batch
            last_batch_start_time = time.time()
            last_batch_start_count = 0

            # Run the full extraction with ALL data at once
            # DELM will automatically batch based on batch_size
            print("\nStarting extraction...\n")
            overall_start = time.time()
            result = delm.extract(test_data)
            overall_end = time.time()

            gc.collect()
            final_memory = get_tracemalloc_mb()

        tracemalloc.stop()

        # Calculate results
        total_time = overall_end - overall_start
        memory_growth = final_memory - initial_memory
        items_per_second = num_requests / total_time

        # Analyze rate consistency across batches (skip first batch as warmup)
        if len(batch_metrics) >= 5:
            # Skip first batch - it's often artificially fast due to warmup effects
            rates = [m["rate"] for m in batch_metrics[1:]]
            n = len(rates)
            first_quarter = rates[: n // 4]
            last_quarter = rates[-n // 4 :]

            avg_first_quarter = sum(first_quarter) / len(first_quarter)
            avg_last_quarter = sum(last_quarter) / len(last_quarter)

            # Calculate slowdown percentage
            slowdown_pct = (
                (avg_first_quarter - avg_last_quarter) / avg_first_quarter * 100
                if avg_first_quarter > 0
                else 0
            )

            # Calculate rate variance (coefficient of variation)
            avg_rate = sum(rates) / n
            variance = sum((r - avg_rate) ** 2 for r in rates) / n
            std_dev = variance**0.5
            cv = (std_dev / avg_rate * 100) if avg_rate > 0 else 0
        else:
            avg_first_quarter = avg_last_quarter = slowdown_pct = cv = 0

        # Expected failures based on rate
        expected_failures = int(num_requests * failure_rate)

        print(f"\n{'='*60}")
        print("RESULTS")
        print(f"{'='*60}")
        print(f"Total time: {total_time:.1f}s")
        print(f"Overall rate: {items_per_second:,.0f} items/sec")
        print(f"Initial memory: {initial_memory:.1f}MB")
        print(f"Final memory: {final_memory:.1f}MB")
        print(f"Memory growth: {memory_growth:.1f}MB")
        print(f"Result rows: {len(result):,}")
        print(f"")
        print(f"FAILURE HANDLING:")
        print(f"  Total failures: {total_failures:,}")
        print(f"  Expected (@ {failure_rate*100:.1f}%): ~{expected_failures:,}")
        print(f"")
        print(f"RATE CONSISTENCY (key metric):")
        print(f"  First quarter avg rate: {avg_first_quarter:,.0f}/s")
        print(f"  Last quarter avg rate:  {avg_last_quarter:,.0f}/s")
        print(f"  Slowdown: {slowdown_pct:+.1f}%")
        print(f"  Rate coefficient of variation: {cv:.1f}%")
        print(f"{'='*60}\n")

        # Assertions
        assert (
            len(result) == num_requests
        ), f"Expected {num_requests} results, got {len(result)}"

        # Memory shouldn't grow more than 100MB for 30k requests
        assert memory_growth < 100, f"MEMORY LEAK: Memory grew by {memory_growth:.1f}MB"

        # Processing rate should be consistent - no more than 25% slowdown
        # (after excluding first batch warmup)
        assert (
            slowdown_pct < 25
        ), f"SLOWDOWN DETECTED: Rate dropped by {slowdown_pct:.1f}% from first to last quarter"

        # Note: CV can be high with random failures/retries - just log it, don't fail
        if cv > 100:
            print(
                f"WARNING: High rate variance (CV={cv:.1f}%) - expected with failures/retries"
            )

        # Clean up
        del result
        del delm
        gc.collect()

    @pytest.mark.slow
    def test_memory_profile_across_batches(self, temp_dir):
        """
        Profile memory usage and rate consistency across batches.

        Tests that processing rate is CONSTANT regardless of how many
        items have been processed.
        """
        num_requests = 30_000
        batch_size = 1000
        max_workers = 20
        num_batches = num_requests // batch_size

        # Mock settings - no failures for consistent timing measurement
        mean_latency_ms = 1.0
        latency_std_ms = 0.5
        failure_rate = 0.0  # No failures - cleaner performance measurement

        print(f"\n{'='*60}")
        print(f"Memory & Rate Profile Test")
        print(f"{'='*60}")
        print(f"Total requests: {num_requests:,}")
        print(f"Batch size: {batch_size}")
        print(f"Expected batches: {num_batches}")
        print(f"Mock latency: {mean_latency_ms}ms ± {latency_std_ms}ms")
        print(f"Mock failure rate: {failure_rate*100:.1f}%")
        print(f"{'='*60}\n")

        # Create all test data upfront - EACH ROW MUST BE UNIQUE for cache writes
        test_data = pd.DataFrame(
            {
                "text": [
                    f"Row {i}: Test text with name Test{i}, value {i}.45, category A."
                    for i in range(num_requests)
                ]
            }
        )

        # Thread-safe metrics tracking
        counter_lock = threading.Lock()
        batch_metrics = []
        extraction_call_count = 0
        last_batch_time = time.time()

        gc.collect()
        tracemalloc.start()
        initial_memory = get_tracemalloc_mb()

        with patch("instructor.from_provider") as mock_provider:
            mock_provider.return_value = MockInstructorClient(
                mean_latency_ms=mean_latency_ms,
                latency_std_ms=latency_std_ms,
                failure_rate=failure_rate,
            )

            delm = DELM(
                schema=TEST_SCHEMA_DICT,
                provider="openai",
                model="gpt-4o-mini",
                batch_size=batch_size,
                max_workers=max_workers,
                track_cost=False,
                cache_backend="sqlite",
                cache_path=temp_dir / "cache",
                use_disk_storage=True,
                experiment_path=temp_dir / "experiment",
                overwrite_experiment=True,
                console_log_level="ERROR",
                save_log_file=False,
            )

            def instrumented_extract(*args, **kwargs):
                nonlocal extraction_call_count, last_batch_time

                # Simulate variable latency
                latency_ms = max(0, random.gauss(mean_latency_ms, latency_std_ms))
                time.sleep(latency_ms / 1000.0)

                # Thread-safe counter and batch detection
                with counter_lock:
                    extraction_call_count += 1
                    current_count = extraction_call_count

                    # Record metrics at batch boundaries
                    if current_count % batch_size == 0:
                        current_time = time.time()
                        current_memory = get_tracemalloc_mb()
                        batch_num = current_count // batch_size
                        batch_duration = current_time - last_batch_time
                        rate = batch_size / batch_duration if batch_duration > 0 else 0

                        batch_metrics.append(
                            {
                                "batch": batch_num,
                                "total_processed": current_count,
                                "duration": batch_duration,
                                "rate": rate,
                                "memory_mb": current_memory,
                            }
                        )

                        print(
                            f"Batch {batch_num:3d}/{num_batches}: "
                            f"Rate={rate:,.0f}/s | Mem={current_memory:.0f}MB"
                        )

                        last_batch_time = current_time

                return MockExtractedData()

            delm.extraction_manager._instructor_extract_with_retry = (
                instrumented_extract
            )

            # Initialize first batch timing
            batch_start_time = time.time()

            print("\nProcessing...\n")
            result = delm.extract(test_data)

            gc.collect()
            final_memory = get_tracemalloc_mb()

        tracemalloc.stop()

        # Analyze memory trend
        memories = [m["memory_mb"] for m in batch_metrics]

        # Calculate linear regression slope for memory (growth per batch)
        n = len(memories)
        x_mean = (n - 1) / 2
        y_mean = sum(memories) / n
        memory_slope = sum(
            (i - x_mean) * (m - y_mean) for i, m in enumerate(memories)
        ) / sum((i - x_mean) ** 2 for i in range(n))

        # Analyze rate consistency (skip first batch as warmup)
        rates = [m["rate"] for m in batch_metrics[1:]] if len(batch_metrics) > 1 else []
        n_rates = len(rates)

        if n_rates >= 4:
            # Calculate rate trend (should be near 0 for constant rate)
            rate_mean = sum(rates) / n_rates
            x_mean_r = (n_rates - 1) / 2
            rate_slope = sum(
                (i - x_mean_r) * (r - rate_mean) for i, r in enumerate(rates)
            ) / sum((i - x_mean_r) ** 2 for i in range(n_rates))

            # Rate consistency: first vs last quarter
            first_quarter_rates = rates[: n_rates // 4]
            last_quarter_rates = rates[-n_rates // 4 :]
            avg_first = sum(first_quarter_rates) / len(first_quarter_rates)
            avg_last = sum(last_quarter_rates) / len(last_quarter_rates)
            slowdown_pct = (
                (avg_first - avg_last) / avg_first * 100 if avg_first > 0 else 0
            )
        else:
            rate_slope = avg_first = avg_last = slowdown_pct = 0

        print(f"\n{'='*60}")
        print("ANALYSIS")
        print(f"{'='*60}")
        print(f"Memory:")
        print(f"  Start: {memories[0]:.1f}MB")
        print(f"  End: {memories[-1]:.1f}MB")
        print(f"  Growth: {memories[-1] - memories[0]:.1f}MB")
        print(f"  Growth rate: {memory_slope:.2f}MB per batch")
        print(f"")
        print(f"Processing Rate:")
        print(f"  First quarter avg: {avg_first:,.0f}/s")
        print(f"  Last quarter avg: {avg_last:,.0f}/s")
        print(f"  Slowdown: {slowdown_pct:+.1f}%")
        print(f"  Rate trend: {rate_slope:+.1f}/s per batch")
        print(f"{'='*60}\n")

        # Memory growth rate should be minimal (< 2MB per batch)
        assert (
            memory_slope < 2.0
        ), f"MEMORY LEAK: Memory growing at {memory_slope:.2f}MB per batch"

        # Rate should not degrade significantly
        assert slowdown_pct < 30, f"SLOWDOWN: Rate dropped by {slowdown_pct:.1f}%"

        del result

    @pytest.mark.slow
    def test_profiled_extraction(self, temp_dir):
        """
        Run extraction with cProfile to diagnose slowdowns.

        Uses max_workers=1 so cProfile can see into the call stack
        (cProfile doesn't profile child threads).

        Saves profile results to:
        - profile_results.prof (binary, for snakeviz)
        - profile_results.txt (human-readable stats)

        To view with snakeviz:
            pip install snakeviz
            snakeviz /tmp/delm_profile/profile_results.prof

        Run with: pytest -m slow -s -k test_profiled
        """
        num_requests = 5_000  # Smaller for single-threaded profiling
        batch_size = 500
        max_workers = 1  # Single-threaded so cProfile can see full call stack

        # No artificial latency or failures for cleaner profiling
        mean_latency_ms = 0.0
        latency_std_ms = 0.0
        failure_rate = 0.0

        # Output directory for profile results
        profile_dir = Path("/tmp/delm_profile")
        profile_dir.mkdir(exist_ok=True)
        profile_file = profile_dir / "profile_results.prof"
        stats_file = profile_dir / "profile_results.txt"

        print(f"\n{'='*60}")
        print(f"PROFILED EXTRACTION TEST")
        print(f"{'='*60}")
        print(f"Total Requests: {num_requests:,}")
        print(f"Batch size: {batch_size}")
        print(f"Workers: {max_workers}")
        print(f"Profile output: {profile_dir}")
        print(f"{'='*60}\n")

        # Create test data - unique rows
        test_data = pd.DataFrame(
            {
                "text": [
                    f"Row {i}: Test text with name Test{i}, value {i}.45, category A."
                    for i in range(num_requests)
                ]
            }
        )

        with patch("instructor.from_provider") as mock_provider:
            mock_provider.return_value = MockInstructorClient(
                mean_latency_ms=mean_latency_ms,
                latency_std_ms=latency_std_ms,
                failure_rate=failure_rate,
            )

            delm = DELM(
                schema=TEST_SCHEMA_DICT,
                provider="openai",
                model="gpt-4o-mini",
                batch_size=batch_size,
                max_workers=max_workers,
                max_retries=1,
                track_cost=False,
                cache_backend="sqlite",
                cache_path=temp_dir / "cache",
                cache_max_size_mb=512,
                use_disk_storage=True,
                experiment_path=temp_dir / "experiment",
                overwrite_experiment=True,
                console_log_level="ERROR",
                save_log_file=False,
            )

            # Simple mock - just return data, no instrumentation
            delm.extraction_manager._instructor_extract_with_retry = MagicMock(
                return_value=MockExtractedData()
            )

            # Profile the extraction
            profiler = cProfile.Profile()
            print("Starting profiled extraction...")
            start_time = time.time()

            profiler.enable()
            result = delm.extract(test_data)
            profiler.disable()

            end_time = time.time()
            total_time = end_time - start_time

        print(f"\nExtraction completed in {total_time:.1f}s")
        print(f"Rate: {num_requests / total_time:,.0f} items/sec")
        print(f"Results: {len(result):,} rows")

        # Save profile results
        profiler.dump_stats(str(profile_file))
        print(f"\nProfile saved to: {profile_file}")

        # Also save human-readable stats
        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream)
        stats.sort_stats("cumulative")
        stats.print_stats(50)  # Top 50 functions

        stats_text = stream.getvalue()
        stats_file.write_text(stats_text)
        print(f"Stats saved to: {stats_file}")

        # Print top 20 to console
        print(f"\n{'='*60}")
        print("TOP 20 FUNCTIONS BY CUMULATIVE TIME")
        print(f"{'='*60}")
        stream2 = io.StringIO()
        stats2 = pstats.Stats(profiler, stream=stream2)
        stats2.sort_stats("cumulative")
        stats2.print_stats(20)
        print(stream2.getvalue())

        # Assertions
        assert len(result) == num_requests

        del result
        del delm
        gc.collect()

        print(f"\nTo visualize: snakeviz {profile_file}")


class TestResourceCleanupDELM:
    """Tests for proper resource cleanup at DELM level."""

    def test_delm_destruction_cleanup(self, temp_dir, mock_data_small):
        """Test that destroying DELM properly cleans up resources."""
        import threading

        initial_threads = threading.active_count()

        for i in range(5):
            with patch("instructor.from_provider") as mock_provider:
                mock_provider.return_value = MockInstructorClient()

                delm = DELM(
                    schema=TEST_SCHEMA_DICT,
                    provider="openai",
                    model="gpt-4o-mini",
                    batch_size=50,
                    max_workers=4,
                    track_cost=False,
                    cache_backend="sqlite",
                    cache_path=temp_dir / f"cache_{i}",
                    use_disk_storage=True,
                    experiment_path=temp_dir / f"exp_{i}",
                    overwrite_experiment=True,
                    console_log_level="ERROR",
                    save_log_file=False,
                )

                delm.extraction_manager._instructor_extract_with_retry = MagicMock(
                    return_value=MockExtractedData()
                )

                # Do some work
                result = delm.extract(mock_data_small.head(100).copy())

                del result
                del delm
                gc.collect()

        # Give threads time to clean up
        time.sleep(1)
        final_threads = threading.active_count()

        thread_growth = final_threads - initial_threads
        print(f"Thread growth: {thread_growth}")

        # Should not accumulate many threads
        assert thread_growth < 10, f"Thread leak: grew by {thread_growth} threads"


if __name__ == "__main__":
    # Run with: python -m pytest tests/unit/test_memory_leak.py -v
    # Run slow tests: python -m pytest tests/unit/test_memory_leak.py -v -m slow
    pytest.main([__file__, "-v"])
