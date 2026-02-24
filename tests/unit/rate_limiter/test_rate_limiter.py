"""
Unit tests for the BucketRateLimiter with configurable period_seconds.
"""

import pytest
import time

from delm.utils.rate_limiter import BucketRateLimiter, NoOpRateLimiter


class TestBucketRateLimiterPeriod:
    """Test that the rate limiter respects custom period_seconds."""

    def test_default_period_is_60(self):
        limiter = BucketRateLimiter(rate_limit_requests=10)
        assert limiter._period_seconds == 60.0

    def test_custom_period_seconds(self):
        limiter = BucketRateLimiter(
            rate_limit_requests=10,
            period_seconds=2.0,
        )
        assert limiter._period_seconds == 2.0

    def test_invalid_period_raises(self):
        with pytest.raises(ValueError, match="period_seconds must be positive"):
            BucketRateLimiter(rate_limit_requests=10, period_seconds=0)

        with pytest.raises(ValueError, match="period_seconds must be positive"):
            BucketRateLimiter(rate_limit_requests=10, period_seconds=-5)

    def test_refill_rate_uses_custom_period(self):
        limiter = BucketRateLimiter(
            rate_limit_requests=100,
            rate_limit_tokens=1000,
            period_seconds=2.0,
        )
        # 100 requests per 2 seconds = 50 requests/sec
        assert limiter._req_rate_per_sec == pytest.approx(50.0)
        # 1000 tokens per 2 seconds = 500 tokens/sec
        assert limiter._tok_rate_per_sec == pytest.approx(500.0)

    def test_refill_rate_with_default_60s(self):
        limiter = BucketRateLimiter(
            rate_limit_requests=120,
            rate_limit_tokens=6000,
            period_seconds=60.0,
        )
        # 120 requests per 60 seconds = 2 requests/sec
        assert limiter._req_rate_per_sec == pytest.approx(2.0)
        # 6000 tokens per 60 seconds = 100 tokens/sec
        assert limiter._tok_rate_per_sec == pytest.approx(100.0)

    def test_capacity_matches_limit(self):
        limiter = BucketRateLimiter(
            rate_limit_requests=5,
            rate_limit_tokens=500,
            period_seconds=10.0,
        )
        assert limiter._req_capacity == 5.0
        assert limiter._tok_capacity == 500.0


class TestBucketRateLimiterBehavior:
    """Test basic rate limiter behavior with custom periods."""

    def test_requests_consumed_from_bucket(self):
        limiter = BucketRateLimiter(
            rate_limit_requests=3,
            period_seconds=60.0,
        )
        limiter.before_request(est_tokens=0)
        limiter.before_request(est_tokens=0)
        limiter.before_request(est_tokens=0)
        assert limiter.total_requests == 3

    def test_tokens_consumed_from_bucket(self):
        limiter = BucketRateLimiter(
            rate_limit_tokens=1000,
            period_seconds=60.0,
        )
        limiter.before_request(est_tokens=200)
        limiter.before_request(est_tokens=300)
        assert limiter.total_tokens == 500

    def test_unlimited_when_none(self):
        limiter = BucketRateLimiter(
            rate_limit_requests=None,
            rate_limit_tokens=None,
            period_seconds=5.0,
        )
        assert limiter._req_capacity == float("inf")
        assert limiter._tok_capacity == float("inf")
        limiter.before_request(est_tokens=999999)
        assert limiter.total_requests == 0

    def test_short_period_blocking(self):
        """With a very short period and low limit, requests should block after exhausting capacity."""
        fake_time = [0.0]

        def time_fn():
            return fake_time[0]

        limiter = BucketRateLimiter(
            rate_limit_requests=2,
            period_seconds=1.0,
            time_fn=time_fn,
        )

        limiter.before_request(est_tokens=0)
        limiter.before_request(est_tokens=0)
        assert limiter.total_requests == 2

        # Advance time by the full period to refill
        fake_time[0] = 1.0
        limiter.before_request(est_tokens=0)
        assert limiter.total_requests == 3

    def test_fractional_period_seconds(self):
        limiter = BucketRateLimiter(
            rate_limit_tokens=100,
            period_seconds=0.5,
        )
        # 100 tokens per 0.5 seconds = 200 tokens/sec
        assert limiter._tok_rate_per_sec == pytest.approx(200.0)

    def test_large_period_seconds(self):
        limiter = BucketRateLimiter(
            rate_limit_requests=1000,
            period_seconds=3600.0,
        )
        # 1000 requests per hour
        assert limiter._req_rate_per_sec == pytest.approx(1000.0 / 3600.0)


class TestOversizedRequestBehavior:
    """Test that requests exceeding bucket capacity are handled correctly.

    When est_tokens > capacity the limiter must:
      1. Wait until the bucket is fully refilled (at capacity).
      2. Deduct the full est_tokens, driving the bucket negative.
      3. Subsequent requests block until the negative balance refills.

    All tests use only the public ``before_request`` interface with a fake
    clock to avoid real-time sleeps.
    """

    def test_oversized_request_allowed_when_bucket_full(self):
        """A 99k request with 33k capacity should proceed immediately on a full bucket."""
        fake_time = [0.0]
        limiter = BucketRateLimiter(
            rate_limit_tokens=33_000,
            period_seconds=1.0,
            time_fn=lambda: fake_time[0],
        )

        limiter.before_request(est_tokens=99_000)
        assert limiter.total_tokens == 99_000
        assert limiter._tok_tokens == pytest.approx(-66_000.0)

    def test_oversized_request_blocks_until_full(self):
        """An oversized request should wait until the bucket reaches full capacity."""
        fake_time = [0.0]
        limiter = BucketRateLimiter(
            rate_limit_tokens=33_000,
            period_seconds=1.0,
            time_fn=lambda: fake_time[0],
        )

        limiter.before_request(est_tokens=20_000)
        assert limiter._tok_tokens == pytest.approx(13_000.0)

        fake_time[0] = 1.0
        limiter.before_request(est_tokens=99_000)
        assert limiter.total_tokens == 20_000 + 99_000

    def test_oversized_request_exact_scenario(self):
        """33k tokens/sec, 99k request → goes through immediately,
        then must wait ~3s for the next full-capacity request."""
        fake_time = [0.0]
        limiter = BucketRateLimiter(
            rate_limit_tokens=33_000,
            period_seconds=1.0,
            time_fn=lambda: fake_time[0],
        )

        # t=0: Submit 99k on full bucket (33k) → bucket = -66k
        limiter.before_request(est_tokens=99_000)
        assert limiter._tok_tokens == pytest.approx(-66_000.0)

        # t=3: bucket has refilled: -66k + 3*33k = 33k. Another 99k can go.
        fake_time[0] = 3.0
        limiter.before_request(est_tokens=99_000)
        assert limiter.total_tokens == 99_000 * 2
        assert limiter._tok_tokens == pytest.approx(-66_000.0)

    def test_negative_bucket_delays_next_normal_request(self):
        """After an oversized request drives the bucket negative, even a
        small follow-up must wait for the bucket to recover above zero."""
        fake_time = [0.0]
        limiter = BucketRateLimiter(
            rate_limit_tokens=33_000,
            period_seconds=1.0,
            time_fn=lambda: fake_time[0],
        )

        limiter.before_request(est_tokens=99_000)
        assert limiter._tok_tokens == pytest.approx(-66_000.0)

        # t=3: bucket = -66k + 3*33k = 33k. 1k request can go.
        fake_time[0] = 3.0
        limiter.before_request(est_tokens=1_000)
        assert limiter.total_tokens == 99_000 + 1_000
        assert limiter._tok_tokens == pytest.approx(32_000.0)

    def test_small_request_after_oversized_with_exact_recovery(self):
        """A tiny request goes through as soon as the bucket has recovered
        enough tokens (above zero)."""
        fake_time = [0.0]
        limiter = BucketRateLimiter(
            rate_limit_tokens=100,
            period_seconds=1.0,
            time_fn=lambda: fake_time[0],
        )

        # 300 tokens on a 100-capacity bucket → bucket = -200
        limiter.before_request(est_tokens=300)
        assert limiter._tok_tokens == pytest.approx(-200.0)

        # t=2.1: bucket = -200 + 2.1*100 = 10. Enough for est_tokens=1.
        fake_time[0] = 2.1
        limiter.before_request(est_tokens=1)
        assert limiter.total_tokens == 301

    def test_bucket_state_after_refill_from_negative(self):
        """Verify internal bucket value after refilling from a negative state."""
        fake_time = [0.0]
        limiter = BucketRateLimiter(
            rate_limit_tokens=100,
            period_seconds=1.0,
            time_fn=lambda: fake_time[0],
        )

        limiter.before_request(est_tokens=250)
        assert limiter._tok_tokens == pytest.approx(-150.0)

        # t=3: bucket = -150 + 3*100 = 150, capped at capacity 100.
        fake_time[0] = 3.0
        limiter.before_request(est_tokens=50)
        # After refill to 100, then deduct 50 → 50
        assert limiter._tok_tokens == pytest.approx(50.0)


class TestNoOpRateLimiter:
    """Test NoOpRateLimiter is a no-op."""

    def test_before_request_does_nothing(self):
        limiter = NoOpRateLimiter()
        limiter.before_request(est_tokens=999999)

    def test_after_request_does_nothing(self):
        limiter = NoOpRateLimiter()
        limiter.after_request(actual_tokens=999999)
