"""Rate limiter interface for DELM.

Provides an abstract base class and concrete implementations for various
rate limiting strategies to be used via dependency injection in the ExtractionManager.
"""

import time
import threading
import logging
from typing import Optional, Callable

log = logging.getLogger(__name__)

from typing import Protocol


class RateLimiter(Protocol):
    def before_request(self, *, est_tokens: int) -> None:
        """
        Block until we’re allowed to send a request that is *estimated*
        to consume est_tokens (input + expected output).

        May sleep internally to respect RPM/TPM.
        """

    def after_request(self, *, actual_tokens: int) -> None:
        """
        Record the actual token usage (input + output) for bookkeeping.
        May be used to adjust buckets or statistics.
        """


class NoOpRateLimiter(RateLimiter):
    def before_request(self, *, est_tokens: int) -> None:
        pass

    def after_request(self, *, actual_tokens: int) -> None:
        pass


class BucketRateLimiter(RateLimiter):
    """
    Simple token-bucket rate limiter supporting both:
      - rate_limit_requests (requests per period)
      - rate_limit_tokens (tokens per period; total tokens: input + output)

    The period is configurable via ``period_seconds`` (defaults to 60s).

    Uses a token-bucket per resource, refilled continuously over time.
    Thread-safe and blocking: callers of `before_request` will sleep
    until there is enough capacity for (1 request, est_tokens tokens).

    Notes
    -----
    - `est_tokens` is used as an upper bound estimate; we *do not* correct
      with `actual_tokens` in the bucket, but you can log it in `after_request`.
    - If a limit is None, that dimension is treated as unlimited.
    """

    def __init__(
        self,
        *,
        rate_limit_requests: Optional[int] = None,
        rate_limit_tokens: Optional[int] = None,
        period_seconds: float = 60.0,
        time_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self._time_fn = time_fn
        self._period_seconds = period_seconds

        if self._period_seconds <= 0:
            raise ValueError(
                f"period_seconds must be positive, got {self._period_seconds}"
            )

        # Requests-per-period setup
        self._rp = rate_limit_requests
        if self._rp is None or self._rp <= 0:
            self._req_capacity = float("inf")
            self._req_rate_per_sec = 0.0
        else:
            self._req_capacity = float(self._rp)
            self._req_rate_per_sec = self._rp / self._period_seconds

        # Tokens-per-period setup
        self._tp = rate_limit_tokens
        if self._tp is None or self._tp <= 0:
            self._tok_capacity = float("inf")
            self._tok_rate_per_sec = 0.0
        else:
            self._tok_capacity = float(self._tp)
            self._tok_rate_per_sec = self._tp / self._period_seconds

        # Current bucket levels
        self._req_tokens = self._req_capacity
        self._tok_tokens = self._tok_capacity

        # Time bookkeeping
        self._last_refill = self._time_fn()

        # Concurrency primitives
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)

        # Simple stats (optional)
        self.total_requests = 0
        self.total_tokens = 0

    # ---------- Internal helpers ----------

    def _refill(self, now: float) -> None:
        """Refill both buckets based on elapsed time."""
        elapsed = now - self._last_refill
        if elapsed <= 0:
            return

        self._last_refill = now

        # Refill requests
        if self._req_rate_per_sec > 0.0 and self._req_tokens < self._req_capacity:
            self._req_tokens = min(
                self._req_capacity,
                self._req_tokens + elapsed * self._req_rate_per_sec,
            )

        # Refill tokens
        if self._tok_rate_per_sec > 0.0 and self._tok_tokens < self._tok_capacity:
            self._tok_tokens = min(
                self._tok_capacity,
                self._tok_tokens + elapsed * self._tok_rate_per_sec,
            )

    def _compute_wait_time(self, need_req: bool, need_tokens: float) -> float:
        """
        Compute how long (in seconds) we should wait until we have enough
        request- and/or token-capacity, based on current bucket levels and rates.
        """
        wait_for = float("inf")

        # Requests
        if need_req and self._req_rate_per_sec > 0.0:
            missing_req = 1.0 - self._req_tokens
            if missing_req > 0:
                wait_for = min(wait_for, missing_req / self._req_rate_per_sec)

        # Tokens
        if need_tokens > 0 and self._tok_rate_per_sec > 0.0:
            missing_tok = need_tokens - self._tok_tokens
            if missing_tok > 0:
                wait_for = min(wait_for, missing_tok / self._tok_rate_per_sec)

        if wait_for == float("inf"):
            # Should not really happen unless everything is unlimited, in which
            # case we wouldn't be here. Use a tiny fallback.
            wait_for = 0.01

        # Avoid 0-sleeps; still yield a tiny bit so time can advance
        return max(wait_for, 0.01)

    # ---------- Public interface ----------

    def before_request(self, *, est_tokens: int) -> None:
        # Quick fast-path: everything unlimited
        if self._req_capacity == float("inf") and self._tok_capacity == float("inf"):
            return

        est_tokens = max(0, est_tokens)

        # For oversized requests (est_tokens > capacity), we require a full
        # bucket before proceeding, then let the bucket go negative.  The
        # negative balance naturally delays subsequent requests.
        if self._tok_capacity != float("inf") and est_tokens > 0:
            tok_threshold = min(float(est_tokens), self._tok_capacity)
        else:
            tok_threshold = 0.0

        with self._cond:
            while True:
                now = self._time_fn()
                self._refill(now)

                need_req = self._req_capacity != float("inf")
                need_tok = self._tok_capacity != float("inf") and est_tokens > 0

                # Small epsilon avoids infinite loops from floating-point drift
                _EPS = 1e-9
                enough_req = (not need_req) or (self._req_tokens >= 1.0 - _EPS)
                enough_tok = (not need_tok) or (self._tok_tokens >= tok_threshold - _EPS)

                if enough_req and enough_tok:
                    if need_req:
                        self._req_tokens -= 1.0
                    if need_tok:
                        self._tok_tokens -= est_tokens

                    self.total_requests += 1
                    self.total_tokens += est_tokens
                    return

                wait_time = self._compute_wait_time(
                    need_req=not enough_req,
                    need_tokens=tok_threshold if not enough_tok else 0,
                )
                self._cond.wait(timeout=wait_time)

    def after_request(self, *, actual_tokens: int) -> None:
        # No-op; BucketRateLimiter uses only est_tokens in before_request.
        pass
