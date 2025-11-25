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
      - requests_per_minute (RPM)
      - tokens_per_minute (TPM; total tokens: input + output)

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
        requests_per_minute: Optional[int] = None,
        tokens_per_minute: Optional[int] = None,
        time_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self._time_fn = time_fn

        # RPM setup
        self._rpm = requests_per_minute
        if self._rpm is None or self._rpm <= 0:
            self._req_capacity = float("inf")
            self._req_rate_per_sec = 0.0  # "unlimited"
        else:
            self._req_capacity = float(self._rpm)
            self._req_rate_per_sec = self._rpm / 60.0

        # TPM setup
        self._tpm = tokens_per_minute
        if self._tpm is None or self._tpm <= 0:
            self._tok_capacity = float("inf")
            self._tok_rate_per_sec = 0.0  # "unlimited"
        else:
            self._tok_capacity = float(self._tpm)
            self._tok_rate_per_sec = self._tpm / 60.0

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

        # Clamp estimate to capacity to avoid "impossible" waits
        if self._tok_capacity != float("inf"):
            est_tokens = min(est_tokens, int(self._tok_capacity))

        with self._cond:
            while True:
                now = self._time_fn()
                self._refill(now)

                need_req = self._req_capacity != float("inf")
                need_tok = self._tok_capacity != float("inf") and est_tokens > 0

                # Do we have enough for this call?
                enough_req = (not need_req) or (self._req_tokens >= 1.0)
                enough_tok = (not need_tok) or (self._tok_tokens >= est_tokens)

                if enough_req and enough_tok:
                    # Consume from buckets and proceed
                    if need_req:
                        self._req_tokens -= 1.0
                    if need_tok:
                        self._tok_tokens -= est_tokens

                    self.total_requests += 1
                    self.total_tokens += est_tokens
                    return

                # Not enough capacity yet — compute how long to wait
                wait_time = self._compute_wait_time(
                    need_req=not enough_req,
                    need_tokens=est_tokens if not enough_tok else 0,
                )
                self._cond.wait(timeout=wait_time)

    def after_request(self, *, actual_tokens: int) -> None:
        # No-op; BucketRateLimiter uses only est_tokens in before_request.
        pass
