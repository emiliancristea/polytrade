"""
Rate Limiter Module.

Implements rate limiting based on documented Polymarket API limits:
- Order burst limit: 240 orders/second
- Price requests: 200 per 10 seconds
- WebSocket: 500 instruments per connection

Uses token bucket algorithm for smooth rate limiting.
"""

import asyncio
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, Optional
from loguru import logger

from src.config import config


@dataclass
class RateLimitConfig:
    """Configuration for a specific rate limit."""
    max_requests: int
    window_seconds: float
    name: str


class TokenBucket:
    """
    Token bucket rate limiter.
    Allows bursts up to bucket size, then refills at constant rate.
    """

    def __init__(
        self,
        rate: float,
        capacity: int,
        name: str = "default",
    ):
        """
        Initialize token bucket.

        Args:
            rate: Tokens added per second
            capacity: Maximum tokens (burst capacity)
            name: Name for logging
        """
        self.rate = rate
        self.capacity = capacity
        self.name = name
        self.tokens = float(capacity)
        self.last_update = time.monotonic()
        self._lock = asyncio.Lock()

    def _refill(self):
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self.last_update
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self.last_update = now

    async def acquire(self, tokens: int = 1) -> bool:
        """
        Acquire tokens, waiting if necessary.

        Args:
            tokens: Number of tokens to acquire

        Returns:
            True if acquired, False if would exceed capacity
        """
        if tokens > self.capacity:
            logger.warning(f"[{self.name}] Requested {tokens} tokens exceeds capacity {self.capacity}")
            return False

        async with self._lock:
            self._refill()

            if self.tokens >= tokens:
                self.tokens -= tokens
                return True

            # Calculate wait time
            needed = tokens - self.tokens
            wait_time = needed / self.rate

            logger.debug(f"[{self.name}] Rate limited, waiting {wait_time:.3f}s")
            await asyncio.sleep(wait_time)

            self._refill()
            self.tokens -= tokens
            return True

    def try_acquire(self, tokens: int = 1) -> bool:
        """
        Try to acquire tokens without waiting.

        Returns:
            True if acquired, False if not enough tokens
        """
        self._refill()

        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

    @property
    def available(self) -> float:
        """Get current available tokens."""
        self._refill()
        return self.tokens


class SlidingWindowLimiter:
    """
    Sliding window rate limiter.
    More accurate than fixed window, prevents burst at window boundaries.
    """

    def __init__(
        self,
        max_requests: int,
        window_seconds: float,
        name: str = "default",
    ):
        """
        Initialize sliding window limiter.

        Args:
            max_requests: Maximum requests in window
            window_seconds: Window duration in seconds
            name: Name for logging
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.name = name
        self.requests: deque = deque()
        self._lock = asyncio.Lock()

    def _cleanup(self):
        """Remove expired timestamps."""
        now = time.monotonic()
        cutoff = now - self.window_seconds

        while self.requests and self.requests[0] < cutoff:
            self.requests.popleft()

    async def acquire(self) -> bool:
        """
        Acquire a request slot, waiting if necessary.

        Returns:
            True when acquired
        """
        async with self._lock:
            self._cleanup()

            if len(self.requests) < self.max_requests:
                self.requests.append(time.monotonic())
                return True

            # Calculate wait time until oldest request expires
            oldest = self.requests[0]
            wait_time = (oldest + self.window_seconds) - time.monotonic()

            if wait_time > 0:
                logger.debug(f"[{self.name}] Rate limited, waiting {wait_time:.3f}s")
                await asyncio.sleep(wait_time)

            self._cleanup()
            self.requests.append(time.monotonic())
            return True

    def try_acquire(self) -> bool:
        """Try to acquire without waiting."""
        self._cleanup()

        if len(self.requests) < self.max_requests:
            self.requests.append(time.monotonic())
            return True
        return False

    @property
    def available(self) -> int:
        """Get available request slots."""
        self._cleanup()
        return self.max_requests - len(self.requests)


class PolymarketRateLimiter:
    """
    Composite rate limiter for all Polymarket API endpoints.
    Manages multiple rate limits for different endpoint types.
    """

    def __init__(self):
        """Initialize with Polymarket-specific rate limits."""
        # Order endpoint: 240 orders/second burst
        self.order_limiter = TokenBucket(
            rate=config.api.order_burst_limit,
            capacity=config.api.order_burst_limit,
            name="orders",
        )

        # Price/market data: 200 requests per 10 seconds
        self.price_limiter = SlidingWindowLimiter(
            max_requests=config.api.price_requests_per_10s,
            window_seconds=10.0,
            name="prices",
        )

        # General API calls (conservative)
        self.general_limiter = SlidingWindowLimiter(
            max_requests=100,
            window_seconds=10.0,
            name="general",
        )

        # WebSocket subscription tracking
        self._ws_subscriptions: Dict[str, int] = {}  # connection_id -> count

        # Stats
        self._stats = {
            "orders_limited": 0,
            "prices_limited": 0,
            "general_limited": 0,
        }

    async def acquire_order(self, count: int = 1) -> bool:
        """Acquire rate limit for order placement."""
        result = await self.order_limiter.acquire(count)
        if not result:
            self._stats["orders_limited"] += 1
        return result

    async def acquire_price(self) -> bool:
        """Acquire rate limit for price/market data request."""
        result = await self.price_limiter.acquire()
        if not self.price_limiter.try_acquire():
            self._stats["prices_limited"] += 1
        return result

    async def acquire_general(self) -> bool:
        """Acquire rate limit for general API call."""
        result = await self.general_limiter.acquire()
        if not result:
            self._stats["general_limited"] += 1
        return result

    def check_ws_subscription_limit(
        self,
        connection_id: str,
        new_subscriptions: int,
    ) -> bool:
        """
        Check if WebSocket subscription would exceed limit.

        Args:
            connection_id: Identifier for the WS connection
            new_subscriptions: Number of new instruments to subscribe

        Returns:
            True if within limits
        """
        current = self._ws_subscriptions.get(connection_id, 0)
        total = current + new_subscriptions

        if total > config.api.max_ws_instruments:
            logger.warning(
                f"WS subscription limit exceeded: {total} > {config.api.max_ws_instruments}"
            )
            return False

        return True

    def add_ws_subscriptions(self, connection_id: str, count: int):
        """Track WebSocket subscriptions."""
        current = self._ws_subscriptions.get(connection_id, 0)
        self._ws_subscriptions[connection_id] = current + count

    def remove_ws_subscriptions(self, connection_id: str, count: int):
        """Remove WebSocket subscription tracking."""
        current = self._ws_subscriptions.get(connection_id, 0)
        self._ws_subscriptions[connection_id] = max(0, current - count)

    def clear_ws_connection(self, connection_id: str):
        """Clear all subscriptions for a connection."""
        if connection_id in self._ws_subscriptions:
            del self._ws_subscriptions[connection_id]

    def get_stats(self) -> dict:
        """Get rate limiting statistics."""
        return {
            "order_tokens_available": self.order_limiter.available,
            "price_slots_available": self.price_limiter.available,
            "general_slots_available": self.general_limiter.available,
            "ws_connections": len(self._ws_subscriptions),
            "ws_total_subscriptions": sum(self._ws_subscriptions.values()),
            **self._stats,
        }


# Global rate limiter instance
rate_limiter = PolymarketRateLimiter()


# Decorator for rate-limited functions
def rate_limited(limiter_type: str = "general"):
    """
    Decorator to apply rate limiting to async functions.

    Args:
        limiter_type: "order", "price", or "general"
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            if limiter_type == "order":
                await rate_limiter.acquire_order()
            elif limiter_type == "price":
                await rate_limiter.acquire_price()
            else:
                await rate_limiter.acquire_general()

            return await func(*args, **kwargs)
        return wrapper
    return decorator
