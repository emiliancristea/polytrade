"""
Adverse Selection Filter Module.

Protects against trading when informed traders are likely active.
Key protections:
- News blackout periods (15-30 min before/after announcements)
- Whale trade detection (>$10K signals informed flow)
- Market timing (avoid initial volatility)
- Spread monitoring (wide spreads = uncertainty)

Research shows the most dangerous moments occur around news events
when informed traders suddenly appear.
"""

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Deque, Dict, List, Optional, Tuple
from loguru import logger

from src.config import config
from src.strategies.base import ArbitrageOpportunity


@dataclass
class LargeTrade:
    """Record of a detected large trade."""
    market_id: str
    side: str
    size: float
    price: float
    detected_at: datetime


@dataclass
class NewsEvent:
    """Scheduled news event for blackout."""
    name: str
    market_id: Optional[str]  # None = affects all markets
    scheduled_at: datetime
    blackout_minutes: int


class AdverseSelectionFilter:
    """
    Filters trading opportunities to avoid adverse selection.
    """

    def __init__(self):
        self.news_blackout_minutes = config.risk.news_blackout_minutes
        self.whale_threshold = config.risk.whale_trade_threshold
        self.wide_spread_threshold = config.risk.wide_spread_threshold

        # Tracked state
        self._recent_large_trades: Deque[LargeTrade] = deque(maxlen=1000)
        self._scheduled_events: List[NewsEvent] = []
        self._market_flow: Dict[str, Dict[str, float]] = {}  # market_id -> {buy: $, sell: $}

        # Blocked markets (temporarily)
        self._blocked_markets: Dict[str, datetime] = {}  # market_id -> blocked_until

    def should_trade(
        self,
        opportunity: ArbitrageOpportunity,
        current_spread: Optional[float] = None,
    ) -> Tuple[bool, str]:
        """
        Check if we should execute this opportunity.

        Args:
            opportunity: The opportunity to check
            current_spread: Current bid-ask spread if known

        Returns:
            Tuple of (should_trade, reason)
        """
        market_id = opportunity.market_id

        # Check if market is temporarily blocked
        if market_id in self._blocked_markets:
            blocked_until = self._blocked_markets[market_id]
            if datetime.now() < blocked_until:
                remaining = (blocked_until - datetime.now()).seconds
                return False, f"Market blocked for {remaining}s due to whale activity"
            else:
                del self._blocked_markets[market_id]

        # Check news blackout
        is_blackout, event_name = self._check_news_blackout(market_id)
        if is_blackout:
            return False, f"News blackout active: {event_name}"

        # Check for recent whale activity
        whale_detected, whale_info = self._check_whale_activity(market_id)
        if whale_detected:
            return False, f"Whale activity detected: {whale_info}"

        # Check spread (if provided)
        if current_spread is not None:
            if current_spread > self.wide_spread_threshold:
                # Don't skip, but note the wide spread
                logger.warning(f"Wide spread {current_spread:.2%} in {market_id}")
                # Could reduce position size here

        # Check informed flow ratio
        flow_imbalance = self._check_flow_imbalance(market_id)
        if flow_imbalance:
            return False, f"Significant flow imbalance detected"

        return True, "OK"

    def record_trade(
        self,
        market_id: str,
        side: str,
        size: float,
        price: float,
    ):
        """
        Record a trade for flow analysis.

        Args:
            market_id: Market identifier
            side: "BUY" or "SELL"
            size: Trade size in dollars
            price: Trade price
        """
        now = datetime.now()

        # Check if this is a whale trade
        if size >= self.whale_threshold:
            trade = LargeTrade(
                market_id=market_id,
                side=side,
                size=size,
                price=price,
                detected_at=now,
            )
            self._recent_large_trades.append(trade)

            # Block market temporarily
            self._blocked_markets[market_id] = now + timedelta(minutes=5)
            logger.warning(
                f"[WHALE] ${size:,.0f} {side} detected in {market_id[:20]}... "
                f"- blocking for 5 minutes"
            )

        # Update flow tracking
        if market_id not in self._market_flow:
            self._market_flow[market_id] = {"buy": 0.0, "sell": 0.0}

        self._market_flow[market_id][side.lower()] += size

    def add_news_event(
        self,
        name: str,
        scheduled_at: datetime,
        market_id: Optional[str] = None,
        blackout_minutes: Optional[int] = None,
    ):
        """
        Add a scheduled news event for blackout.

        Args:
            name: Event name (e.g., "Presidential Debate")
            scheduled_at: When the event occurs
            market_id: Specific market (None = all markets)
            blackout_minutes: Override default blackout
        """
        event = NewsEvent(
            name=name,
            market_id=market_id,
            scheduled_at=scheduled_at,
            blackout_minutes=blackout_minutes or self.news_blackout_minutes,
        )
        self._scheduled_events.append(event)
        logger.info(f"Added news event: {name} at {scheduled_at}")

    def _check_news_blackout(
        self, market_id: str
    ) -> Tuple[bool, Optional[str]]:
        """Check if currently in a news blackout period."""
        now = datetime.now()

        for event in self._scheduled_events:
            # Check if event applies to this market
            if event.market_id is not None and event.market_id != market_id:
                continue

            # Check blackout window
            start = event.scheduled_at - timedelta(minutes=event.blackout_minutes)
            end = event.scheduled_at + timedelta(minutes=event.blackout_minutes)

            if start <= now <= end:
                return True, event.name

        return False, None

    def _check_whale_activity(
        self, market_id: str
    ) -> Tuple[bool, Optional[str]]:
        """Check for recent whale activity in market."""
        now = datetime.now()
        cutoff = now - timedelta(minutes=5)

        recent_whales = [
            t for t in self._recent_large_trades
            if t.market_id == market_id and t.detected_at > cutoff
        ]

        if recent_whales:
            total_volume = sum(t.size for t in recent_whales)
            sides = set(t.side for t in recent_whales)

            if len(sides) == 1:
                # Unidirectional whale flow - very suspicious
                return True, f"${total_volume:,.0f} {recent_whales[0].side} in last 5 min"

        return False, None

    def _check_flow_imbalance(self, market_id: str) -> bool:
        """
        Check for significant flow imbalance.
        Returns True if imbalance suggests informed trading.
        """
        flow = self._market_flow.get(market_id)
        if not flow:
            return False

        buy_volume = flow.get("buy", 0)
        sell_volume = flow.get("sell", 0)
        total = buy_volume + sell_volume

        if total < 1000:  # Not enough data
            return False

        # Check imbalance ratio
        imbalance = abs(buy_volume - sell_volume) / total

        # >80% one-sided is suspicious
        if imbalance > 0.8 and total > 10000:
            logger.warning(
                f"Flow imbalance in {market_id}: "
                f"buy=${buy_volume:,.0f}, sell=${sell_volume:,.0f}"
            )
            return True

        return False

    def get_market_safety_score(self, market_id: str) -> float:
        """
        Get a safety score for a market (0-1, higher is safer).

        Args:
            market_id: Market to check

        Returns:
            Safety score 0-1
        """
        score = 1.0

        # Check blocked
        if market_id in self._blocked_markets:
            return 0.0

        # Check news blackout
        is_blackout, _ = self._check_news_blackout(market_id)
        if is_blackout:
            score *= 0.2

        # Check whale activity
        whale, _ = self._check_whale_activity(market_id)
        if whale:
            score *= 0.3

        # Check flow imbalance
        if self._check_flow_imbalance(market_id):
            score *= 0.5

        return score

    def clear_old_data(self, hours: int = 24):
        """Clear tracking data older than specified hours."""
        cutoff = datetime.now() - timedelta(hours=hours)

        # Clear old large trades
        while self._recent_large_trades and self._recent_large_trades[0].detected_at < cutoff:
            self._recent_large_trades.popleft()

        # Clear old events
        self._scheduled_events = [
            e for e in self._scheduled_events
            if e.scheduled_at + timedelta(minutes=e.blackout_minutes) > cutoff
        ]

        # Clear expired blocks
        now = datetime.now()
        self._blocked_markets = {
            k: v for k, v in self._blocked_markets.items()
            if v > now
        }

        # Reset flow tracking (or could decay it)
        self._market_flow.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get filter statistics."""
        now = datetime.now()
        return {
            "blocked_markets": len(self._blocked_markets),
            "scheduled_events": len(self._scheduled_events),
            "recent_whale_trades": len(self._recent_large_trades),
            "tracked_markets": len(self._market_flow),
            "active_blackouts": sum(
                1 for e in self._scheduled_events
                if (e.scheduled_at - timedelta(minutes=e.blackout_minutes)) <= now
                <= (e.scheduled_at + timedelta(minutes=e.blackout_minutes))
            ),
        }


# Common news events that affect prediction markets
COMMON_NEWS_EVENTS = [
    # US Politics
    "Presidential Debate",
    "State of the Union",
    "Primary Election Results",
    "General Election Results",
    "Supreme Court Decision",

    # Economics
    "Fed Rate Decision",
    "CPI Release",
    "Jobs Report",
    "GDP Report",

    # Sports
    "Super Bowl",
    "World Cup Final",
    "Championship Game",
]


def create_event_from_name(
    name: str,
    scheduled_at: datetime,
    market_id: Optional[str] = None,
) -> NewsEvent:
    """Helper to create a news event."""
    return NewsEvent(
        name=name,
        market_id=market_id,
        scheduled_at=scheduled_at,
        blackout_minutes=config.risk.news_blackout_minutes,
    )
