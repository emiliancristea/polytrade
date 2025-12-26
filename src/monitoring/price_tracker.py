"""
Price Tracker for real-time arbitrage detection.
Combines WebSocket updates with arbitrage logic.
"""

import asyncio
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from loguru import logger

from src.config import config
from src.strategies.base import ArbitrageOpportunity, RiskLevel, StrategyType
from .websocket_monitor import WebSocketMonitor


class MarketPair:
    """Represents a binary market pair for tracking."""

    def __init__(
        self,
        market_id: str,
        question: str,
        yes_token: str,
        no_token: str,
    ):
        self.market_id = market_id
        self.question = question
        self.yes_token = yes_token
        self.no_token = no_token
        self.yes_price: Optional[float] = None
        self.no_price: Optional[float] = None
        self.last_update: Optional[datetime] = None

    @property
    def total_cost(self) -> Optional[float]:
        """Total cost to buy both outcomes."""
        if self.yes_price is not None and self.no_price is not None:
            return self.yes_price + self.no_price
        return None

    @property
    def profit_margin(self) -> Optional[float]:
        """Profit margin (1 - total_cost)."""
        cost = self.total_cost
        if cost is not None:
            return 1.0 - cost
        return None

    @property
    def is_profitable(self) -> bool:
        """Check if this pair offers profit."""
        margin = self.profit_margin
        return margin is not None and margin > config.trading.min_profit_margin

    def update_price(self, token_id: str, price: float):
        """Update price for a token."""
        if token_id == self.yes_token:
            self.yes_price = price
        elif token_id == self.no_token:
            self.no_price = price
        self.last_update = datetime.now()


class PriceTracker:
    """
    Tracks prices and detects arbitrage in real-time.
    Uses WebSocket for sub-second updates.
    """

    def __init__(self, on_opportunity: Optional[Callable] = None):
        """
        Initialize price tracker.

        Args:
            on_opportunity: Callback when opportunity detected
        """
        self.ws_monitor = WebSocketMonitor()
        self.on_opportunity = on_opportunity

        # Tracking state
        self._market_pairs: Dict[str, MarketPair] = {}
        self._token_to_market: Dict[str, str] = {}  # token_id -> market_id
        self._active_opportunities: Dict[str, ArbitrageOpportunity] = {}

        # Stats
        self._opportunities_detected = 0
        self._price_updates_processed = 0
        self._start_time: Optional[datetime] = None

    async def start(self):
        """Start the price tracker."""
        self._start_time = datetime.now()
        logger.info("Starting price tracker...")

        # Connect WebSocket
        connected = await self.ws_monitor.connect()
        if not connected:
            logger.error("Failed to connect WebSocket")
            return

        # Add our price handler
        self.ws_monitor.add_callback(self._on_price_update)

        # Start WebSocket event loop
        await self.ws_monitor.run()

    async def stop(self):
        """Stop the price tracker."""
        await self.ws_monitor.disconnect()
        logger.info(
            f"Price tracker stopped. "
            f"Detected {self._opportunities_detected} opportunities "
            f"from {self._price_updates_processed} price updates."
        )

    async def add_markets(self, markets: List[Dict[str, Any]]):
        """
        Add markets to track.

        Args:
            markets: List of market data from Gamma API
        """
        tokens_to_subscribe = []

        for market in markets:
            outcomes = market.get("outcomes", [])
            if len(outcomes) != 2:
                continue

            tokens = self._extract_tokens(market)
            if len(tokens) != 2:
                continue

            market_id = market.get("conditionId") or market.get("id", "")
            question = market.get("question", "Unknown")

            # Create market pair
            pair = MarketPair(
                market_id=market_id,
                question=question,
                yes_token=tokens[0],
                no_token=tokens[1],
            )

            self._market_pairs[market_id] = pair
            self._token_to_market[tokens[0]] = market_id
            self._token_to_market[tokens[1]] = market_id
            tokens_to_subscribe.extend(tokens)

        # Subscribe to all tokens
        if tokens_to_subscribe:
            await self.ws_monitor.subscribe(tokens_to_subscribe)
            logger.info(f"Added {len(self._market_pairs)} markets, tracking {len(tokens_to_subscribe)} tokens")

    def _extract_tokens(self, market: Dict[str, Any]) -> List[str]:
        """Extract token IDs from market."""
        tokens = market.get("clobTokenIds", [])
        if isinstance(tokens, str):
            tokens = tokens.strip("[]").replace('"', "").split(",")
            tokens = [t.strip() for t in tokens if t.strip()]
        return tokens

    async def _on_price_update(self, data: Dict[str, Any]):
        """Handle incoming price update."""
        self._price_updates_processed += 1

        token_id = data.get("token_id")
        if not token_id:
            return

        market_id = self._token_to_market.get(token_id)
        if not market_id:
            return

        pair = self._market_pairs.get(market_id)
        if not pair:
            return

        # Update price
        price = data.get("price") or data.get("best_ask")
        if price is not None:
            pair.update_price(token_id, float(price))

        # Check for arbitrage
        if pair.is_profitable:
            opportunity = self._create_opportunity(pair)
            if opportunity:
                await self._handle_opportunity(opportunity)

    def _create_opportunity(self, pair: MarketPair) -> Optional[ArbitrageOpportunity]:
        """Create arbitrage opportunity from market pair."""
        if pair.profit_margin is None or pair.profit_margin < config.trading.min_profit_margin:
            return None

        import uuid
        investment = 100.0
        shares = investment / pair.total_cost
        profit = shares * pair.profit_margin

        return ArbitrageOpportunity(
            id=f"live-{uuid.uuid4().hex[:8]}",
            strategy_type=StrategyType.BINARY,
            market_id=pair.market_id,
            market_question=pair.question,
            profit_margin=pair.profit_margin,
            profit_amount=profit,
            investment_required=investment,
            expected_return=investment + profit,
            trades=[
                {
                    "action": "BUY",
                    "outcome": "Yes",
                    "token_id": pair.yes_token,
                    "price": pair.yes_price,
                    "side": "BUY",
                },
                {
                    "action": "BUY",
                    "outcome": "No",
                    "token_id": pair.no_token,
                    "price": pair.no_price,
                    "side": "BUY",
                },
            ],
            token_ids=[pair.yes_token, pair.no_token],
            prices={"Yes": pair.yes_price, "No": pair.no_price},
            risk_level=RiskLevel.LOW,
            confidence=0.95,
            metadata={
                "detected_via": "websocket",
                "total_cost": pair.total_cost,
            },
        )

    async def _handle_opportunity(self, opportunity: ArbitrageOpportunity):
        """Handle detected opportunity."""
        # Check if we already have this opportunity active
        if opportunity.market_id in self._active_opportunities:
            existing = self._active_opportunities[opportunity.market_id]
            # Only update if profit improved
            if opportunity.profit_margin <= existing.profit_margin:
                return

        self._active_opportunities[opportunity.market_id] = opportunity
        self._opportunities_detected += 1

        logger.info(
            f"[LIVE ARB] {opportunity.market_question[:50]}... "
            f"| Profit: {opportunity.profit_margin * 100:.3f}% "
            f"(${opportunity.profit_amount:.4f})"
        )

        # Trigger callback
        if self.on_opportunity:
            try:
                if asyncio.iscoroutinefunction(self.on_opportunity):
                    await self.on_opportunity(opportunity)
                else:
                    self.on_opportunity(opportunity)
            except Exception as e:
                logger.error(f"Opportunity callback error: {e}")

    def get_active_opportunities(self) -> List[ArbitrageOpportunity]:
        """Get currently active opportunities."""
        # Clean up stale opportunities (older than 30 seconds)
        cutoff = datetime.now() - timedelta(seconds=30)
        stale = [
            mid for mid, opp in self._active_opportunities.items()
            if opp.detected_at < cutoff
        ]
        for mid in stale:
            del self._active_opportunities[mid]

        return list(self._active_opportunities.values())

    def get_stats(self) -> Dict[str, Any]:
        """Get tracker statistics."""
        uptime = (datetime.now() - self._start_time).total_seconds() if self._start_time else 0
        return {
            "uptime_seconds": uptime,
            "markets_tracked": len(self._market_pairs),
            "tokens_subscribed": self.ws_monitor.subscribed_count,
            "price_updates": self._price_updates_processed,
            "opportunities_detected": self._opportunities_detected,
            "active_opportunities": len(self._active_opportunities),
            "ws_connected": self.ws_monitor.is_connected,
        }
