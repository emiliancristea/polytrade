"""
Real-time Arbitrage Scanner for Polymarket.

This module subscribes to WebSocket price feeds and detects arbitrage
opportunities the INSTANT prices change - before REST API updates.

Key insight: Arbitrage gaps exist for milliseconds. This scanner:
1. Subscribes to all active market tokens via WebSocket
2. Maintains a real-time price cache for YES/NO pairs
3. Detects gaps the moment ANY price changes
4. Triggers immediate execution

This is the key differentiator for profitable arbitrage.
"""

import asyncio
import json
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set
from loguru import logger

from src.config import config
from src.monitoring.websocket_monitor import WebSocketMonitor


class RealtimeArbitrageScanner:
    """
    Millisecond-level arbitrage detection using WebSocket feeds.

    Architecture:
    - Maintains price cache for all YES/NO token pairs
    - On ANY price update, immediately checks for arbitrage
    - Supports callback for instant trade execution
    """

    def __init__(self):
        self._ws_monitor = WebSocketMonitor(ws_type="clob")

        # Market structure: condition_id -> {yes_token, no_token, yes_price, no_price, ...}
        self._markets: Dict[str, Dict[str, Any]] = {}

        # Token ID -> condition_id mapping for reverse lookup
        self._token_to_condition: Dict[str, str] = {}

        # Execution callback - called when opportunity found
        self._execution_callback: Optional[Callable] = None

        # Stats
        self._opportunities_found = 0
        self._price_updates_processed = 0
        self._scans_performed = 0

        # Last opportunity details
        self._last_opportunities: List[Dict[str, Any]] = []

        # Running flag
        self._running = False

    def set_execution_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """
        Set callback for when arbitrage opportunity is detected.

        The callback receives opportunity dict with:
        - condition_id: Market identifier
        - question: Market question
        - yes_price: Current YES price
        - no_price: Current NO price
        - total: YES + NO total
        - profit_margin: 1.0 - total (the arbitrage gap)
        - timestamp: Detection time
        """
        self._execution_callback = callback

    async def load_markets(self, markets: List[Dict[str, Any]]):
        """
        Load markets and subscribe to their token price feeds.

        Args:
            markets: List of market dicts from Gamma API with token info
        """
        token_ids = []

        for market in markets:
            condition_id = market.get("conditionId")
            if not condition_id:
                continue

            # Parse clobTokenIds - this is where YES/NO token IDs are stored
            clob_token_ids = market.get("clobTokenIds")
            if not clob_token_ids:
                continue

            # Parse if it's a JSON string
            if isinstance(clob_token_ids, str):
                try:
                    clob_token_ids = json.loads(clob_token_ids)
                except json.JSONDecodeError:
                    continue

            if not isinstance(clob_token_ids, list) or len(clob_token_ids) < 2:
                continue

            # clobTokenIds[0] = YES, clobTokenIds[1] = NO
            yes_token = str(clob_token_ids[0])
            no_token = str(clob_token_ids[1])

            if not yes_token or not no_token:
                continue

            # Parse current prices from outcomePrices
            outcome_prices = market.get("outcomePrices", "")
            if isinstance(outcome_prices, str):
                try:
                    prices = json.loads(outcome_prices)
                    yes_price = float(prices[0]) if len(prices) > 0 else 0.5
                    no_price = float(prices[1]) if len(prices) > 1 else 0.5
                except (json.JSONDecodeError, ValueError, IndexError):
                    yes_price = 0.5
                    no_price = 0.5
            elif isinstance(outcome_prices, list):
                yes_price = float(outcome_prices[0]) if len(outcome_prices) > 0 else 0.5
                no_price = float(outcome_prices[1]) if len(outcome_prices) > 1 else 0.5
            else:
                yes_price = 0.5
                no_price = 0.5

            # Store market structure
            self._markets[condition_id] = {
                "condition_id": condition_id,
                "question": market.get("question", "Unknown")[:80],
                "yes_token": yes_token,
                "no_token": no_token,
                "yes_price": yes_price,
                "no_price": no_price,
                "volume": float(market.get("volume", 0) or 0),
                "liquidity": float(market.get("liquidity", 0) or 0),
                "last_update": datetime.now(),
            }

            # Reverse lookup
            self._token_to_condition[yes_token] = condition_id
            self._token_to_condition[no_token] = condition_id

            token_ids.extend([yes_token, no_token])

        logger.info(f"Loaded {len(self._markets)} markets with {len(token_ids)} tokens")

        # Subscribe to WebSocket feeds (max 500 per connection)
        # Prioritize by volume/liquidity
        sorted_markets = sorted(
            self._markets.values(),
            key=lambda m: m["volume"],
            reverse=True
        )

        priority_tokens = []
        for market in sorted_markets[:250]:  # Top 250 markets = 500 tokens
            priority_tokens.append(market["yes_token"])
            priority_tokens.append(market["no_token"])

        if priority_tokens:
            # Connect and subscribe
            await self._ws_monitor.connect()
            self._ws_monitor.add_callback(self._on_price_update)
            await self._ws_monitor.subscribe(priority_tokens)

            logger.info(f"Subscribed to {len(priority_tokens)} tokens via WebSocket")

    async def _on_price_update(self, data: Dict[str, Any]):
        """
        Handle real-time price update from WebSocket.

        This is the hot path - must be extremely fast.

        For arbitrage detection, we use the BEST ASK price because:
        - To buy YES, we pay the best ask price
        - To buy NO, we pay the best ask price
        - If YES_ask + NO_ask < $1.00, we have arbitrage
        """
        self._price_updates_processed += 1

        token_id = data.get("token_id")
        if not token_id:
            return

        # Convert to string for consistent lookup
        token_id = str(token_id)

        if token_id not in self._token_to_condition:
            return

        condition_id = self._token_to_condition[token_id]
        market = self._markets.get(condition_id)
        if not market:
            return

        # Get the best ask price (what we'd pay to buy this outcome)
        # For arbitrage, we care about ask prices since we're buying
        best_ask = data.get("best_ask")
        if best_ask is None:
            # Try mid price as fallback
            best_ask = data.get("price")
        if best_ask is None:
            return

        new_price = float(best_ask)

        # Determine if YES or NO token and update
        if token_id == market["yes_token"]:
            old_price = market["yes_price"]
            if abs(new_price - old_price) > 0.0001:  # Only log significant changes
                logger.debug(f"YES ask update: {old_price:.4f} -> {new_price:.4f}")
            market["yes_price"] = new_price
        elif token_id == market["no_token"]:
            old_price = market["no_price"]
            if abs(new_price - old_price) > 0.0001:
                logger.debug(f"NO ask update: {old_price:.4f} -> {new_price:.4f}")
            market["no_price"] = new_price
        else:
            return

        market["last_update"] = datetime.now()

        # IMMEDIATELY check for arbitrage
        await self._check_arbitrage(market)

    async def _check_arbitrage(self, market: Dict[str, Any]):
        """
        Check single market for arbitrage opportunity.

        Called on EVERY price update - must be fast.
        """
        self._scans_performed += 1

        yes_price = market["yes_price"]
        no_price = market["no_price"]
        total = yes_price + no_price

        # Calculate profit margin
        profit_margin = 1.0 - total

        # Check if opportunity exists
        if profit_margin > config.trading.min_profit_margin:
            self._opportunities_found += 1

            opportunity = {
                "condition_id": market["condition_id"],
                "question": market["question"],
                "yes_price": yes_price,
                "no_price": no_price,
                "total": total,
                "profit_margin": profit_margin,
                "profit_percent": profit_margin * 100,
                "volume": market["volume"],
                "liquidity": market["liquidity"],
                "timestamp": datetime.now(),
                "detection_latency_ms": 0,  # TODO: measure actual latency
            }

            # Store for reporting
            self._last_opportunities.append(opportunity)
            if len(self._last_opportunities) > 100:
                self._last_opportunities = self._last_opportunities[-100:]

            # Log the opportunity
            logger.success(
                f"🎯 ARBITRAGE DETECTED! "
                f"YES={yes_price:.4f} + NO={no_price:.4f} = {total:.4f} "
                f"PROFIT: {profit_margin*100:.3f}% | {market['question'][:50]}"
            )

            # Trigger execution callback
            if self._execution_callback:
                try:
                    if asyncio.iscoroutinefunction(self._execution_callback):
                        await self._execution_callback(opportunity)
                    else:
                        self._execution_callback(opportunity)
                except Exception as e:
                    logger.error(f"Execution callback error: {e}")

    async def run(self):
        """
        Main loop - runs WebSocket monitor and processes updates.
        """
        self._running = True

        logger.info("Starting real-time arbitrage scanner...")
        logger.info(f"Monitoring {len(self._markets)} markets")
        logger.info(f"Min profit threshold: {config.trading.min_profit_margin * 100:.2f}%")

        # Start WebSocket event loop
        try:
            await self._ws_monitor.run()
        except asyncio.CancelledError:
            pass
        finally:
            self._running = False
            await self._ws_monitor.disconnect()

    async def stop(self):
        """Stop the scanner."""
        self._running = False
        await self._ws_monitor.disconnect()

    def get_stats(self) -> Dict[str, Any]:
        """Get scanner statistics."""
        return {
            "markets_loaded": len(self._markets),
            "tokens_subscribed": self._ws_monitor.subscribed_count,
            "ws_connected": self._ws_monitor.is_connected,
            "price_updates_processed": self._price_updates_processed,
            "scans_performed": self._scans_performed,
            "opportunities_found": self._opportunities_found,
            "last_opportunities": self._last_opportunities[-10:],
            "ws_stats": self._ws_monitor.get_stats(),
        }

    def do_full_scan(self) -> List[Dict[str, Any]]:
        """
        Perform full scan of all cached markets for arbitrage.

        This is a synchronous scan of cached prices - useful for
        periodic checks or debugging.

        Returns:
            List of arbitrage opportunities
        """
        opportunities = []

        for condition_id, market in self._markets.items():
            yes_price = market["yes_price"]
            no_price = market["no_price"]
            total = yes_price + no_price
            profit_margin = 1.0 - total

            if profit_margin > config.trading.min_profit_margin:
                opportunities.append({
                    "condition_id": condition_id,
                    "question": market["question"],
                    "yes_price": yes_price,
                    "no_price": no_price,
                    "total": total,
                    "profit_margin": profit_margin,
                    "profit_percent": profit_margin * 100,
                    "volume": market["volume"],
                    "liquidity": market["liquidity"],
                })

        return sorted(opportunities, key=lambda x: x["profit_margin"], reverse=True)


async def run_realtime_scanner():
    """
    Main entry point for real-time arbitrage scanning.
    """
    from src.api.gamma_api import GammaAPI

    logger.info("=" * 60)
    logger.info("REAL-TIME ARBITRAGE SCANNER")
    logger.info("=" * 60)
    logger.info("")
    logger.info("This scanner uses WebSocket feeds to detect arbitrage")
    logger.info("opportunities the INSTANT prices change.")
    logger.info("")

    # Load markets from API
    gamma = GammaAPI()
    logger.info("Fetching active markets...")

    markets = await gamma.get_markets(
        active=True,
        closed=False,
        limit=500,  # Get top 500 markets
    )

    if not markets:
        logger.error("No markets found!")
        return

    logger.info(f"Found {len(markets)} active markets")

    # Initialize scanner
    scanner = RealtimeArbitrageScanner()

    # Set up execution callback (dry run logging for now)
    async def on_opportunity(opp: Dict[str, Any]):
        logger.info(f"Would execute trade: {opp['profit_percent']:.3f}% profit")
        if config.trading.dry_run:
            logger.info("[DRY RUN] Trade not executed")
        # TODO: Integrate with OrderExecutor for live trading

    scanner.set_execution_callback(on_opportunity)

    # Load markets and subscribe to feeds
    await scanner.load_markets(markets)

    # Do initial scan of cached prices
    initial_opps = scanner.do_full_scan()
    if initial_opps:
        logger.success(f"Initial scan found {len(initial_opps)} opportunities!")
        for opp in initial_opps[:5]:
            logger.info(f"  {opp['profit_percent']:.3f}%: {opp['question']}")
    else:
        logger.info("No opportunities in initial scan - waiting for price changes...")

    # Run the scanner
    logger.info("")
    logger.info("Starting WebSocket price feed... Press Ctrl+C to stop")
    logger.info("")

    try:
        await scanner.run()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        await scanner.stop()

        # Print final stats
        stats = scanner.get_stats()
        logger.info("")
        logger.info("=" * 60)
        logger.info("SESSION STATISTICS")
        logger.info("=" * 60)
        logger.info(f"Price updates processed: {stats['price_updates_processed']}")
        logger.info(f"Scans performed: {stats['scans_performed']}")
        logger.info(f"Opportunities found: {stats['opportunities_found']}")
        logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_realtime_scanner())
