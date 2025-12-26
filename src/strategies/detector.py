"""
Arbitrage Detector - Combines all strategies to find opportunities.
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional
from loguru import logger

from .base import ArbitrageOpportunity, RiskLevel
from .binary_arb import BinaryArbitrageStrategy
from .multi_outcome_arb import MultiOutcomeArbitrageStrategy
from .endgame_arb import EndgameArbitrageStrategy
from src.api.gamma_api import GammaAPI
from src.config import config


class ArbitrageDetector:
    """
    Main arbitrage detection engine.
    Combines multiple strategies to scan for opportunities.
    """

    def __init__(
        self,
        min_profit_margin: Optional[float] = None,
        min_liquidity: Optional[float] = None,
    ):
        """
        Initialize the arbitrage detector.

        Args:
            min_profit_margin: Override minimum profit margin
            min_liquidity: Override minimum liquidity requirement
        """
        self.min_profit_margin = min_profit_margin or config.trading.min_profit_margin
        self.min_liquidity = min_liquidity or config.trading.min_liquidity

        # Initialize strategies based on config
        self.strategies = []

        if config.strategy.enable_binary_arb:
            self.strategies.append(BinaryArbitrageStrategy(
                min_profit_margin=self.min_profit_margin
            ))
            logger.info("Binary arbitrage strategy enabled")

        if config.strategy.enable_multi_outcome_arb:
            self.strategies.append(MultiOutcomeArbitrageStrategy(
                min_profit_margin=self.min_profit_margin
            ))
            logger.info("Multi-outcome arbitrage strategy enabled")

        if config.strategy.enable_endgame_arb:
            self.strategies.append(EndgameArbitrageStrategy())
            logger.info("Endgame arbitrage strategy enabled")

        # API client
        self.gamma_api = GammaAPI()

        # State
        self._last_scan: Optional[datetime] = None
        self._cached_opportunities: List[ArbitrageOpportunity] = []

    async def scan(
        self,
        markets: Optional[List[Dict[str, Any]]] = None,
        max_risk: RiskLevel = RiskLevel.HIGH,
    ) -> List[ArbitrageOpportunity]:
        """
        Scan markets for arbitrage opportunities.

        Args:
            markets: Optional pre-fetched markets (fetches if None)
            max_risk: Maximum acceptable risk level

        Returns:
            List of detected opportunities, sorted by profit margin
        """
        start_time = datetime.now()

        # Fetch markets if not provided
        if markets is None:
            logger.info("Fetching active markets...")
            markets = await self.gamma_api.get_all_active_markets()

        logger.info(f"Scanning {len(markets)} markets with {len(self.strategies)} strategies...")

        # Run all strategies concurrently
        all_opportunities = []
        tasks = [strategy.detect(markets) for strategy in self.strategies]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Strategy {i} failed: {result}")
                continue
            all_opportunities.extend(result)

        # Filter and sort
        filtered = self._filter_opportunities(all_opportunities, max_risk)

        # Cache results
        self._cached_opportunities = filtered
        self._last_scan = datetime.now()

        scan_duration = (datetime.now() - start_time).total_seconds()
        logger.info(
            f"Scan complete in {scan_duration:.2f}s. "
            f"Found {len(filtered)} opportunities from {len(all_opportunities)} candidates."
        )

        return filtered

    def _filter_opportunities(
        self,
        opportunities: List[ArbitrageOpportunity],
        max_risk: RiskLevel,
    ) -> List[ArbitrageOpportunity]:
        """Filter and deduplicate opportunities."""
        risk_order = {RiskLevel.LOW: 0, RiskLevel.MEDIUM: 1, RiskLevel.HIGH: 2}

        # Filter by risk and liquidity
        filtered = [
            opp for opp in opportunities
            if risk_order[opp.risk_level] <= risk_order[max_risk]
            and opp.liquidity >= self.min_liquidity
            and opp.profit_margin >= self.min_profit_margin
        ]

        # Deduplicate by market ID (keep highest profit)
        seen_markets: Dict[str, ArbitrageOpportunity] = {}
        for opp in filtered:
            key = f"{opp.market_id}-{opp.strategy_type.value}"
            if key not in seen_markets or opp.profit_margin > seen_markets[key].profit_margin:
                seen_markets[key] = opp

        # Sort by profit margin
        result = sorted(seen_markets.values(), key=lambda x: x.profit_margin, reverse=True)

        return result

    async def scan_continuous(
        self,
        interval_seconds: int = 5,
        callback=None,
        max_iterations: Optional[int] = None,
    ):
        """
        Continuously scan for opportunities.

        Args:
            interval_seconds: Seconds between scans
            callback: Optional callback function(opportunities)
            max_iterations: Maximum number of scans (None for infinite)
        """
        iteration = 0

        while max_iterations is None or iteration < max_iterations:
            try:
                opportunities = await self.scan()

                if callback:
                    await callback(opportunities) if asyncio.iscoroutinefunction(callback) else callback(opportunities)

                if opportunities:
                    self._log_opportunities(opportunities[:5])  # Log top 5

                await asyncio.sleep(interval_seconds)
                iteration += 1

            except KeyboardInterrupt:
                logger.info("Stopping continuous scan...")
                break
            except Exception as e:
                logger.error(f"Scan error: {e}")
                await asyncio.sleep(interval_seconds)

    def _log_opportunities(self, opportunities: List[ArbitrageOpportunity]):
        """Log opportunities in a readable format."""
        logger.info("=" * 60)
        logger.info(f"Top {len(opportunities)} Opportunities:")
        logger.info("=" * 60)

        for i, opp in enumerate(opportunities, 1):
            logger.info(
                f"{i}. [{opp.strategy_type.value.upper()}] "
                f"ROI: {opp.roi_percent:.2f}% | "
                f"Profit: ${opp.profit_amount:.4f} | "
                f"Risk: {opp.risk_level.value}"
            )
            logger.info(f"   Market: {opp.market_question[:60]}...")

        logger.info("=" * 60)

    def get_cached_opportunities(self) -> List[ArbitrageOpportunity]:
        """Get opportunities from last scan."""
        return self._cached_opportunities.copy()

    def get_last_scan_time(self) -> Optional[datetime]:
        """Get timestamp of last scan."""
        return self._last_scan

    async def close(self):
        """Clean up resources."""
        await self.gamma_api.close()


# Convenience function for quick scans
async def quick_scan(
    min_profit: float = 0.005,
    max_risk: RiskLevel = RiskLevel.HIGH,
) -> List[ArbitrageOpportunity]:
    """
    Run a quick one-time scan for arbitrage opportunities.

    Args:
        min_profit: Minimum profit margin (0.005 = 0.5%)
        max_risk: Maximum risk level to include

    Returns:
        List of opportunities
    """
    detector = ArbitrageDetector(min_profit_margin=min_profit)
    try:
        return await detector.scan(max_risk=max_risk)
    finally:
        await detector.close()
