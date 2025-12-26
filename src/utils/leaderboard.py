"""
Leaderboard Scraper for researching successful Polymarket traders.
Helps identify profitable strategies by analyzing top traders.
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional
import aiohttp
from loguru import logger

from src.config import config


class TraderProfile:
    """Profile of a successful trader."""

    def __init__(self, data: Dict[str, Any]):
        self.address = data.get("address", "")
        self.username = data.get("username", "")
        self.profit = float(data.get("profit", 0) or 0)
        self.volume = float(data.get("volume", 0) or 0)
        self.trades_count = int(data.get("numTrades", 0) or 0)
        self.positions_count = int(data.get("positionsCount", 0) or 0)
        self.rank = int(data.get("rank", 0) or 0)
        self.raw_data = data

    @property
    def roi(self) -> float:
        """Return on investment percentage."""
        if self.volume > 0:
            return (self.profit / self.volume) * 100
        return 0.0

    @property
    def avg_trade_size(self) -> float:
        """Average trade size."""
        if self.trades_count > 0:
            return self.volume / self.trades_count
        return 0.0

    def __str__(self) -> str:
        return (
            f"#{self.rank} {self.username or self.address[:10]}... | "
            f"Profit: ${self.profit:,.2f} | "
            f"Volume: ${self.volume:,.2f} | "
            f"Trades: {self.trades_count:,}"
        )


class LeaderboardScraper:
    """
    Scrapes and analyzes Polymarket leaderboard data.
    Useful for researching successful trading strategies.
    """

    def __init__(self):
        self.gamma_api = config.api.gamma_api_host
        self._session: Optional[aiohttp.ClientSession] = None
        self._cached_leaderboard: List[TraderProfile] = []
        self._last_fetch: Optional[datetime] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create async session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self):
        """Close session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def fetch_leaderboard(
        self,
        limit: int = 100,
        offset: int = 0,
    ) -> List[TraderProfile]:
        """
        Fetch leaderboard data.

        Args:
            limit: Number of entries to fetch
            offset: Pagination offset

        Returns:
            List of TraderProfile objects
        """
        session = await self._get_session()
        params = {"limit": limit, "offset": offset}

        try:
            async with session.get(
                f"{self.gamma_api}/leaderboard",
                params=params
            ) as response:
                response.raise_for_status()
                data = await response.json()

                if isinstance(data, dict):
                    entries = data.get("data", data.get("leaderboard", []))
                else:
                    entries = data

                profiles = [TraderProfile(entry) for entry in entries]
                self._cached_leaderboard = profiles
                self._last_fetch = datetime.now()

                logger.info(f"Fetched {len(profiles)} leaderboard entries")
                return profiles

        except Exception as e:
            logger.error(f"Failed to fetch leaderboard: {e}")
            return []

    async def fetch_top_traders(self, count: int = 50) -> List[TraderProfile]:
        """Fetch top N traders."""
        return await self.fetch_leaderboard(limit=count)

    async def fetch_trader_positions(
        self,
        address: str,
    ) -> List[Dict[str, Any]]:
        """
        Fetch positions for a specific trader.

        Args:
            address: Wallet address of the trader

        Returns:
            List of position dictionaries
        """
        session = await self._get_session()

        try:
            async with session.get(
                f"{self.gamma_api}/positions",
                params={"user": address}
            ) as response:
                response.raise_for_status()
                positions = await response.json()
                logger.debug(f"Fetched {len(positions)} positions for {address[:10]}...")
                return positions

        except Exception as e:
            logger.error(f"Failed to fetch positions for {address}: {e}")
            return []

    async def analyze_trader(self, address: str) -> Dict[str, Any]:
        """
        Analyze a trader's strategy based on their positions.

        Args:
            address: Wallet address

        Returns:
            Analysis dictionary
        """
        positions = await self.fetch_trader_positions(address)

        if not positions:
            return {"error": "No positions found"}

        # Analyze position patterns
        total_value = 0.0
        markets = set()
        outcomes = {"Yes": 0, "No": 0}
        price_ranges = []

        for pos in positions:
            value = float(pos.get("value", 0) or 0)
            total_value += value
            markets.add(pos.get("conditionId", ""))

            outcome = pos.get("outcome", "")
            if outcome in outcomes:
                outcomes[outcome] += 1

            avg_price = float(pos.get("avgPrice", 0) or 0)
            if avg_price > 0:
                price_ranges.append(avg_price)

        analysis = {
            "address": address,
            "total_positions": len(positions),
            "unique_markets": len(markets),
            "total_value": total_value,
            "outcome_distribution": outcomes,
            "avg_position_size": total_value / len(positions) if positions else 0,
        }

        if price_ranges:
            analysis["avg_entry_price"] = sum(price_ranges) / len(price_ranges)
            analysis["min_entry_price"] = min(price_ranges)
            analysis["max_entry_price"] = max(price_ranges)

        # Detect potential strategies
        strategies = []
        if outcomes["Yes"] > 0 and outcomes["No"] > 0:
            yes_ratio = outcomes["Yes"] / (outcomes["Yes"] + outcomes["No"])
            if 0.4 <= yes_ratio <= 0.6:
                strategies.append("hedge/arbitrage")

        if price_ranges:
            avg_price = sum(price_ranges) / len(price_ranges)
            if avg_price > 0.90:
                strategies.append("endgame")
            elif avg_price < 0.10:
                strategies.append("long_shot")

        if len(positions) > 100:
            strategies.append("high_volume")

        analysis["detected_strategies"] = strategies

        return analysis

    async def find_arbitrage_traders(
        self,
        min_trades: int = 1000,
        min_profit: float = 10000,
    ) -> List[TraderProfile]:
        """
        Find traders likely using arbitrage strategies.

        Criteria:
        - High number of trades
        - Consistent profits
        - High volume

        Args:
            min_trades: Minimum number of trades
            min_profit: Minimum all-time profit

        Returns:
            List of likely arbitrage traders
        """
        traders = await self.fetch_leaderboard(limit=500)

        arb_candidates = []
        for trader in traders:
            if trader.trades_count < min_trades:
                continue
            if trader.profit < min_profit:
                continue

            # High trade count with consistent profits suggests arbitrage
            # Low ROI with high volume also suggests low-risk strategies
            if trader.trades_count > 1000 and trader.roi < 50:
                arb_candidates.append(trader)
            # Very high trade counts
            elif trader.trades_count > 5000:
                arb_candidates.append(trader)

        logger.info(f"Found {len(arb_candidates)} potential arbitrage traders")
        return arb_candidates

    async def export_leaderboard_csv(
        self,
        filename: str = "leaderboard.csv",
        limit: int = 500,
    ):
        """Export leaderboard to CSV file."""
        traders = await self.fetch_leaderboard(limit=limit)

        if not traders:
            logger.error("No traders to export")
            return

        import csv
        with open(filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Rank", "Address", "Username", "Profit", "Volume",
                "Trades", "ROI%", "Avg Trade Size"
            ])

            for trader in traders:
                writer.writerow([
                    trader.rank,
                    trader.address,
                    trader.username,
                    f"{trader.profit:.2f}",
                    f"{trader.volume:.2f}",
                    trader.trades_count,
                    f"{trader.roi:.2f}",
                    f"{trader.avg_trade_size:.2f}",
                ])

        logger.info(f"Exported {len(traders)} traders to {filename}")

    def get_cached_leaderboard(self) -> List[TraderProfile]:
        """Get cached leaderboard data."""
        return self._cached_leaderboard.copy()


# Convenience function
async def analyze_top_traders(count: int = 10) -> List[Dict[str, Any]]:
    """
    Quick analysis of top traders.

    Args:
        count: Number of traders to analyze

    Returns:
        List of trader analyses
    """
    scraper = LeaderboardScraper()
    try:
        traders = await scraper.fetch_top_traders(count)
        analyses = []

        for trader in traders[:count]:
            analysis = await scraper.analyze_trader(trader.address)
            analysis["profile"] = str(trader)
            analyses.append(analysis)
            await asyncio.sleep(0.5)  # Rate limiting

        return analyses
    finally:
        await scraper.close()
