"""
Gamma API client for fetching Polymarket market data.
Provides read-only access to markets, prices, and leaderboards.
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional
import aiohttp
import requests
from loguru import logger

from src.config import config


class GammaAPI:
    """
    Client for Polymarket's Gamma API.
    Used for fetching market data, prices, and leaderboards.
    """

    def __init__(self, host: Optional[str] = None):
        self.host = host or config.api.gamma_api_host
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create async session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self):
        """Close the async session."""
        if self._session and not self._session.closed:
            await self._session.close()

    # =========================================
    # Market Data Methods
    # =========================================

    async def get_markets(
        self,
        active: bool = True,
        closed: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Fetch all markets from Gamma API.

        Args:
            active: Only return active markets
            closed: Include closed markets
            limit: Maximum number of markets to return
            offset: Pagination offset

        Returns:
            List of market dictionaries
        """
        session = await self._get_session()
        params = {
            "active": str(active).lower(),
            "closed": str(closed).lower(),
            "limit": limit,
            "offset": offset,
        }

        try:
            async with session.get(f"{self.host}/markets", params=params) as response:
                response.raise_for_status()
                markets = await response.json()
                logger.debug(f"Fetched {len(markets)} markets")
                return markets
        except Exception as e:
            logger.error(f"Failed to fetch markets: {e}")
            return []

    async def get_market(self, condition_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch a specific market by condition ID.

        Args:
            condition_id: The market's condition ID

        Returns:
            Market dictionary or None
        """
        session = await self._get_session()

        try:
            async with session.get(f"{self.host}/markets/{condition_id}") as response:
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            logger.error(f"Failed to fetch market {condition_id}: {e}")
            return None

    async def get_all_active_markets(self) -> List[Dict[str, Any]]:
        """
        Fetch all active markets with pagination.

        Returns:
            Complete list of active markets
        """
        all_markets = []
        offset = 0
        limit = 100

        while True:
            markets = await self.get_markets(active=True, limit=limit, offset=offset)
            if not markets:
                break
            all_markets.extend(markets)
            if len(markets) < limit:
                break
            offset += limit

        logger.info(f"Fetched {len(all_markets)} total active markets")
        return all_markets

    # =========================================
    # Market Analysis Methods
    # =========================================

    def parse_market_prices(self, market: Dict[str, Any]) -> Dict[str, float]:
        """
        Parse outcome prices from a market.

        Args:
            market: Market dictionary from API

        Returns:
            Dictionary mapping outcome names to prices
        """
        prices = {}
        outcomes = market.get("outcomes", [])
        outcome_prices = market.get("outcomePrices", [])

        if not outcome_prices:
            # Try to parse from string if needed
            price_str = market.get("outcomePrices", "")
            if isinstance(price_str, str) and price_str:
                try:
                    outcome_prices = [float(p) for p in price_str.strip("[]").split(",")]
                except ValueError:
                    return {}

        for i, outcome in enumerate(outcomes):
            if i < len(outcome_prices):
                try:
                    price = float(outcome_prices[i])
                    prices[outcome] = price
                except (ValueError, TypeError):
                    continue

        return prices

    def is_binary_market(self, market: Dict[str, Any]) -> bool:
        """Check if market is a binary YES/NO market."""
        outcomes = market.get("outcomes", [])
        return len(outcomes) == 2 and set(outcomes) == {"Yes", "No"}

    def get_market_tokens(self, market: Dict[str, Any]) -> List[str]:
        """Extract CLOB token IDs from market."""
        tokens = market.get("clobTokenIds", [])
        if isinstance(tokens, str):
            tokens = tokens.strip("[]").split(",")
        return [t.strip().strip('"') for t in tokens if t.strip()]

    # =========================================
    # Leaderboard Methods
    # =========================================

    async def get_leaderboard(
        self, limit: int = 100, offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Fetch the trading leaderboard.

        Args:
            limit: Number of entries to fetch
            offset: Pagination offset

        Returns:
            List of leaderboard entries
        """
        session = await self._get_session()
        params = {"limit": limit, "offset": offset}

        try:
            async with session.get(
                f"{self.host}/leaderboard", params=params
            ) as response:
                response.raise_for_status()
                data = await response.json()
                return data if isinstance(data, list) else data.get("data", [])
        except Exception as e:
            logger.error(f"Failed to fetch leaderboard: {e}")
            return []

    async def get_user_positions(self, address: str) -> List[Dict[str, Any]]:
        """
        Fetch positions for a specific wallet address.

        Args:
            address: Wallet address

        Returns:
            List of position dictionaries
        """
        session = await self._get_session()

        try:
            async with session.get(
                f"{self.host}/positions", params={"user": address}
            ) as response:
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            logger.error(f"Failed to fetch positions for {address}: {e}")
            return []

    # =========================================
    # Synchronous Convenience Methods
    # =========================================

    def get_markets_sync(self, active: bool = True, limit: int = 100) -> List[Dict]:
        """Synchronous version of get_markets for simple scripts."""
        params = {"active": str(active).lower(), "limit": limit}
        try:
            response = requests.get(f"{self.host}/markets", params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to fetch markets (sync): {e}")
            return []


# Convenience function for quick market fetching
async def fetch_markets() -> List[Dict[str, Any]]:
    """Quick helper to fetch all active markets."""
    api = GammaAPI()
    try:
        return await api.get_all_active_markets()
    finally:
        await api.close()
