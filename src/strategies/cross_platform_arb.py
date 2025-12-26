"""
Cross-Platform Arbitrage Strategy.

Detects arbitrage opportunities between Polymarket and other
prediction market platforms (Kalshi, PredictIt).

Key insight: Platforms serve different user bases who react
differently to identical news, creating persistent mispricings.

Example:
- Polymarket YES: $0.42
- Kalshi NO: $0.56 (equivalent to YES at $0.44)
- Combined cost: $0.98
- Guaranteed payout: $1.00
- Profit: 2.04% risk-free

Fee considerations:
- Polymarket: 0% (zero fees)
- Kalshi: ~0.7%
- PredictIt: $850 limit + 10% profit fee
"""

import asyncio
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import aiohttp
from loguru import logger

from .base import ArbitrageOpportunity, BaseStrategy, RiskLevel, StrategyType
from src.config import config


@dataclass
class ExternalMarket:
    """Market data from external platform."""
    platform: str
    market_id: str
    question: str
    yes_price: float
    no_price: float
    fee_rate: float  # Platform fee rate
    last_updated: datetime


class KalshiClient:
    """
    Client for Kalshi prediction market API.

    Note: Kalshi requires authentication for trading.
    This implementation focuses on price fetching for arbitrage detection.
    """

    BASE_URL = "https://trading-api.kalshi.com/trade-api/v2"

    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None
        self.fee_rate = 0.007  # ~0.7% fee

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def get_markets(
        self,
        status: str = "open",
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Fetch markets from Kalshi.

        Args:
            status: "open", "closed", "settled"
            limit: Max markets to return

        Returns:
            List of market dictionaries
        """
        session = await self._get_session()
        params = {"status": status, "limit": limit}

        try:
            async with session.get(
                f"{self.BASE_URL}/markets",
                params=params,
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("markets", [])
                else:
                    logger.debug(f"Kalshi API returned {response.status}")
                    return []
        except Exception as e:
            logger.debug(f"Kalshi fetch failed: {e}")
            return []

    async def get_market(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Fetch a specific market by ticker."""
        session = await self._get_session()

        try:
            async with session.get(
                f"{self.BASE_URL}/markets/{ticker}"
            ) as response:
                if response.status == 200:
                    return await response.json()
                return None
        except Exception as e:
            logger.debug(f"Kalshi market fetch failed: {e}")
            return None

    def parse_market(self, market: Dict[str, Any]) -> Optional[ExternalMarket]:
        """Parse Kalshi market data into ExternalMarket."""
        try:
            ticker = market.get("ticker", "")
            title = market.get("title", "")

            # Kalshi uses yes_bid/yes_ask for pricing
            yes_bid = float(market.get("yes_bid", 0) or 0) / 100  # Kalshi uses cents
            yes_ask = float(market.get("yes_ask", 0) or 0) / 100

            if yes_ask <= 0:
                return None

            return ExternalMarket(
                platform="kalshi",
                market_id=ticker,
                question=title,
                yes_price=yes_ask,  # What we'd pay to buy YES
                no_price=1.0 - yes_bid,  # What we'd pay to buy NO
                fee_rate=self.fee_rate,
                last_updated=datetime.now(),
            )
        except Exception as e:
            logger.debug(f"Failed to parse Kalshi market: {e}")
            return None


class CrossPlatformArbitrageStrategy(BaseStrategy):
    """
    Strategy for cross-platform arbitrage between prediction markets.
    """

    def __init__(
        self,
        min_profit_margin: float = 0.01,  # 1% minimum after fees
        platforms: Optional[List[str]] = None,
    ):
        """
        Initialize cross-platform strategy.

        Args:
            min_profit_margin: Minimum profit after fees
            platforms: List of platforms to check ("kalshi", "predictit")
        """
        super().__init__(min_profit_margin)
        self.strategy_type = StrategyType.CROSS_PLATFORM
        self.platforms = platforms or ["kalshi"]

        # Platform clients
        self.kalshi = KalshiClient() if "kalshi" in self.platforms else None

        # Cache for external markets
        self._external_cache: Dict[str, List[ExternalMarket]] = {}

    async def close(self):
        """Close platform clients."""
        if self.kalshi:
            await self.kalshi.close()

    async def detect(
        self, markets: List[Dict[str, Any]]
    ) -> List[ArbitrageOpportunity]:
        """
        Detect cross-platform arbitrage opportunities.

        Args:
            markets: Polymarket markets from Gamma API

        Returns:
            List of cross-platform arbitrage opportunities
        """
        opportunities = []

        # Fetch external market data
        await self._refresh_external_markets()

        for pm_market in markets:
            try:
                opp = await self._analyze_market(pm_market)
                if opp:
                    opportunities.append(opp)
            except Exception as e:
                logger.debug(f"Cross-platform analysis error: {e}")

        logger.info(f"Cross-platform strategy found {len(opportunities)} opportunities")
        return self.filter_opportunities(opportunities)

    async def _refresh_external_markets(self):
        """Refresh external market cache."""
        if self.kalshi:
            try:
                kalshi_markets = await self.kalshi.get_markets(limit=200)
                self._external_cache["kalshi"] = [
                    self.kalshi.parse_market(m)
                    for m in kalshi_markets
                    if self.kalshi.parse_market(m) is not None
                ]
                logger.debug(f"Cached {len(self._external_cache.get('kalshi', []))} Kalshi markets")
            except Exception as e:
                logger.debug(f"Kalshi refresh failed: {e}")

    async def _analyze_market(
        self, pm_market: Dict[str, Any]
    ) -> Optional[ArbitrageOpportunity]:
        """Analyze a Polymarket against external platforms."""
        pm_question = pm_market.get("question", "").lower()
        pm_outcomes = pm_market.get("outcomes", [])

        if len(pm_outcomes) != 2:
            return None

        # Get Polymarket prices
        pm_prices = self._parse_pm_prices(pm_market)
        if not pm_prices:
            return None

        pm_yes = pm_prices.get("Yes", 0)
        pm_no = pm_prices.get("No", 0)

        # Check against each external platform
        for platform, ext_markets in self._external_cache.items():
            for ext in ext_markets:
                if not self._markets_match(pm_question, ext.question):
                    continue

                # Check for arbitrage
                opp = self._check_cross_arb(
                    pm_market, pm_yes, pm_no, ext
                )
                if opp:
                    return opp

        return None

    def _markets_match(self, pm_question: str, ext_question: str) -> bool:
        """
        Check if two market questions match.
        Uses simple keyword matching - could be improved with fuzzy matching.
        """
        pm_words = set(pm_question.lower().split())
        ext_words = set(ext_question.lower().split())

        # Remove common words
        stopwords = {"will", "the", "a", "an", "be", "in", "on", "at", "to", "for", "of"}
        pm_words -= stopwords
        ext_words -= stopwords

        # Check overlap
        overlap = len(pm_words & ext_words)
        min_len = min(len(pm_words), len(ext_words))

        if min_len == 0:
            return False

        return overlap / min_len >= 0.5  # 50% word overlap

    def _check_cross_arb(
        self,
        pm_market: Dict[str, Any],
        pm_yes: float,
        pm_no: float,
        ext: ExternalMarket,
    ) -> Optional[ArbitrageOpportunity]:
        """
        Check for arbitrage between Polymarket and external platform.

        Scenarios:
        1. Buy PM YES + Buy EXT NO = guaranteed $1 for < $1
        2. Buy PM NO + Buy EXT YES = guaranteed $1 for < $1
        """
        # Scenario 1: PM YES + External NO
        cost_1 = pm_yes + ext.no_price
        # Account for external platform fee
        effective_cost_1 = pm_yes + ext.no_price * (1 + ext.fee_rate)
        profit_1 = 1.0 - effective_cost_1

        # Scenario 2: PM NO + External YES
        cost_2 = pm_no + ext.yes_price
        effective_cost_2 = pm_no + ext.yes_price * (1 + ext.fee_rate)
        profit_2 = 1.0 - effective_cost_2

        # Find best scenario
        if profit_1 > profit_2 and profit_1 >= self.min_profit_margin:
            return self._create_opportunity(
                pm_market, pm_yes, ext, "NO", profit_1, effective_cost_1
            )
        elif profit_2 >= self.min_profit_margin:
            return self._create_opportunity(
                pm_market, pm_no, ext, "YES", profit_2, effective_cost_2
            )

        return None

    def _create_opportunity(
        self,
        pm_market: Dict[str, Any],
        pm_price: float,
        ext: ExternalMarket,
        ext_side: str,  # What to buy on external platform
        profit_margin: float,
        total_cost: float,
    ) -> ArbitrageOpportunity:
        """Create cross-platform arbitrage opportunity."""
        investment = 100.0
        shares = investment / total_cost
        profit = shares * profit_margin

        pm_side = "YES" if ext_side == "NO" else "NO"
        ext_price = ext.no_price if ext_side == "NO" else ext.yes_price

        trades = [
            {
                "platform": "polymarket",
                "action": "BUY",
                "outcome": pm_side,
                "price": pm_price,
            },
            {
                "platform": ext.platform,
                "action": "BUY",
                "outcome": ext_side,
                "price": ext_price,
                "fee_rate": ext.fee_rate,
            },
        ]

        return ArbitrageOpportunity(
            id=f"cross-{uuid.uuid4().hex[:8]}",
            strategy_type=StrategyType.CROSS_PLATFORM,
            market_id=pm_market.get("conditionId", ""),
            market_question=pm_market.get("question", ""),
            profit_margin=profit_margin,
            profit_amount=profit,
            investment_required=investment,
            expected_return=investment + profit,
            trades=trades,
            token_ids=[],  # Cross-platform uses different identifiers
            prices={
                f"polymarket_{pm_side}": pm_price,
                f"{ext.platform}_{ext_side}": ext_price,
            },
            risk_level=RiskLevel.MEDIUM,  # Cross-platform has execution risk
            liquidity=0,  # Would need to check both platforms
            confidence=0.85,
            metadata={
                "external_platform": ext.platform,
                "external_market_id": ext.market_id,
                "total_cost": total_cost,
                "fee_adjusted": True,
            },
        )

    def _parse_pm_prices(self, market: Dict[str, Any]) -> Dict[str, float]:
        """Parse Polymarket prices."""
        outcomes = market.get("outcomes", [])
        prices_raw = market.get("outcomePrices", [])

        if isinstance(prices_raw, str):
            try:
                prices_raw = [float(p.strip()) for p in prices_raw.strip("[]").split(",")]
            except ValueError:
                return {}

        prices = {}
        for i, outcome in enumerate(outcomes):
            if i < len(prices_raw):
                prices[outcome] = float(prices_raw[i])

        return prices

    def calculate_profit(
        self, prices: Dict[str, float], investment: float
    ) -> Optional[ArbitrageOpportunity]:
        """
        Calculate cross-platform profit.

        Args:
            prices: {"polymarket_yes": 0.42, "kalshi_no": 0.56, "kalshi_fee": 0.007}
            investment: Investment amount
        """
        pm_yes = prices.get("polymarket_yes", 0)
        ext_no = prices.get("kalshi_no", prices.get("external_no", 0))
        fee = prices.get("kalshi_fee", prices.get("external_fee", 0.007))

        if pm_yes <= 0 or ext_no <= 0:
            return None

        total_cost = pm_yes + ext_no * (1 + fee)
        profit_margin = 1.0 - total_cost

        if profit_margin < self.min_profit_margin:
            return None

        shares = investment / total_cost
        profit = shares * profit_margin

        return ArbitrageOpportunity(
            id=f"calc-cross-{uuid.uuid4().hex[:8]}",
            strategy_type=StrategyType.CROSS_PLATFORM,
            market_id="calculation",
            market_question="Manual cross-platform calculation",
            profit_margin=profit_margin,
            profit_amount=profit,
            investment_required=investment,
            expected_return=investment + profit,
            trades=[],
            token_ids=[],
            prices=prices,
            risk_level=RiskLevel.MEDIUM,
            confidence=1.0,
        )


# Utility function
def calculate_cross_platform_arb(
    pm_yes: float,
    ext_no: float,
    ext_fee: float = 0.007,
    investment: float = 100,
) -> Dict[str, float]:
    """
    Quick calculation for cross-platform arbitrage.

    Args:
        pm_yes: Polymarket YES price
        ext_no: External platform NO price
        ext_fee: External platform fee rate
        investment: Investment amount

    Returns:
        Calculation results
    """
    effective_ext_no = ext_no * (1 + ext_fee)
    total_cost = pm_yes + effective_ext_no
    profit_margin = 1.0 - total_cost
    shares = investment / total_cost if total_cost > 0 else 0
    profit = shares * profit_margin

    return {
        "pm_yes_price": pm_yes,
        "ext_no_price": ext_no,
        "ext_fee": ext_fee,
        "effective_ext_no": effective_ext_no,
        "total_cost": total_cost,
        "profit_margin": profit_margin,
        "profit_margin_percent": profit_margin * 100,
        "investment": investment,
        "shares": shares,
        "profit": profit,
        "return": investment + profit,
        "is_profitable": profit_margin > 0,
    }
