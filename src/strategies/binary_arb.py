"""
Binary Arbitrage Strategy.

Detects opportunities where YES + NO prices sum to less than $1.00.
This is the most common and often risk-free arbitrage in prediction markets.

Example:
- YES price: $0.45
- NO price: $0.50
- Total cost: $0.95
- Guaranteed payout: $1.00
- Profit: $0.05 (5.26% return)
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional
from loguru import logger

from .base import (
    ArbitrageOpportunity,
    BaseStrategy,
    RiskLevel,
    StrategyType,
)
from src.api.clob_client import PolymarketClient


class BinaryArbitrageStrategy(BaseStrategy):
    """
    Strategy for detecting binary market arbitrage.
    Looks for YES/NO pairs where combined cost < $1.00.
    """

    def __init__(
        self,
        min_profit_margin: float = 0.005,
        use_live_prices: bool = True,
    ):
        """
        Initialize binary arbitrage strategy.

        Args:
            min_profit_margin: Minimum profit margin (0.005 = 0.5%)
            use_live_prices: If True, fetch live prices from CLOB
        """
        super().__init__(min_profit_margin)
        self.strategy_type = StrategyType.BINARY
        self.use_live_prices = use_live_prices
        self._clob_client: Optional[PolymarketClient] = None

    def _get_clob_client(self) -> PolymarketClient:
        """Get or create CLOB client for live prices."""
        if self._clob_client is None:
            self._clob_client = PolymarketClient(read_only=True)
        return self._clob_client

    async def detect(
        self, markets: List[Dict[str, Any]]
    ) -> List[ArbitrageOpportunity]:
        """
        Detect binary arbitrage opportunities.

        Args:
            markets: List of market data from Gamma API

        Returns:
            List of binary arbitrage opportunities
        """
        opportunities = []

        for market in markets:
            try:
                opportunity = await self._analyze_market(market)
                if opportunity:
                    opportunities.append(opportunity)
            except Exception as e:
                logger.debug(f"Error analyzing market {market.get('id', 'unknown')}: {e}")
                continue

        logger.info(f"Binary strategy found {len(opportunities)} opportunities")
        return self.filter_opportunities(opportunities)

    async def _analyze_market(
        self, market: Dict[str, Any]
    ) -> Optional[ArbitrageOpportunity]:
        """Analyze a single market for binary arbitrage."""
        # Check if it's a binary market
        outcomes = market.get("outcomes", [])
        if len(outcomes) != 2:
            return None

        # Get market identifiers
        market_id = market.get("conditionId") or market.get("id", "")
        question = market.get("question", "Unknown")
        token_ids = self._extract_token_ids(market)

        if len(token_ids) != 2:
            return None

        # Get prices
        prices = await self._get_prices(market, token_ids)
        if not prices or len(prices) != 2:
            return None

        # Calculate arbitrage
        yes_price = prices.get("Yes", prices.get(outcomes[0], 0))
        no_price = prices.get("No", prices.get(outcomes[1], 0))

        if yes_price <= 0 or no_price <= 0:
            return None

        total_cost = yes_price + no_price
        profit_margin = 1.0 - total_cost

        # Check if profitable (accounting for small buffer for fees/slippage)
        if profit_margin < self.min_profit_margin:
            return None

        # Calculate for $100 investment
        investment = 100.0
        shares_purchasable = investment / total_cost
        profit_amount = shares_purchasable * profit_margin

        # Get liquidity
        liquidity = float(market.get("liquidityNum", 0) or market.get("liquidity", 0) or 0)

        # Build trades
        trades = [
            {
                "action": "BUY",
                "outcome": outcomes[0],
                "token_id": token_ids[0],
                "price": yes_price,
                "side": "BUY",
            },
            {
                "action": "BUY",
                "outcome": outcomes[1],
                "token_id": token_ids[1],
                "price": no_price,
                "side": "BUY",
            },
        ]

        return ArbitrageOpportunity(
            id=f"binary-{uuid.uuid4().hex[:8]}",
            strategy_type=StrategyType.BINARY,
            market_id=market_id,
            market_question=question,
            profit_margin=profit_margin,
            profit_amount=profit_amount,
            investment_required=investment,
            expected_return=investment + profit_amount,
            trades=trades,
            token_ids=token_ids,
            prices=prices,
            risk_level=RiskLevel.LOW,  # Binary arb is low risk
            liquidity=liquidity,
            confidence=0.95,  # High confidence for binary arb
            metadata={
                "total_cost": total_cost,
                "yes_price": yes_price,
                "no_price": no_price,
                "market_slug": market.get("slug", ""),
            },
        )

    async def _get_prices(
        self, market: Dict[str, Any], token_ids: List[str]
    ) -> Dict[str, float]:
        """
        Get prices for market outcomes.
        Uses live CLOB prices if enabled, otherwise API data.
        """
        outcomes = market.get("outcomes", [])
        prices = {}

        if self.use_live_prices and token_ids:
            # Fetch live prices from CLOB
            try:
                client = self._get_clob_client()
                for i, token_id in enumerate(token_ids):
                    if i < len(outcomes):
                        # Use best ask price (what we'd pay to buy)
                        price = client.get_price(token_id, side="BUY")
                        if price is not None:
                            prices[outcomes[i]] = price
            except Exception as e:
                logger.debug(f"Failed to get live prices: {e}")

        # Fall back to API prices if needed
        if len(prices) != len(outcomes):
            api_prices = market.get("outcomePrices", [])
            if isinstance(api_prices, str):
                try:
                    api_prices = [float(p.strip()) for p in api_prices.strip("[]").split(",")]
                except ValueError:
                    api_prices = []

            for i, outcome in enumerate(outcomes):
                if outcome not in prices and i < len(api_prices):
                    try:
                        prices[outcome] = float(api_prices[i])
                    except (ValueError, TypeError):
                        continue

        return prices

    def _extract_token_ids(self, market: Dict[str, Any]) -> List[str]:
        """Extract CLOB token IDs from market data."""
        tokens = market.get("clobTokenIds", [])

        if isinstance(tokens, str):
            # Parse string format: "[\"id1\", \"id2\"]"
            tokens = tokens.strip("[]").replace('"', "").split(",")
            tokens = [t.strip() for t in tokens if t.strip()]

        return tokens

    def calculate_profit(
        self, prices: Dict[str, float], investment: float
    ) -> Optional[ArbitrageOpportunity]:
        """
        Calculate profit for given prices and investment.

        Args:
            prices: {"Yes": price, "No": price}
            investment: Amount to invest in USDC

        Returns:
            ArbitrageOpportunity if profitable
        """
        if "Yes" not in prices or "No" not in prices:
            return None

        yes_price = prices["Yes"]
        no_price = prices["No"]
        total_cost = yes_price + no_price

        if total_cost >= 1.0:
            return None

        profit_margin = 1.0 - total_cost
        if profit_margin < self.min_profit_margin:
            return None

        shares = investment / total_cost
        profit_amount = shares * profit_margin

        return ArbitrageOpportunity(
            id=f"calc-{uuid.uuid4().hex[:8]}",
            strategy_type=StrategyType.BINARY,
            market_id="calculation",
            market_question="Manual calculation",
            profit_margin=profit_margin,
            profit_amount=profit_amount,
            investment_required=investment,
            expected_return=investment + profit_amount,
            trades=[],
            token_ids=[],
            prices=prices,
            risk_level=RiskLevel.LOW,
            confidence=1.0,
        )


# Quick utility function
def check_binary_arb(yes_price: float, no_price: float, investment: float = 100) -> Dict[str, float]:
    """
    Quick check for binary arbitrage opportunity.

    Args:
        yes_price: Price of YES shares (0-1)
        no_price: Price of NO shares (0-1)
        investment: Investment amount in USDC

    Returns:
        Dictionary with profit details
    """
    total_cost = yes_price + no_price
    profit_margin = 1.0 - total_cost
    shares = investment / total_cost if total_cost > 0 else 0
    profit = shares * profit_margin

    return {
        "total_cost": total_cost,
        "profit_margin": profit_margin,
        "profit_margin_percent": profit_margin * 100,
        "investment": investment,
        "shares": shares,
        "profit": profit,
        "return": investment + profit,
        "is_profitable": profit_margin > 0,
    }
