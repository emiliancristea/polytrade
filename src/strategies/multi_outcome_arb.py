"""
Multi-Outcome Arbitrage Strategy.

Detects opportunities in markets with 3+ outcomes where the sum of all
outcome prices is less than $1.00.

Example (4-candidate election):
- Candidate A: $0.35
- Candidate B: $0.30
- Candidate C: $0.20
- Candidate D: $0.10
- Total: $0.95
- Profit: $0.05 per complete set (5.26% return)
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


class MultiOutcomeArbitrageStrategy(BaseStrategy):
    """
    Strategy for detecting multi-outcome market arbitrage.
    Looks for markets where sum of all outcome prices < $1.00.
    """

    def __init__(
        self,
        min_profit_margin: float = 0.005,
        min_outcomes: int = 3,
        use_live_prices: bool = True,
    ):
        """
        Initialize multi-outcome arbitrage strategy.

        Args:
            min_profit_margin: Minimum profit margin (0.005 = 0.5%)
            min_outcomes: Minimum number of outcomes to consider
            use_live_prices: If True, fetch live prices from CLOB
        """
        super().__init__(min_profit_margin)
        self.strategy_type = StrategyType.MULTI_OUTCOME
        self.min_outcomes = min_outcomes
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
        Detect multi-outcome arbitrage opportunities.

        Args:
            markets: List of market data from Gamma API

        Returns:
            List of multi-outcome arbitrage opportunities
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

        logger.info(f"Multi-outcome strategy found {len(opportunities)} opportunities")
        return self.filter_opportunities(opportunities)

    async def _analyze_market(
        self, market: Dict[str, Any]
    ) -> Optional[ArbitrageOpportunity]:
        """Analyze a single market for multi-outcome arbitrage."""
        outcomes = market.get("outcomes", [])

        # Must have minimum number of outcomes
        if len(outcomes) < self.min_outcomes:
            return None

        # Get market identifiers
        market_id = market.get("conditionId") or market.get("id", "")
        question = market.get("question", "Unknown")
        token_ids = self._extract_token_ids(market)

        if len(token_ids) != len(outcomes):
            return None

        # Get prices for all outcomes
        prices = await self._get_prices(market, token_ids, outcomes)
        if len(prices) != len(outcomes):
            return None

        # Check if any price is zero or invalid
        if any(p <= 0 for p in prices.values()):
            return None

        # Calculate total cost
        total_cost = sum(prices.values())
        profit_margin = 1.0 - total_cost

        # Check if profitable
        if profit_margin < self.min_profit_margin:
            return None

        # Calculate for $100 investment
        investment = 100.0
        shares_purchasable = investment / total_cost
        profit_amount = shares_purchasable * profit_margin

        # Get liquidity
        liquidity = float(market.get("liquidityNum", 0) or market.get("liquidity", 0) or 0)

        # Build trades for all outcomes
        trades = []
        for i, outcome in enumerate(outcomes):
            if i < len(token_ids):
                trades.append({
                    "action": "BUY",
                    "outcome": outcome,
                    "token_id": token_ids[i],
                    "price": prices[outcome],
                    "side": "BUY",
                })

        # Risk is slightly higher for multi-outcome due to execution complexity
        risk_level = RiskLevel.MEDIUM if len(outcomes) > 4 else RiskLevel.LOW

        return ArbitrageOpportunity(
            id=f"multi-{uuid.uuid4().hex[:8]}",
            strategy_type=StrategyType.MULTI_OUTCOME,
            market_id=market_id,
            market_question=question,
            profit_margin=profit_margin,
            profit_amount=profit_amount,
            investment_required=investment,
            expected_return=investment + profit_amount,
            trades=trades,
            token_ids=token_ids,
            prices=prices,
            risk_level=risk_level,
            liquidity=liquidity,
            confidence=0.90,  # Slightly lower confidence due to complexity
            metadata={
                "total_cost": total_cost,
                "num_outcomes": len(outcomes),
                "outcomes": outcomes,
                "market_slug": market.get("slug", ""),
            },
        )

    async def _get_prices(
        self,
        market: Dict[str, Any],
        token_ids: List[str],
        outcomes: List[str],
    ) -> Dict[str, float]:
        """Get prices for all market outcomes."""
        prices = {}

        if self.use_live_prices and token_ids:
            try:
                client = self._get_clob_client()
                for i, token_id in enumerate(token_ids):
                    if i < len(outcomes):
                        price = client.get_price(token_id, side="BUY")
                        if price is not None:
                            prices[outcomes[i]] = price
            except Exception as e:
                logger.debug(f"Failed to get live prices: {e}")

        # Fall back to API prices
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
            tokens = tokens.strip("[]").replace('"', "").split(",")
            tokens = [t.strip() for t in tokens if t.strip()]

        return tokens

    def calculate_profit(
        self, prices: Dict[str, float], investment: float
    ) -> Optional[ArbitrageOpportunity]:
        """
        Calculate profit for given outcome prices.

        Args:
            prices: Dictionary mapping outcome names to prices
            investment: Amount to invest in USDC

        Returns:
            ArbitrageOpportunity if profitable
        """
        if len(prices) < self.min_outcomes:
            return None

        if any(p <= 0 for p in prices.values()):
            return None

        total_cost = sum(prices.values())
        profit_margin = 1.0 - total_cost

        if profit_margin < self.min_profit_margin:
            return None

        shares = investment / total_cost
        profit_amount = shares * profit_margin

        return ArbitrageOpportunity(
            id=f"calc-multi-{uuid.uuid4().hex[:8]}",
            strategy_type=StrategyType.MULTI_OUTCOME,
            market_id="calculation",
            market_question="Manual calculation",
            profit_margin=profit_margin,
            profit_amount=profit_amount,
            investment_required=investment,
            expected_return=investment + profit_amount,
            trades=[],
            token_ids=[],
            prices=prices,
            risk_level=RiskLevel.MEDIUM,
            confidence=1.0,
            metadata={"num_outcomes": len(prices)},
        )


# Quick utility function
def check_multi_outcome_arb(prices: List[float], investment: float = 100) -> Dict[str, float]:
    """
    Quick check for multi-outcome arbitrage.

    Args:
        prices: List of prices for each outcome
        investment: Investment amount in USDC

    Returns:
        Dictionary with profit details
    """
    total_cost = sum(prices)
    profit_margin = 1.0 - total_cost
    shares = investment / total_cost if total_cost > 0 else 0
    profit = shares * profit_margin

    return {
        "num_outcomes": len(prices),
        "total_cost": total_cost,
        "profit_margin": profit_margin,
        "profit_margin_percent": profit_margin * 100,
        "investment": investment,
        "shares": shares,
        "profit": profit,
        "return": investment + profit,
        "is_profitable": profit_margin > 0,
    }
