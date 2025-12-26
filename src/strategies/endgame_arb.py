"""
Endgame Arbitrage Strategy.

Targets near-certain outcomes (95-99% probability) close to market resolution.
While not risk-free, these can offer high annualized returns.

Example:
- Market: "Will sun rise tomorrow?" resolves in 1 day
- YES price: $0.99
- If correct: Profit $0.01 per share (1.01% return in 1 day = 368% annualized)

Risk: The 1% chance of being wrong means total loss of investment.
"""

import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from loguru import logger

from .base import (
    ArbitrageOpportunity,
    BaseStrategy,
    RiskLevel,
    StrategyType,
)
from src.api.clob_client import PolymarketClient


class EndgameArbitrageStrategy(BaseStrategy):
    """
    Strategy for endgame arbitrage on near-certain outcomes.
    Higher risk but potentially high annualized returns.
    """

    def __init__(
        self,
        min_probability: float = 0.95,
        max_days_to_resolution: int = 7,
        min_annualized_return: float = 0.50,  # 50% minimum annualized
        use_live_prices: bool = True,
    ):
        """
        Initialize endgame arbitrage strategy.

        Args:
            min_probability: Minimum probability threshold (0.95 = 95%)
            max_days_to_resolution: Maximum days until market resolves
            min_annualized_return: Minimum annualized return (0.50 = 50%)
            use_live_prices: If True, fetch live prices from CLOB
        """
        super().__init__(min_profit_margin=0.001)  # Lower threshold for endgame
        self.strategy_type = StrategyType.ENDGAME
        self.min_probability = min_probability
        self.max_days_to_resolution = max_days_to_resolution
        self.min_annualized_return = min_annualized_return
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
        Detect endgame arbitrage opportunities.

        Args:
            markets: List of market data from Gamma API

        Returns:
            List of endgame arbitrage opportunities
        """
        opportunities = []
        now = datetime.now()

        for market in markets:
            try:
                # Check if market is near resolution
                end_date = self._parse_date(market.get("endDate"))
                if not end_date:
                    continue

                days_to_resolution = (end_date - now).days
                if days_to_resolution < 0 or days_to_resolution > self.max_days_to_resolution:
                    continue

                opportunity = await self._analyze_market(market, days_to_resolution, end_date)
                if opportunity:
                    opportunities.append(opportunity)

            except Exception as e:
                logger.debug(f"Error analyzing market {market.get('id', 'unknown')}: {e}")
                continue

        logger.info(f"Endgame strategy found {len(opportunities)} opportunities")
        return self.filter_opportunities(opportunities)

    async def _analyze_market(
        self,
        market: Dict[str, Any],
        days_to_resolution: int,
        end_date: datetime,
    ) -> Optional[ArbitrageOpportunity]:
        """Analyze a market for endgame opportunities."""
        outcomes = market.get("outcomes", [])
        if len(outcomes) < 2:
            return None

        market_id = market.get("conditionId") or market.get("id", "")
        question = market.get("question", "Unknown")
        token_ids = self._extract_token_ids(market)

        if len(token_ids) != len(outcomes):
            return None

        # Get prices
        prices = await self._get_prices(market, token_ids, outcomes)
        if len(prices) != len(outcomes):
            return None

        # Find the highest probability outcome
        best_outcome = None
        best_price = 0.0
        best_token_id = None

        for i, outcome in enumerate(outcomes):
            price = prices.get(outcome, 0)
            if price > best_price and price >= self.min_probability:
                best_price = price
                best_outcome = outcome
                if i < len(token_ids):
                    best_token_id = token_ids[i]

        if not best_outcome or not best_token_id:
            return None

        # Calculate returns
        profit_per_share = 1.0 - best_price  # What we gain if correct
        profit_margin = profit_per_share / best_price  # ROI

        # Calculate annualized return
        # If days_to_resolution is 0, use 1 day minimum
        days = max(days_to_resolution, 1)
        annualized_return = (1 + profit_margin) ** (365 / days) - 1

        if annualized_return < self.min_annualized_return:
            return None

        # Calculate for $100 investment
        investment = 100.0
        shares = investment / best_price
        profit_amount = shares * profit_per_share

        # Get liquidity
        liquidity = float(market.get("liquidityNum", 0) or market.get("liquidity", 0) or 0)

        # Build trade
        trades = [{
            "action": "BUY",
            "outcome": best_outcome,
            "token_id": best_token_id,
            "price": best_price,
            "side": "BUY",
        }]

        # Risk is HIGH for endgame (not guaranteed profit)
        return ArbitrageOpportunity(
            id=f"endgame-{uuid.uuid4().hex[:8]}",
            strategy_type=StrategyType.ENDGAME,
            market_id=market_id,
            market_question=question,
            profit_margin=profit_margin,
            profit_amount=profit_amount,
            investment_required=investment,
            expected_return=investment + profit_amount,
            trades=trades,
            token_ids=[best_token_id],
            prices={best_outcome: best_price},
            risk_level=RiskLevel.HIGH,  # Endgame has risk of total loss
            liquidity=liquidity,
            confidence=best_price,  # Confidence is the probability
            expires_at=end_date,
            metadata={
                "best_outcome": best_outcome,
                "probability": best_price,
                "days_to_resolution": days_to_resolution,
                "annualized_return": annualized_return,
                "annualized_return_percent": annualized_return * 100,
                "profit_per_share": profit_per_share,
                "market_slug": market.get("slug", ""),
            },
        )

    async def _get_prices(
        self,
        market: Dict[str, Any],
        token_ids: List[str],
        outcomes: List[str],
    ) -> Dict[str, float]:
        """Get prices for market outcomes."""
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

    def _parse_date(self, date_str: Optional[str]) -> Optional[datetime]:
        """Parse date string to datetime."""
        if not date_str:
            return None

        try:
            # Try ISO format
            if "T" in date_str:
                return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            # Try simple date format
            return datetime.strptime(date_str, "%Y-%m-%d")
        except (ValueError, TypeError):
            return None

    def calculate_profit(
        self, prices: Dict[str, float], investment: float
    ) -> Optional[ArbitrageOpportunity]:
        """
        Calculate profit for a single high-probability outcome.

        Args:
            prices: {"outcome_name": probability}
            investment: Amount to invest

        Returns:
            ArbitrageOpportunity if meets criteria
        """
        if not prices:
            return None

        # Find highest probability
        best_outcome = max(prices.keys(), key=lambda k: prices[k])
        best_price = prices[best_outcome]

        if best_price < self.min_probability:
            return None

        profit_per_share = 1.0 - best_price
        profit_margin = profit_per_share / best_price
        shares = investment / best_price
        profit_amount = shares * profit_per_share

        return ArbitrageOpportunity(
            id=f"calc-endgame-{uuid.uuid4().hex[:8]}",
            strategy_type=StrategyType.ENDGAME,
            market_id="calculation",
            market_question="Manual calculation",
            profit_margin=profit_margin,
            profit_amount=profit_amount,
            investment_required=investment,
            expected_return=investment + profit_amount,
            trades=[],
            token_ids=[],
            prices=prices,
            risk_level=RiskLevel.HIGH,
            confidence=best_price,
            metadata={
                "best_outcome": best_outcome,
                "probability": best_price,
            },
        )


# Quick utility function
def calculate_endgame_returns(
    probability: float,
    days_to_resolution: int,
    investment: float = 100,
) -> Dict[str, float]:
    """
    Calculate potential returns for endgame arbitrage.

    Args:
        probability: Current price/probability (0.95-0.99)
        days_to_resolution: Days until market resolves
        investment: Investment amount in USDC

    Returns:
        Dictionary with return calculations
    """
    if probability <= 0 or probability >= 1:
        return {"error": "Invalid probability"}

    profit_per_share = 1.0 - probability
    shares = investment / probability
    profit = shares * profit_per_share
    roi = profit_per_share / probability

    days = max(days_to_resolution, 1)
    annualized = (1 + roi) ** (365 / days) - 1

    return {
        "probability": probability,
        "days_to_resolution": days_to_resolution,
        "investment": investment,
        "shares": shares,
        "profit_if_correct": profit,
        "loss_if_wrong": investment,
        "roi_percent": roi * 100,
        "annualized_return_percent": annualized * 100,
        "expected_value": (probability * profit) - ((1 - probability) * investment),
    }
