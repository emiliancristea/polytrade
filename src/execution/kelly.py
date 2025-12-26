"""
Kelly Criterion Position Sizing Module.

Implements optimal position sizing based on the Kelly formula
adapted for prediction markets with capital tier adjustments.

Kelly Formula for prediction markets:
    f = (q - p) / (1 - p)

Where:
    f = fraction of bankroll to bet
    q = your estimated true probability
    p = market price (implied probability)

Capital Tier Adjustments:
    $50-100:  Quarter-Kelly (0.25x) - conservative for small accounts
    $100-500: Half-Kelly (0.50x) - moderate aggression
    $500+:    Half-Kelly (0.50x) - still conservative for protection
"""

from dataclasses import dataclass
from typing import Optional, Tuple
from loguru import logger

from src.config import config
from src.strategies.base import ArbitrageOpportunity, RiskLevel


@dataclass
class PositionSizeResult:
    """Result of position sizing calculation."""
    recommended_size: float
    kelly_fraction: float
    kelly_multiplier: float
    reason: str
    adjusted_for: list  # What adjustments were made


class KellyPositionSizer:
    """
    Calculates optimal position sizes using Kelly Criterion
    with conservative adjustments for prediction market trading.
    """

    def __init__(self, bankroll: float):
        """
        Initialize position sizer.

        Args:
            bankroll: Current available capital in USDC
        """
        self.bankroll = bankroll
        self.kelly_multiplier = config.kelly.get_kelly_multiplier(bankroll)
        self.current_exposure = 0.0  # Track total open positions

    def update_bankroll(self, new_bankroll: float):
        """Update bankroll and recalculate multiplier."""
        self.bankroll = new_bankroll
        self.kelly_multiplier = config.kelly.get_kelly_multiplier(new_bankroll)

    def add_exposure(self, amount: float):
        """Track new position exposure."""
        self.current_exposure += amount

    def remove_exposure(self, amount: float):
        """Remove closed position exposure."""
        self.current_exposure = max(0, self.current_exposure - amount)

    def calculate_kelly_fraction(
        self,
        estimated_prob: float,
        market_price: float,
    ) -> float:
        """
        Calculate raw Kelly fraction.

        Args:
            estimated_prob: Your estimated true probability (0-1)
            market_price: Current market price (0-1)

        Returns:
            Raw Kelly fraction (before multiplier adjustment)
        """
        if market_price >= 1.0 or market_price <= 0.0:
            return 0.0

        if estimated_prob <= market_price:
            # No edge - don't bet
            return 0.0

        # Kelly formula: f = (q - p) / (1 - p)
        kelly = (estimated_prob - market_price) / (1.0 - market_price)

        return max(0.0, kelly)

    def calculate_arbitrage_size(
        self,
        opportunity: ArbitrageOpportunity,
    ) -> PositionSizeResult:
        """
        Calculate position size for arbitrage opportunities.
        Arbitrage is near-guaranteed, so Kelly isn't strictly needed,
        but we still apply exposure limits.

        Args:
            opportunity: The arbitrage opportunity

        Returns:
            PositionSizeResult with recommended size
        """
        adjustments = []

        # For arbitrage, we can be more aggressive since it's low risk
        # But still respect position limits
        base_size = self.bankroll * config.kelly.max_position_percent * 2  # 2x normal for arb

        # Check total exposure limit
        remaining_exposure = (
            self.bankroll * config.kelly.max_total_exposure
        ) - self.current_exposure

        if remaining_exposure <= 0:
            return PositionSizeResult(
                recommended_size=0.0,
                kelly_fraction=1.0,  # Arb is 100% confidence
                kelly_multiplier=self.kelly_multiplier,
                reason="Max exposure limit reached",
                adjusted_for=["exposure_limit"],
            )

        if base_size > remaining_exposure:
            base_size = remaining_exposure
            adjustments.append("exposure_limit")

        # Adjust for risk level
        risk_multipliers = {
            RiskLevel.LOW: 1.0,
            RiskLevel.MEDIUM: 0.7,
            RiskLevel.HIGH: 0.4,
        }
        risk_mult = risk_multipliers.get(opportunity.risk_level, 0.4)
        if risk_mult < 1.0:
            base_size *= risk_mult
            adjustments.append(f"risk_{opportunity.risk_level.value}")

        # Adjust for spread (profit margin)
        # Sweet spot: 2-3% - full size
        # 1-2%: 70% size
        # <1%: 50% size (slippage risk)
        spread = opportunity.profit_margin
        if spread < 0.01:
            base_size *= 0.5
            adjustments.append("low_spread")
        elif spread < 0.02:
            base_size *= 0.7
            adjustments.append("moderate_spread")

        # Ensure minimum size
        if base_size < config.kelly.min_trade_size:
            return PositionSizeResult(
                recommended_size=0.0,
                kelly_fraction=1.0,
                kelly_multiplier=self.kelly_multiplier,
                reason=f"Size ${base_size:.2f} below minimum ${config.kelly.min_trade_size}",
                adjusted_for=adjustments,
            )

        # Cap at max position size from config
        if base_size > config.trading.max_position_size:
            base_size = config.trading.max_position_size
            adjustments.append("max_position_cap")

        return PositionSizeResult(
            recommended_size=round(base_size, 2),
            kelly_fraction=1.0,  # Arb confidence
            kelly_multiplier=self.kelly_multiplier,
            reason="OK",
            adjusted_for=adjustments,
        )

    def calculate_directional_size(
        self,
        estimated_prob: float,
        market_price: float,
        risk_level: RiskLevel = RiskLevel.MEDIUM,
    ) -> PositionSizeResult:
        """
        Calculate position size for directional bets (non-arbitrage).

        Args:
            estimated_prob: Your estimated true probability
            market_price: Current market price
            risk_level: Risk assessment of the opportunity

        Returns:
            PositionSizeResult with recommended size
        """
        adjustments = []

        # Calculate raw Kelly
        raw_kelly = self.calculate_kelly_fraction(estimated_prob, market_price)

        if raw_kelly <= 0:
            return PositionSizeResult(
                recommended_size=0.0,
                kelly_fraction=0.0,
                kelly_multiplier=self.kelly_multiplier,
                reason="No edge - don't bet",
                adjusted_for=[],
            )

        # Apply Kelly multiplier (Quarter/Half Kelly)
        adjusted_kelly = raw_kelly * self.kelly_multiplier
        adjustments.append(f"kelly_{self.kelly_multiplier}x")

        # Calculate dollar amount
        base_size = self.bankroll * adjusted_kelly

        # Cap at max position percent
        max_position = self.bankroll * config.kelly.max_position_percent
        if base_size > max_position:
            base_size = max_position
            adjustments.append("max_position_percent")

        # Check total exposure
        remaining_exposure = (
            self.bankroll * config.kelly.max_total_exposure
        ) - self.current_exposure

        if base_size > remaining_exposure:
            if remaining_exposure <= 0:
                return PositionSizeResult(
                    recommended_size=0.0,
                    kelly_fraction=raw_kelly,
                    kelly_multiplier=self.kelly_multiplier,
                    reason="Max exposure limit reached",
                    adjusted_for=["exposure_limit"],
                )
            base_size = remaining_exposure
            adjustments.append("exposure_limit")

        # Risk adjustment
        risk_multipliers = {
            RiskLevel.LOW: 1.0,
            RiskLevel.MEDIUM: 0.8,
            RiskLevel.HIGH: 0.5,
        }
        risk_mult = risk_multipliers.get(risk_level, 0.5)
        if risk_mult < 1.0:
            base_size *= risk_mult
            adjustments.append(f"risk_{risk_level.value}")

        # Minimum size check
        if base_size < config.kelly.min_trade_size:
            return PositionSizeResult(
                recommended_size=0.0,
                kelly_fraction=raw_kelly,
                kelly_multiplier=self.kelly_multiplier,
                reason=f"Size ${base_size:.2f} below minimum",
                adjusted_for=adjustments,
            )

        return PositionSizeResult(
            recommended_size=round(base_size, 2),
            kelly_fraction=raw_kelly,
            kelly_multiplier=self.kelly_multiplier,
            reason="OK",
            adjusted_for=adjustments,
        )

    def get_stats(self) -> dict:
        """Get current position sizing stats."""
        return {
            "bankroll": self.bankroll,
            "kelly_multiplier": self.kelly_multiplier,
            "current_exposure": self.current_exposure,
            "remaining_exposure": max(
                0,
                self.bankroll * config.kelly.max_total_exposure - self.current_exposure
            ),
            "exposure_percent": (
                self.current_exposure / self.bankroll * 100
                if self.bankroll > 0 else 0
            ),
        }


# Utility functions

def calculate_kelly(edge: float, odds: float) -> float:
    """
    Calculate Kelly fraction from edge and odds.

    Args:
        edge: Your edge as decimal (0.05 = 5%)
        odds: Decimal odds (2.0 = even money)

    Returns:
        Kelly fraction
    """
    if odds <= 1:
        return 0.0
    return edge / (odds - 1)


def kelly_for_arbitrage(total_cost: float) -> Tuple[float, float]:
    """
    Calculate optimal position for arbitrage.

    Args:
        total_cost: Combined cost of YES + NO (e.g., 0.95)

    Returns:
        Tuple of (profit_margin, suggested_fraction)
    """
    if total_cost >= 1.0:
        return (0.0, 0.0)

    profit_margin = 1.0 - total_cost
    # For arbitrage, you can bet more aggressively
    # since it's guaranteed profit
    suggested_fraction = min(0.10, profit_margin * 2)  # Up to 10% of bankroll

    return (profit_margin, suggested_fraction)


def optimal_bet_size(
    bankroll: float,
    win_prob: float,
    market_price: float,
    kelly_fraction: float = 0.25,
) -> dict:
    """
    Calculate optimal bet size with full details.

    Args:
        bankroll: Available capital
        win_prob: Your estimated win probability
        market_price: Current market price
        kelly_fraction: Kelly multiplier (default quarter-Kelly)

    Returns:
        Dictionary with calculation details
    """
    if win_prob <= market_price:
        return {
            "has_edge": False,
            "edge": win_prob - market_price,
            "kelly_raw": 0.0,
            "kelly_adjusted": 0.0,
            "bet_size": 0.0,
            "expected_value": 0.0,
        }

    # Raw Kelly
    kelly_raw = (win_prob - market_price) / (1.0 - market_price)
    kelly_adjusted = kelly_raw * kelly_fraction

    # Bet size
    bet_size = bankroll * kelly_adjusted

    # Expected value
    win_return = (1.0 / market_price - 1) * bet_size
    lose_amount = bet_size
    expected_value = (win_prob * win_return) - ((1 - win_prob) * lose_amount)

    return {
        "has_edge": True,
        "edge": win_prob - market_price,
        "kelly_raw": kelly_raw,
        "kelly_adjusted": kelly_adjusted,
        "bet_size": round(bet_size, 2),
        "expected_value": round(expected_value, 2),
        "win_return": round(win_return, 2),
        "lose_amount": round(lose_amount, 2),
    }
