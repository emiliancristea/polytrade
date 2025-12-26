"""
Base classes for arbitrage strategies.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class StrategyType(Enum):
    """Types of arbitrage strategies."""
    BINARY = "binary"
    MULTI_OUTCOME = "multi_outcome"
    ENDGAME = "endgame"
    CROSS_PLATFORM = "cross_platform"


class RiskLevel(Enum):
    """Risk classification for opportunities."""
    LOW = "low"          # Near risk-free (e.g., pure arbitrage)
    MEDIUM = "medium"    # Some execution risk
    HIGH = "high"        # Significant risk (e.g., endgame)


@dataclass
class ArbitrageOpportunity:
    """
    Represents a detected arbitrage opportunity.
    """
    # Identification
    id: str
    strategy_type: StrategyType
    market_id: str
    market_question: str

    # Financials
    profit_margin: float          # As decimal (0.05 = 5%)
    profit_amount: float          # Absolute profit in USDC
    investment_required: float    # Total capital needed
    expected_return: float        # Total return (investment + profit)

    # Execution details
    trades: List[Dict[str, Any]]  # List of trades to execute
    token_ids: List[str]          # Token IDs involved
    prices: Dict[str, float]      # Current prices

    # Metadata
    risk_level: RiskLevel
    detected_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    liquidity: float = 0.0
    confidence: float = 1.0       # 0-1 confidence score

    # Additional data
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def roi_percent(self) -> float:
        """Return on investment as percentage."""
        if self.investment_required <= 0:
            return 0.0
        return (self.profit_amount / self.investment_required) * 100

    @property
    def is_profitable(self) -> bool:
        """Check if opportunity is still profitable."""
        return self.profit_margin > 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "strategy_type": self.strategy_type.value,
            "market_id": self.market_id,
            "market_question": self.market_question,
            "profit_margin": self.profit_margin,
            "profit_amount": self.profit_amount,
            "investment_required": self.investment_required,
            "expected_return": self.expected_return,
            "roi_percent": self.roi_percent,
            "risk_level": self.risk_level.value,
            "detected_at": self.detected_at.isoformat(),
            "trades": self.trades,
            "liquidity": self.liquidity,
            "confidence": self.confidence,
        }

    def __str__(self) -> str:
        return (
            f"[{self.strategy_type.value.upper()}] {self.market_question[:50]}... "
            f"| Profit: ${self.profit_amount:.4f} ({self.roi_percent:.2f}%) "
            f"| Risk: {self.risk_level.value}"
        )


class BaseStrategy(ABC):
    """
    Abstract base class for arbitrage strategies.
    """

    def __init__(self, min_profit_margin: float = 0.005):
        """
        Initialize strategy.

        Args:
            min_profit_margin: Minimum profit margin to consider (0.005 = 0.5%)
        """
        self.min_profit_margin = min_profit_margin
        self.strategy_type: StrategyType = StrategyType.BINARY

    @abstractmethod
    async def detect(
        self, markets: List[Dict[str, Any]]
    ) -> List[ArbitrageOpportunity]:
        """
        Detect arbitrage opportunities in markets.

        Args:
            markets: List of market data from Gamma API

        Returns:
            List of detected opportunities
        """
        pass

    @abstractmethod
    def calculate_profit(
        self, prices: Dict[str, float], investment: float
    ) -> Optional[ArbitrageOpportunity]:
        """
        Calculate potential profit for given prices.

        Args:
            prices: Dictionary of outcome -> price
            investment: Amount to invest

        Returns:
            ArbitrageOpportunity if profitable, None otherwise
        """
        pass

    def filter_opportunities(
        self,
        opportunities: List[ArbitrageOpportunity],
        min_liquidity: float = 0,
        max_risk: RiskLevel = RiskLevel.HIGH,
    ) -> List[ArbitrageOpportunity]:
        """
        Filter opportunities based on criteria.

        Args:
            opportunities: List of opportunities to filter
            min_liquidity: Minimum liquidity required
            max_risk: Maximum acceptable risk level

        Returns:
            Filtered list of opportunities
        """
        risk_order = {RiskLevel.LOW: 0, RiskLevel.MEDIUM: 1, RiskLevel.HIGH: 2}

        filtered = [
            opp for opp in opportunities
            if opp.liquidity >= min_liquidity
            and risk_order[opp.risk_level] <= risk_order[max_risk]
            and opp.profit_margin >= self.min_profit_margin
        ]

        # Sort by profit margin descending
        return sorted(filtered, key=lambda x: x.profit_margin, reverse=True)
