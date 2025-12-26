"""Arbitrage detection strategies for Polymarket."""

from .base import ArbitrageOpportunity, BaseStrategy, StrategyType, RiskLevel
from .binary_arb import BinaryArbitrageStrategy
from .multi_outcome_arb import MultiOutcomeArbitrageStrategy
from .endgame_arb import EndgameArbitrageStrategy
from .cross_platform_arb import CrossPlatformArbitrageStrategy
from .whale_tracker import WhaleTracker, ConsensusSignal
from .detector import ArbitrageDetector

__all__ = [
    "ArbitrageOpportunity",
    "BaseStrategy",
    "StrategyType",
    "RiskLevel",
    "BinaryArbitrageStrategy",
    "MultiOutcomeArbitrageStrategy",
    "EndgameArbitrageStrategy",
    "CrossPlatformArbitrageStrategy",
    "WhaleTracker",
    "ConsensusSignal",
    "ArbitrageDetector",
]
