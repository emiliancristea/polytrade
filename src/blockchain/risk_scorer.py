"""
Risk Scorer with On-Chain Intelligence

Calculates risk scores for arbitrage opportunities using:
- On-chain trade flow analysis
- Whale activity detection
- Market dynamics (minting/burning)
- Position conversion signals

Higher quality signals = higher win rates.
"""

from typing import Any, Dict, Optional
from dataclasses import dataclass
from loguru import logger

from src.blockchain.onchain_monitor import OnChainMonitor, create_onchain_monitor


@dataclass
class RiskAssessment:
    """Complete risk assessment for an opportunity."""
    risk_score: float  # 0-1 (lower is better)
    confidence: float  # 0-1 (higher is better)
    whale_activity: bool
    flow_imbalance: float
    market_health: str
    recommendation: str  # 'trade', 'caution', 'skip'
    reasons: list
    adjusted_position_multiplier: float  # 0-1, reduce position by this


class OnChainRiskScorer:
    """
    Risk scoring using on-chain intelligence.

    Key risk indicators (from research):
    - Whale activity: Informed traders may know something
    - Strong directional flow: Market moving against arb
    - Spread too good to be true: >5% usually means hidden risk
    - Market contracting: Burning > minting
    - Recent position conversions: Smart money rebalancing
    """

    def __init__(self, on_chain_monitor: Optional[OnChainMonitor] = None):
        """Initialize risk scorer."""
        self.chain = on_chain_monitor or create_onchain_monitor()

        # Thresholds from research
        self.WHALE_TRADE_THRESHOLD = 10000  # $10k+
        self.SUSPICIOUS_SPREAD = 0.05  # 5%+
        self.HIGH_FLOW_IMBALANCE = 0.7  # 70%+ one-sided

    def assess_opportunity(
        self,
        opportunity: Dict[str, Any],
        lookback_blocks: int = 50
    ) -> RiskAssessment:
        """
        Full risk assessment for an arbitrage opportunity.

        Args:
            opportunity: Dict with spread, prices, market info
            lookback_blocks: How far back to analyze (~2 min)

        Returns:
            RiskAssessment with score and recommendation
        """
        reasons = []
        risk_score = 0.0

        spread = opportunity.get('spread', 0) or opportunity.get('profit_margin', 0)
        market_id = opportunity.get('condition_id', '')

        # 1. Check on-chain trade flow
        flow_data = self._analyze_trade_flow(lookback_blocks)
        flow_imbalance = flow_data.get('imbalance', 0)

        if flow_imbalance > self.HIGH_FLOW_IMBALANCE:
            risk_score += 0.3
            reasons.append(f"High flow imbalance: {flow_imbalance:.1%}")

        # 2. Check for whale activity
        whale_activity = self._detect_whale_activity(lookback_blocks)

        if whale_activity:
            risk_score += 0.25
            reasons.append("Whale activity detected in recent blocks")

        # 3. Check market dynamics
        market_health = self._check_market_health()

        if market_health == 'contracting':
            risk_score += 0.15
            reasons.append("Market liquidity contracting (burning > minting)")

        # 4. Check spread size (too good = suspicious)
        if spread > self.SUSPICIOUS_SPREAD:
            risk_score += 0.2
            reasons.append(f"Suspiciously large spread: {spread:.1%}")
        elif spread > 0.03:
            risk_score += 0.1
            reasons.append(f"Large spread warrants verification: {spread:.1%}")

        # 5. Check for position conversions (smart money signals)
        conversions = self._check_conversions(lookback_blocks)
        if conversions > 5:
            risk_score += 0.1
            reasons.append(f"High conversion activity: {conversions} events")

        # Cap at 1.0
        risk_score = min(risk_score, 1.0)

        # Calculate confidence (inverse of risk)
        confidence = 1.0 - risk_score

        # Determine recommendation
        if risk_score < 0.3:
            recommendation = 'trade'
        elif risk_score < 0.6:
            recommendation = 'caution'
        else:
            recommendation = 'skip'

        # Position multiplier (reduce size based on risk)
        position_multiplier = max(0.2, 1.0 - risk_score)

        return RiskAssessment(
            risk_score=risk_score,
            confidence=confidence,
            whale_activity=whale_activity,
            flow_imbalance=flow_imbalance,
            market_health=market_health,
            recommendation=recommendation,
            reasons=reasons if reasons else ['No significant risks detected'],
            adjusted_position_multiplier=position_multiplier
        )

    def _analyze_trade_flow(self, lookback_blocks: int) -> Dict[str, Any]:
        """Analyze recent trade flow for sentiment."""
        if not self.chain.connected:
            return {'imbalance': 0, 'activity': 'unknown'}

        flow = self.chain.calculate_trade_flow(lookback_blocks)

        return {
            'imbalance': 0.0,  # Would need full decoding for directional
            'activity': flow.get('activity_level', 'unknown'),
            'trade_count': flow.get('trade_count', 0)
        }

    def _detect_whale_activity(self, lookback_blocks: int) -> bool:
        """Check for large trades in recent blocks."""
        if not self.chain.connected:
            return False

        whale_trades = self.chain.detect_whale_trades(
            min_size_usd=self.WHALE_TRADE_THRESHOLD,
            lookback_blocks=lookback_blocks
        )

        return len(whale_trades) > 0

    def _check_market_health(self) -> str:
        """Check minting vs burning dynamics."""
        if not self.chain.connected:
            return 'unknown'

        dynamics = self.chain.analyze_market_dynamics(lookback_blocks=500)
        return dynamics.liquidity_health

    def _check_conversions(self, lookback_blocks: int) -> int:
        """Count recent position conversions."""
        if not self.chain.connected:
            return 0

        conversions = self.chain.get_position_conversions(lookback_blocks=lookback_blocks)
        return len(conversions)

    def should_skip_opportunity(
        self,
        opportunity: Dict[str, Any]
    ) -> tuple[bool, Optional[str]]:
        """
        Quick check if opportunity should be skipped.

        Returns:
            (should_skip, reason)
        """
        assessment = self.assess_opportunity(opportunity)

        if assessment.recommendation == 'skip':
            return True, assessment.reasons[0] if assessment.reasons else "High risk"

        return False, None

    def adjust_position_size(
        self,
        base_size: float,
        opportunity: Dict[str, Any]
    ) -> float:
        """
        Adjust position size based on risk.

        Args:
            base_size: Original position size
            opportunity: Opportunity dict

        Returns:
            Risk-adjusted position size
        """
        assessment = self.assess_opportunity(opportunity)
        adjusted = base_size * assessment.adjusted_position_multiplier

        if adjusted != base_size:
            logger.debug(
                f"Position adjusted: ${base_size:.2f} -> ${adjusted:.2f} "
                f"(risk: {assessment.risk_score:.2f})"
            )

        return adjusted

    def get_quick_score(self, opportunity: Dict[str, Any]) -> float:
        """
        Get just the risk score without full assessment.

        Returns:
            Risk score 0-1 (lower is better)
        """
        return self.assess_opportunity(opportunity).risk_score


class SimpleRiskScorer:
    """
    Simplified risk scorer for when on-chain monitoring is not available.

    Uses only the opportunity data itself.
    """

    def __init__(self):
        self.SUSPICIOUS_SPREAD = 0.05

    def assess_opportunity(self, opportunity: Dict[str, Any]) -> RiskAssessment:
        """Basic risk assessment without on-chain data."""
        reasons = []
        risk_score = 0.0

        spread = opportunity.get('spread', 0) or opportunity.get('profit_margin', 0)

        # Check spread size
        if spread > self.SUSPICIOUS_SPREAD:
            risk_score += 0.3
            reasons.append(f"Suspiciously large spread: {spread:.1%}")
        elif spread > 0.03:
            risk_score += 0.15
            reasons.append(f"Large spread: {spread:.1%}")

        # Check liquidity
        liquidity = opportunity.get('liquidity', 0)
        if liquidity < 100:
            risk_score += 0.2
            reasons.append(f"Low liquidity: ${liquidity}")

        # Check volume
        volume = opportunity.get('volume', 0)
        if volume < 1000:
            risk_score += 0.15
            reasons.append(f"Low volume: ${volume}")

        risk_score = min(risk_score, 1.0)
        confidence = 1.0 - risk_score

        if risk_score < 0.3:
            recommendation = 'trade'
        elif risk_score < 0.6:
            recommendation = 'caution'
        else:
            recommendation = 'skip'

        return RiskAssessment(
            risk_score=risk_score,
            confidence=confidence,
            whale_activity=False,
            flow_imbalance=0,
            market_health='unknown',
            recommendation=recommendation,
            reasons=reasons if reasons else ['Basic assessment only'],
            adjusted_position_multiplier=max(0.3, 1.0 - risk_score)
        )


def create_risk_scorer(
    use_onchain: bool = True,
    rpc_url: Optional[str] = None
) -> OnChainRiskScorer | SimpleRiskScorer:
    """
    Create appropriate risk scorer.

    Args:
        use_onchain: Whether to use on-chain monitoring
        rpc_url: Custom RPC URL for on-chain
    """
    if use_onchain:
        try:
            monitor = create_onchain_monitor(rpc_url)
            if monitor.connected:
                return OnChainRiskScorer(monitor)
        except Exception as e:
            logger.warning(f"On-chain scorer unavailable: {e}")

    return SimpleRiskScorer()
