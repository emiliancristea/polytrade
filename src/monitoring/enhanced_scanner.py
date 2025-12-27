"""
Enhanced Arbitrage Scanner with On-Chain Intelligence

Combines real-time WebSocket monitoring with blockchain analysis
for higher quality signals and better win rates.

Target: 80-90% win rate through:
1. On-chain trade flow analysis
2. Whale activity detection
3. Market dynamics (minting/burning)
4. Risk-adjusted position sizing
"""

import asyncio
from typing import Any, Callable, Dict, List, Optional
from datetime import datetime
from loguru import logger

from src.monitoring.realtime_scanner import RealtimeArbitrageScanner
from src.blockchain.onchain_monitor import create_onchain_monitor, OnChainMonitor
from src.blockchain.risk_scorer import create_risk_scorer, OnChainRiskScorer, SimpleRiskScorer
from src.blockchain.whale_tracker import create_whale_tracker, WhaleTracker


class EnhancedArbitrageScanner:
    """
    Enhanced arbitrage scanner with on-chain intelligence.

    Key improvements over basic scanner:
    1. Risk scoring for each opportunity
    2. Whale activity awareness
    3. Market dynamics analysis
    4. Better position sizing based on confidence
    """

    def __init__(
        self,
        use_onchain: bool = True,
        rpc_url: Optional[str] = None,
        min_confidence: float = 0.5
    ):
        """
        Initialize enhanced scanner.

        Args:
            use_onchain: Enable on-chain monitoring (recommended)
            rpc_url: Custom Polygon RPC URL
            min_confidence: Minimum confidence to trade (0-1)
        """
        # Core scanner for WebSocket monitoring
        self._scanner = RealtimeArbitrageScanner()

        # On-chain components
        self._use_onchain = use_onchain
        self._onchain: Optional[OnChainMonitor] = None
        self._risk_scorer = None
        self._whale_tracker: Optional[WhaleTracker] = None

        if use_onchain:
            try:
                self._onchain = create_onchain_monitor(rpc_url)
                self._risk_scorer = create_risk_scorer(True, rpc_url)
                self._whale_tracker = create_whale_tracker(rpc_url)

                if self._onchain.connected:
                    logger.info("On-chain intelligence enabled")
                else:
                    logger.warning("On-chain connection failed - using basic mode")
                    self._risk_scorer = create_risk_scorer(False)
            except Exception as e:
                logger.warning(f"On-chain setup failed: {e}")
                self._risk_scorer = create_risk_scorer(False)
        else:
            self._risk_scorer = create_risk_scorer(False)

        # Filtering parameters
        self._min_confidence = min_confidence

        # Execution callback
        self._execution_callback: Optional[Callable] = None

        # Enhanced stats
        self._opportunities_assessed = 0
        self._opportunities_passed = 0
        self._opportunities_filtered = 0
        self._filter_reasons: Dict[str, int] = {}

    def set_execution_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Set callback for approved opportunities."""
        self._execution_callback = callback

        # Also set callback on base scanner, but we'll intercept it
        self._scanner.set_execution_callback(self._on_opportunity)

    async def load_markets(self, markets: List[Dict[str, Any]]):
        """Load markets into scanner."""
        await self._scanner.load_markets(markets)

    async def _on_opportunity(self, opportunity: Dict[str, Any]):
        """
        Handle opportunity from base scanner with enhanced filtering.

        This is where the on-chain intelligence adds value.
        """
        self._opportunities_assessed += 1

        # Get risk assessment
        assessment = self._risk_scorer.assess_opportunity(opportunity)

        # Log the assessment
        logger.debug(
            f"Opportunity assessed: {opportunity.get('question', '')[:30]}... | "
            f"Risk: {assessment.risk_score:.2f} | "
            f"Confidence: {assessment.confidence:.2f} | "
            f"Rec: {assessment.recommendation}"
        )

        # Filter based on recommendation
        if assessment.recommendation == 'skip':
            self._opportunities_filtered += 1
            reason = assessment.reasons[0] if assessment.reasons else "High risk"
            self._filter_reasons[reason] = self._filter_reasons.get(reason, 0) + 1
            logger.warning(
                f"[FILTERED] {opportunity.get('question', '')[:40]}... | "
                f"Risk: {assessment.risk_score:.2f} | "
                f"Reason: {reason}"
            )
            return

        # Check minimum confidence
        if assessment.confidence < self._min_confidence:
            self._opportunities_filtered += 1
            reason = f"Low confidence ({assessment.confidence:.1%})"
            self._filter_reasons[reason] = self._filter_reasons.get(reason, 0) + 1
            return

        # Approved - enhance opportunity with assessment data
        enhanced_opportunity = {
            **opportunity,
            'risk_score': assessment.risk_score,
            'confidence': assessment.confidence,
            'whale_activity': assessment.whale_activity,
            'market_health': assessment.market_health,
            'position_multiplier': assessment.adjusted_position_multiplier,
            'recommendation': assessment.recommendation,
            'risk_reasons': assessment.reasons
        }

        self._opportunities_passed += 1

        # Log enhanced opportunity
        logger.success(
            f"[APPROVED] {opportunity.get('question', '')[:40]}... | "
            f"Spread: {opportunity.get('profit_margin', 0)*100:.3f}% | "
            f"Confidence: {assessment.confidence:.1%} | "
            f"Position mult: {assessment.adjusted_position_multiplier:.1%}"
        )

        # Trigger callback
        if self._execution_callback:
            try:
                if asyncio.iscoroutinefunction(self._execution_callback):
                    await self._execution_callback(enhanced_opportunity)
                else:
                    self._execution_callback(enhanced_opportunity)
            except Exception as e:
                logger.error(f"Execution callback error: {e}")

    def do_full_scan(self) -> List[Dict[str, Any]]:
        """Perform full scan with enhanced filtering."""
        base_opportunities = self._scanner.do_full_scan()
        enhanced = []

        for opp in base_opportunities:
            assessment = self._risk_scorer.assess_opportunity(opp)

            if assessment.recommendation != 'skip' and assessment.confidence >= self._min_confidence:
                enhanced.append({
                    **opp,
                    'risk_score': assessment.risk_score,
                    'confidence': assessment.confidence,
                    'position_multiplier': assessment.adjusted_position_multiplier
                })

        logger.info(
            f"Full scan: {len(base_opportunities)} found, "
            f"{len(enhanced)} passed filters"
        )

        return enhanced

    async def run(self):
        """Run enhanced scanner."""
        logger.info("Starting enhanced arbitrage scanner with on-chain intelligence")

        if self._onchain and self._onchain.connected:
            # Log on-chain status
            dynamics = self._onchain.analyze_market_dynamics(lookback_blocks=500)
            logger.info(
                f"Market dynamics: {dynamics.liquidity_health} | "
                f"Net flow: ${dynamics.net_flow:,.0f}"
            )

        await self._scanner.run()

    async def stop(self):
        """Stop scanner."""
        await self._scanner.stop()

    def add_whale_wallet(self, address: str, label: Optional[str] = None):
        """Add a wallet to whale tracking."""
        if self._whale_tracker:
            self._whale_tracker.add_wallet(address, label)

    def get_stats(self) -> Dict[str, Any]:
        """Get enhanced scanner statistics."""
        base_stats = self._scanner.get_stats()

        return {
            **base_stats,
            'enhanced': {
                'on_chain_enabled': self._use_onchain,
                'on_chain_connected': self._onchain.connected if self._onchain else False,
                'opportunities_assessed': self._opportunities_assessed,
                'opportunities_passed': self._opportunities_passed,
                'opportunities_filtered': self._opportunities_filtered,
                'pass_rate': (
                    self._opportunities_passed / max(self._opportunities_assessed, 1)
                ) * 100,
                'filter_reasons': self._filter_reasons,
                'min_confidence': self._min_confidence,
                'tracked_whales': (
                    len(self._whale_tracker.tracked_wallets)
                    if self._whale_tracker else 0
                )
            }
        }

    def print_stats(self):
        """Print detailed statistics."""
        stats = self.get_stats()
        enhanced = stats.get('enhanced', {})

        print("")
        print("=" * 60)
        print("ENHANCED SCANNER STATISTICS")
        print("=" * 60)
        print("")

        print("On-Chain Intelligence:")
        print(f"  Enabled: {enhanced.get('on_chain_enabled')}")
        print(f"  Connected: {enhanced.get('on_chain_connected')}")
        print(f"  Tracked whales: {enhanced.get('tracked_whales', 0)}")
        print("")

        print("Opportunity Filtering:")
        print(f"  Assessed: {enhanced.get('opportunities_assessed', 0)}")
        print(f"  Passed: {enhanced.get('opportunities_passed', 0)}")
        print(f"  Filtered: {enhanced.get('opportunities_filtered', 0)}")
        print(f"  Pass rate: {enhanced.get('pass_rate', 0):.1f}%")
        print("")

        if enhanced.get('filter_reasons'):
            print("Filter Reasons:")
            for reason, count in enhanced['filter_reasons'].items():
                print(f"  {reason}: {count}")

        print("")
        print("=" * 60)


async def run_enhanced_scanner(
    starting_balance: float = 100.0,
    use_onchain: bool = True,
    min_confidence: float = 0.5,
    duration_minutes: Optional[int] = None
):
    """
    Run enhanced scanner with on-chain intelligence.

    Args:
        starting_balance: For position sizing
        use_onchain: Enable blockchain monitoring
        min_confidence: Minimum confidence threshold (0-1)
        duration_minutes: Run duration (None = until Ctrl+C)
    """
    from src.api.gamma_api import GammaAPI
    from src.config import config

    print("")
    print("=" * 70)
    print("ENHANCED ARBITRAGE SCANNER - ON-CHAIN INTELLIGENCE")
    print("=" * 70)
    print("")
    print(f"On-Chain Monitoring: {'Enabled' if use_onchain else 'Disabled'}")
    print(f"Min Confidence: {min_confidence:.0%}")
    print(f"Min Profit Threshold: {config.trading.min_profit_margin * 100:.3f}%")
    print("")

    # Initialize scanner
    scanner = EnhancedArbitrageScanner(
        use_onchain=use_onchain,
        min_confidence=min_confidence
    )

    # Simple callback for demo
    async def on_approved(opp):
        logger.success(
            f"[TRADE SIGNAL] {opp.get('question', '')[:50]} | "
            f"Profit: {opp.get('profit_margin', 0)*100:.3f}% | "
            f"Confidence: {opp.get('confidence', 0):.0%}"
        )

    scanner.set_execution_callback(on_approved)

    # Load markets
    gamma = GammaAPI()
    logger.info("Fetching active markets...")

    markets = await gamma.get_markets(
        active=True,
        closed=False,
        limit=500,
    )

    if not markets:
        logger.error("No markets found!")
        return

    logger.info(f"Found {len(markets)} active markets")
    await scanner.load_markets(markets)

    # Initial scan
    initial = scanner.do_full_scan()
    if initial:
        logger.info(f"Initial scan: {len(initial)} high-confidence opportunities")
        for opp in initial[:3]:
            await on_approved(opp)

    print("")
    logger.info("Starting enhanced monitoring... Press Ctrl+C to stop")
    print("")

    # Run with optional timeout
    try:
        if duration_minutes:
            await asyncio.wait_for(
                scanner.run(),
                timeout=duration_minutes * 60
            )
        else:
            await scanner.run()
    except asyncio.TimeoutError:
        logger.info(f"Completed after {duration_minutes} minutes")
    except KeyboardInterrupt:
        logger.info("Stopped by user")
    finally:
        await scanner.stop()
        scanner.print_stats()


if __name__ == "__main__":
    asyncio.run(run_enhanced_scanner(use_onchain=True, min_confidence=0.5))
