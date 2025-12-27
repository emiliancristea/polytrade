#!/usr/bin/env python3
"""
Enhanced Simulation with On-Chain Intelligence

Combines realistic simulation constraints with on-chain risk scoring
for more accurate profit projections.

Features:
- Risk-adjusted position sizing
- Whale activity awareness
- Market dynamics filtering
- Higher quality signal selection

Usage:
    python enhanced_simulate.py [--balance 100] [--duration 60] [--min-confidence 0.5]
"""

import asyncio
import argparse
import sys
import random
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set

sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
    level="INFO",
    colorize=True,
)


@dataclass
class EnhancedPosition:
    """Position with risk-adjusted sizing."""
    condition_id: str
    market_name: str
    entry_time: datetime
    estimated_resolution: datetime
    cost: float
    payout: float
    profit: float
    risk_score: float
    confidence: float
    position_multiplier: float


@dataclass
class EnhancedStats:
    """Enhanced simulation statistics."""
    start_time: datetime = field(default_factory=datetime.now)
    starting_balance: float = 100.0
    total_balance: float = 100.0

    # Trade tracking
    opportunities_seen: int = 0
    opportunities_filtered: int = 0
    opportunities_traded: int = 0
    resolved_trades: int = 0

    # Profit tracking
    realized_profit: float = 0.0

    # Quality metrics
    avg_confidence: float = 0.0
    avg_risk_score: float = 0.0
    total_volume: float = 0.0

    @property
    def runtime_minutes(self) -> float:
        return (datetime.now() - self.start_time).total_seconds() / 60


class EnhancedSimulator:
    """
    Enhanced simulator with on-chain intelligence integration.

    Key improvements:
    1. Risk-adjusted position sizing (reduce size for risky opps)
    2. Filters out high-risk opportunities
    3. Tracks quality metrics (confidence, risk score)
    4. More realistic profit projections
    """

    def __init__(
        self,
        starting_balance: float = 100.0,
        min_confidence: float = 0.5,
        use_onchain: bool = True
    ):
        self.stats = EnhancedStats(
            starting_balance=starting_balance,
            total_balance=starting_balance,
        )

        self.min_confidence = min_confidence
        self.max_position_pct = 0.25
        self.min_trade_size = 5.0

        # Tracking
        self.traded_markets: Set[str] = set()
        self.open_positions: Dict[str, EnhancedPosition] = {}
        self.resolved_trades: List[dict] = []

        # Time acceleration
        self.time_acceleration = 60

        # Risk scorer
        self._risk_scorer = None
        if use_onchain:
            try:
                from src.blockchain.risk_scorer import create_risk_scorer
                self._risk_scorer = create_risk_scorer(use_onchain=True)
                logger.info("On-chain risk scoring enabled")
            except Exception as e:
                logger.warning(f"On-chain scoring unavailable: {e}")

        if not self._risk_scorer:
            from src.blockchain.risk_scorer import SimpleRiskScorer
            self._risk_scorer = SimpleRiskScorer()

    @property
    def locked_capital(self) -> float:
        return sum(pos.cost for pos in self.open_positions.values())

    @property
    def available_balance(self) -> float:
        return self.stats.total_balance - self.locked_capital

    @property
    def unrealized_profit(self) -> float:
        return sum(pos.profit for pos in self.open_positions.values())

    def estimate_resolution_time(self) -> datetime:
        hours = random.uniform(1, 48)
        real_seconds = (hours * 3600) / self.time_acceleration
        return datetime.now() + timedelta(seconds=real_seconds)

    async def check_resolutions(self):
        """Check and resolve completed positions."""
        now = datetime.now()
        resolved_ids = []

        for cid, pos in self.open_positions.items():
            if now >= pos.estimated_resolution:
                resolved_ids.append(cid)

                self.stats.total_balance += pos.payout
                self.stats.realized_profit += pos.profit
                self.stats.resolved_trades += 1

                self.resolved_trades.append({
                    'profit': pos.profit,
                    'confidence': pos.confidence,
                    'market': pos.market_name
                })

                logger.success(
                    f"[RESOLVED] +${pos.profit:.4f} | "
                    f"Confidence: {pos.confidence:.0%} | "
                    f"{pos.market_name[:35]}"
                )

        for cid in resolved_ids:
            del self.open_positions[cid]

    async def on_opportunity(self, opp: Dict[str, Any]):
        """Handle opportunity with enhanced filtering."""
        self.stats.opportunities_seen += 1
        await self.check_resolutions()

        condition_id = opp["condition_id"]

        # Skip if already traded
        if condition_id in self.traded_markets:
            return

        # Get risk assessment
        assessment = self._risk_scorer.assess_opportunity(opp)

        # Filter based on confidence
        if assessment.confidence < self.min_confidence:
            self.stats.opportunities_filtered += 1
            return

        # Filter based on recommendation
        if assessment.recommendation == 'skip':
            self.stats.opportunities_filtered += 1
            return

        # Calculate risk-adjusted position size
        available = self.available_balance
        base_size = available * self.max_position_pct
        adjusted_size = base_size * assessment.adjusted_position_multiplier

        if adjusted_size < self.min_trade_size:
            self.stats.opportunities_filtered += 1
            return

        adjusted_size = min(adjusted_size, available * 0.5)

        # Calculate trade
        total = opp["total"]
        total_cost = adjusted_size * total
        payout = adjusted_size
        profit = payout - total_cost

        # Execute
        self.stats.total_balance -= total_cost
        self.traded_markets.add(condition_id)

        position = EnhancedPosition(
            condition_id=condition_id,
            market_name=opp["question"][:50],
            entry_time=datetime.now(),
            estimated_resolution=self.estimate_resolution_time(),
            cost=total_cost,
            payout=payout,
            profit=profit,
            risk_score=assessment.risk_score,
            confidence=assessment.confidence,
            position_multiplier=assessment.adjusted_position_multiplier
        )
        self.open_positions[condition_id] = position

        # Update stats
        self.stats.opportunities_traded += 1
        self.stats.total_volume += adjusted_size

        # Update averages
        n = self.stats.opportunities_traded
        self.stats.avg_confidence = (
            (self.stats.avg_confidence * (n-1) + assessment.confidence) / n
        )
        self.stats.avg_risk_score = (
            (self.stats.avg_risk_score * (n-1) + assessment.risk_score) / n
        )

        logger.success(
            f"[TRADE #{n}] "
            f"Size: ${adjusted_size:.2f} (adjusted) | "
            f"Profit: ${profit:.4f} | "
            f"Confidence: {assessment.confidence:.0%} | "
            f"Risk: {assessment.risk_score:.2f}"
        )
        logger.info(f"  {opp['question'][:60]}")

    def print_summary(self):
        """Print enhanced simulation summary."""
        total_profit = self.stats.realized_profit + self.unrealized_profit

        print("")
        print("=" * 70)
        print("ENHANCED SIMULATION SUMMARY (On-Chain Intelligence)")
        print("=" * 70)
        print("")
        print(f"Runtime: {self.stats.runtime_minutes:.1f} minutes")
        print(f"Min Confidence Threshold: {self.min_confidence:.0%}")
        print("")

        print("-" * 70)
        print("CAPITAL STATUS")
        print("-" * 70)
        print(f"Starting Balance:     ${self.stats.starting_balance:.2f}")
        print(f"Available Balance:    ${self.available_balance:.2f}")
        print(f"Locked in Positions:  ${self.locked_capital:.2f}")
        print(f"Total Value:          ${self.stats.total_balance + self.locked_capital:.2f}")
        print("")

        print("-" * 70)
        print("PROFIT STATUS")
        print("-" * 70)
        print(f"Realized Profit:      ${self.stats.realized_profit:.4f}")
        print(f"Unrealized Profit:    ${self.unrealized_profit:.4f}")
        print(f"Total Profit:         ${total_profit:.4f}")
        print("")

        print("-" * 70)
        print("QUALITY METRICS (Higher = Better Selection)")
        print("-" * 70)
        print(f"Average Confidence:   {self.stats.avg_confidence:.1%}")
        print(f"Average Risk Score:   {self.stats.avg_risk_score:.2f} (lower is better)")
        print(f"Filter Rate:          {(self.stats.opportunities_filtered / max(self.stats.opportunities_seen, 1)) * 100:.1f}%")
        print("")

        print("-" * 70)
        print("TRADE STATISTICS")
        print("-" * 70)
        print(f"Opportunities Seen:    {self.stats.opportunities_seen}")
        print(f"Filtered Out:          {self.stats.opportunities_filtered}")
        print(f"Unique Markets Traded: {len(self.traded_markets)}")
        print(f"Open Positions:        {len(self.open_positions)}")
        print(f"Resolved:              {self.stats.resolved_trades}")
        print(f"Total Volume:          ${self.stats.total_volume:.2f}")
        print("")

        if len(self.traded_markets) > 0:
            print("-" * 70)
            print("EXPECTED WIN RATE")
            print("-" * 70)
            # With proper filtering, win rate should be higher
            expected_wr = min(95, 70 + (self.stats.avg_confidence * 25))
            print(f"Based on avg confidence {self.stats.avg_confidence:.0%}:")
            print(f"  Expected Win Rate: ~{expected_wr:.0f}%")
            print("")
            print("NOTE: Arbitrage trades have ~95%+ win rate when executed properly.")
            print("      Losses come from execution failures, not prediction errors.")

        print("=" * 70)


async def run_enhanced_simulation(
    starting_balance: float = 100.0,
    min_confidence: float = 0.5,
    duration_minutes: Optional[int] = None,
    use_onchain: bool = True
):
    """Run enhanced simulation."""
    from src.api.gamma_api import GammaAPI
    from src.monitoring.realtime_scanner import RealtimeArbitrageScanner
    from src.config import config

    print("")
    print("=" * 70)
    print("ENHANCED ARBITRAGE SIMULATOR")
    print("=" * 70)
    print("")
    print("FEATURES:")
    print("  [1] On-chain risk scoring")
    print("  [2] Confidence-based filtering")
    print("  [3] Risk-adjusted position sizing")
    print("  [4] Realistic capital locking")
    print("")
    print(f"Starting Balance: ${starting_balance:.2f}")
    print(f"Min Confidence: {min_confidence:.0%}")
    print(f"On-Chain Intelligence: {'Enabled' if use_onchain else 'Disabled'}")
    print("")
    print("-" * 70)

    # Initialize
    simulator = EnhancedSimulator(
        starting_balance=starting_balance,
        min_confidence=min_confidence,
        use_onchain=use_onchain
    )

    # Load markets
    gamma = GammaAPI()
    logger.info("Fetching active markets...")

    markets = await gamma.get_markets(active=True, closed=False, limit=500)
    if not markets:
        logger.error("No markets found!")
        return

    logger.info(f"Found {len(markets)} active markets")

    # Initialize scanner
    scanner = RealtimeArbitrageScanner()
    scanner.set_execution_callback(simulator.on_opportunity)
    await scanner.load_markets(markets)

    # Initial scan
    initial = scanner.do_full_scan()
    if initial:
        logger.info(f"Initial scan: {len(initial)} opportunities, filtering...")
        for opp in initial:
            await simulator.on_opportunity(opp)

    print("")
    logger.info("Starting enhanced monitoring... Press Ctrl+C to stop")
    print("")

    # Background tasks
    async def status_printer():
        while True:
            await asyncio.sleep(30)
            await simulator.check_resolutions()
            logger.info(
                f"[STATUS] Traded: {len(simulator.traded_markets)} | "
                f"Open: {len(simulator.open_positions)} | "
                f"Filtered: {simulator.stats.opportunities_filtered} | "
                f"Avg Conf: {simulator.stats.avg_confidence:.0%}"
            )

    async def resolution_checker():
        while True:
            await asyncio.sleep(10)
            await simulator.check_resolutions()

    status_task = asyncio.create_task(status_printer())
    resolution_task = asyncio.create_task(resolution_checker())

    try:
        if duration_minutes:
            await asyncio.wait_for(scanner.run(), timeout=duration_minutes * 60)
        else:
            await scanner.run()
    except asyncio.TimeoutError:
        logger.info(f"Completed after {duration_minutes} minutes")
    except KeyboardInterrupt:
        logger.info("Stopped by user")
    finally:
        status_task.cancel()
        resolution_task.cancel()
        try:
            await status_task
        except asyncio.CancelledError:
            pass
        try:
            await resolution_task
        except asyncio.CancelledError:
            pass
        await scanner.stop()
        await simulator.check_resolutions()
        simulator.print_summary()


def main():
    parser = argparse.ArgumentParser(description="Enhanced Arbitrage Simulator")
    parser.add_argument(
        "--balance", type=float, default=100.0,
        help="Starting balance (default: 100)"
    )
    parser.add_argument(
        "--duration", type=int, default=None,
        help="Duration in minutes (default: until Ctrl+C)"
    )
    parser.add_argument(
        "--min-confidence", type=float, default=0.5,
        help="Minimum confidence threshold 0-1 (default: 0.5)"
    )
    parser.add_argument(
        "--no-onchain", action="store_true",
        help="Disable on-chain intelligence"
    )

    args = parser.parse_args()

    try:
        asyncio.run(run_enhanced_simulation(
            starting_balance=args.balance,
            min_confidence=args.min_confidence,
            duration_minutes=args.duration,
            use_onchain=not args.no_onchain
        ))
    except KeyboardInterrupt:
        print("\nSimulation ended.")


if __name__ == "__main__":
    main()
