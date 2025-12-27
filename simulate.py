#!/usr/bin/env python3
"""
Polymarket Arbitrage Simulator - REALISTIC VERSION

Simulates trading with virtual money using realistic constraints:
- Each market can only be traded ONCE (no re-trading same opportunity)
- Capital is LOCKED until market resolution
- Only available (unlocked) capital can be used for new trades
- Simulates resolution timing based on market end dates

Usage:
    python simulate.py [--balance 100] [--duration 60]

Options:
    --balance   Starting balance in USDC (default: 100)
    --duration  Run for N minutes (default: runs until Ctrl+C)
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
class OpenPosition:
    """An open position waiting for market resolution."""
    condition_id: str
    market_name: str
    entry_time: datetime
    estimated_resolution: datetime
    cost: float  # What we paid (YES + NO)
    payout: float  # What we get back ($1 per share)
    profit: float  # Guaranteed profit when resolved
    yes_price: float
    no_price: float


@dataclass
class ResolvedTrade:
    """A trade that has been resolved (profit realized)."""
    condition_id: str
    market_name: str
    entry_time: datetime
    resolution_time: datetime
    cost: float
    payout: float
    profit: float
    hold_time_hours: float


@dataclass
class SimulatorStats:
    """Simulation statistics."""
    start_time: datetime = field(default_factory=datetime.now)
    starting_balance: float = 100.0
    total_balance: float = 100.0  # Total value (available + locked)

    # Tracking
    total_trades: int = 0
    resolved_trades: int = 0
    realized_profit: float = 0.0
    unrealized_profit: float = 0.0
    total_volume: float = 0.0

    # Opportunity tracking
    opportunities_seen: int = 0
    opportunities_skipped_duplicate: int = 0
    opportunities_skipped_no_capital: int = 0
    opportunities_traded: int = 0

    largest_profit: float = 0.0

    @property
    def runtime_minutes(self) -> float:
        return (datetime.now() - self.start_time).total_seconds() / 60


class RealisticArbitrageSimulator:
    """
    REALISTIC Arbitrage Simulator.

    Key constraints:
    1. Each market can only be traded ONCE
    2. Capital is locked until resolution
    3. Only available balance can be used
    4. Tracks open positions separately from resolved trades
    """

    def __init__(self, starting_balance: float = 100.0):
        self.stats = SimulatorStats(
            starting_balance=starting_balance,
            total_balance=starting_balance,
        )

        # Position sizing
        self.max_position_pct = 0.25  # Max 25% of AVAILABLE balance per trade
        self.min_trade_size = 5.0     # $5 minimum

        # REALISTIC TRACKING
        self.traded_markets: Set[str] = set()  # Markets we've already traded
        self.open_positions: Dict[str, OpenPosition] = {}  # condition_id -> position
        self.resolved_trades: List[ResolvedTrade] = []

        # For simulation: accelerate time (1 real minute = X simulated hours)
        # Set to 60 means 1 minute real time = 1 hour simulated
        self.time_acceleration = 60

    @property
    def locked_capital(self) -> float:
        """Capital currently locked in open positions."""
        return sum(pos.cost for pos in self.open_positions.values())

    @property
    def available_balance(self) -> float:
        """Capital available for new trades."""
        return self.stats.total_balance - self.locked_capital

    @property
    def unrealized_profit(self) -> float:
        """Profit from open positions (will be realized on resolution)."""
        return sum(pos.profit for pos in self.open_positions.values())

    def calculate_trade_size(self, profit_margin: float) -> float:
        """
        Calculate trade size based on AVAILABLE balance only.
        """
        available = self.available_balance
        max_size = available * self.max_position_pct

        # Scale based on profit margin
        if profit_margin > 0.005:  # > 0.5% profit
            size = max_size
        elif profit_margin > 0.001:  # > 0.1% profit
            size = max_size * 0.8
        else:
            size = max_size * 0.6

        # Ensure minimum trade size and don't exceed available
        size = max(self.min_trade_size, size)
        size = min(size, available * 0.5)  # Never use more than 50% available

        return size if size >= self.min_trade_size else 0

    def estimate_resolution_time(self, market_data: Optional[Dict] = None) -> datetime:
        """
        Estimate when a market will resolve.
        Uses actual end date if available, otherwise random 1-7 days.
        """
        # For simulation, we accelerate time
        # Real markets can take days/weeks, but we simulate faster

        # Random resolution: 1-48 hours (simulated)
        hours_until_resolution = random.uniform(1, 48)

        # With time acceleration, this happens faster in real time
        real_seconds = (hours_until_resolution * 3600) / self.time_acceleration

        return datetime.now() + timedelta(seconds=real_seconds)

    async def check_resolutions(self):
        """Check if any positions have resolved and realize profits."""
        now = datetime.now()
        resolved_ids = []

        for cid, position in self.open_positions.items():
            if now >= position.estimated_resolution:
                resolved_ids.append(cid)

                # Realize the profit
                hold_time = (now - position.entry_time).total_seconds() / 3600

                resolved = ResolvedTrade(
                    condition_id=cid,
                    market_name=position.market_name,
                    entry_time=position.entry_time,
                    resolution_time=now,
                    cost=position.cost,
                    payout=position.payout,
                    profit=position.profit,
                    hold_time_hours=hold_time * self.time_acceleration,  # Simulated hours
                )
                self.resolved_trades.append(resolved)

                # Add payout back to balance
                self.stats.total_balance += position.payout
                self.stats.realized_profit += position.profit
                self.stats.resolved_trades += 1

                logger.success(
                    f"[RESOLVED] +${position.profit:.4f} | "
                    f"Held {hold_time*self.time_acceleration:.1f}h (sim) | "
                    f"{position.market_name[:40]}"
                )

        # Remove resolved positions
        for cid in resolved_ids:
            del self.open_positions[cid]

    async def on_opportunity(self, opp: Dict[str, Any]):
        """
        Handle detected arbitrage opportunity with REALISTIC constraints.
        """
        self.stats.opportunities_seen += 1

        # Check for resolutions first
        await self.check_resolutions()

        yes_price = opp["yes_price"]
        no_price = opp["no_price"]
        total = opp["total"]
        profit_margin = opp["profit_margin"]
        question = opp["question"]
        condition_id = opp["condition_id"]

        # CONSTRAINT 1: Already traded this market?
        if condition_id in self.traded_markets:
            self.stats.opportunities_skipped_duplicate += 1
            # Silent skip - don't spam logs with duplicates
            return

        # Calculate trade size based on AVAILABLE balance
        trade_size = self.calculate_trade_size(profit_margin)

        # CONSTRAINT 2: Enough available capital?
        if trade_size < self.min_trade_size:
            self.stats.opportunities_skipped_no_capital += 1
            if self.stats.opportunities_skipped_no_capital % 10 == 1:  # Log occasionally
                logger.warning(
                    f"Capital locked - Available: ${self.available_balance:.2f}, "
                    f"Locked: ${self.locked_capital:.2f}, "
                    f"Open positions: {len(self.open_positions)}"
                )
            return

        # Calculate costs and profit
        total_cost = trade_size * total  # What we pay for YES + NO
        payout = trade_size  # We get $1 per share when resolved
        profit = payout - total_cost

        # CONSTRAINT 3: Lock the capital (deduct from total balance)
        self.stats.total_balance -= total_cost

        # Mark market as traded
        self.traded_markets.add(condition_id)

        # Create open position
        position = OpenPosition(
            condition_id=condition_id,
            market_name=question[:50],
            entry_time=datetime.now(),
            estimated_resolution=self.estimate_resolution_time(),
            cost=total_cost,
            payout=payout,
            profit=profit,
            yes_price=yes_price,
            no_price=no_price,
        )
        self.open_positions[condition_id] = position

        # Update stats
        self.stats.total_trades += 1
        self.stats.opportunities_traded += 1
        self.stats.total_volume += trade_size

        if profit > self.stats.largest_profit:
            self.stats.largest_profit = profit

        # Calculate time until resolution for display
        time_until = (position.estimated_resolution - datetime.now()).total_seconds()
        sim_hours = (time_until * self.time_acceleration) / 3600

        # Log the trade
        logger.success(
            f"[TRADE #{self.stats.total_trades}] "
            f"Cost: ${total_cost:.2f} | "
            f"Profit: ${profit:.4f} ({profit_margin*100:.3f}%) | "
            f"Resolves in ~{sim_hours:.1f}h (sim)"
        )
        logger.info(
            f"  Market: {question[:55]} | "
            f"Available: ${self.available_balance:.2f} | "
            f"Locked: ${self.locked_capital:.2f}"
        )

    def print_summary(self):
        """Print realistic simulation summary."""
        # Final resolution check
        print("")
        print("=" * 70)
        print("REALISTIC SIMULATION SUMMARY")
        print("=" * 70)
        print("")
        print(f"Runtime:              {self.stats.runtime_minutes:.1f} minutes")
        print(f"Time Acceleration:    {self.time_acceleration}x (1 min = {self.time_acceleration} sim minutes)")
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
        print(f"Unrealized Profit:    ${self.unrealized_profit:.4f} (in {len(self.open_positions)} open positions)")
        print(f"Total Profit:         ${self.stats.realized_profit + self.unrealized_profit:.4f}")
        print("")

        print("-" * 70)
        print("TRADE STATISTICS")
        print("-" * 70)
        print(f"Unique Markets Traded: {len(self.traded_markets)}")
        print(f"Open Positions:        {len(self.open_positions)}")
        print(f"Resolved Trades:       {self.stats.resolved_trades}")
        print(f"Total Volume:          ${self.stats.total_volume:.2f}")
        print(f"Largest Single Profit: ${self.stats.largest_profit:.4f}")
        print("")

        print("-" * 70)
        print("OPPORTUNITY TRACKING")
        print("-" * 70)
        print(f"Opportunities Seen:          {self.stats.opportunities_seen}")
        print(f"Opportunities Traded:        {self.stats.opportunities_traded}")
        print(f"Skipped (already traded):    {self.stats.opportunities_skipped_duplicate}")
        print(f"Skipped (no capital):        {self.stats.opportunities_skipped_no_capital}")
        print("")

        # Show open positions
        if self.open_positions:
            print("-" * 70)
            print(f"OPEN POSITIONS ({len(self.open_positions)})")
            print("-" * 70)
            for pos in list(self.open_positions.values())[:10]:
                time_left = (pos.estimated_resolution - datetime.now()).total_seconds()
                sim_hours_left = max(0, (time_left * self.time_acceleration) / 3600)
                print(
                    f"  ${pos.profit:.4f} profit | "
                    f"~{sim_hours_left:.1f}h left | "
                    f"{pos.market_name[:40]}"
                )
            if len(self.open_positions) > 10:
                print(f"  ... and {len(self.open_positions) - 10} more")

        # Show recent resolved trades
        if self.resolved_trades:
            print("")
            print("-" * 70)
            print(f"RECENT RESOLVED TRADES ({len(self.resolved_trades)} total)")
            print("-" * 70)
            for trade in self.resolved_trades[-5:]:
                print(
                    f"  +${trade.profit:.4f} | "
                    f"Held {trade.hold_time_hours:.1f}h | "
                    f"{trade.market_name[:40]}"
                )

        print("")
        print("=" * 70)

        # Realistic projections
        if self.stats.runtime_minutes > 0:
            total_profit = self.stats.realized_profit + self.unrealized_profit
            unique_opps = len(self.traded_markets)

            if unique_opps > 0:
                avg_profit_per_trade = total_profit / unique_opps

                print("")
                print("REALISTIC ANALYSIS")
                print("-" * 70)
                print(f"Unique opportunities found:   {unique_opps} in {self.stats.runtime_minutes:.1f} min")
                print(f"Average profit per trade:     ${avg_profit_per_trade:.4f}")
                print(f"Capital utilization:          {(self.locked_capital / self.stats.starting_balance) * 100:.1f}%")
                print("")
                print("NOTE: In real trading:")
                print("  - Each market can only be traded ONCE")
                print("  - Capital is locked until market resolution (days/weeks)")
                print("  - New opportunities require available capital")
                print("  - Profits compound only AFTER positions resolve")

        print("=" * 70)


async def run_simulation(starting_balance: float, duration_minutes: Optional[int] = None):
    """Run the REALISTIC arbitrage simulation."""
    from src.api.gamma_api import GammaAPI
    from src.monitoring.realtime_scanner import RealtimeArbitrageScanner
    from src.config import config

    print("")
    print("=" * 70)
    print("POLYMARKET ARBITRAGE SIMULATOR - REALISTIC MODE")
    print("=" * 70)
    print("")
    print("REALISTIC CONSTRAINTS:")
    print("  [1] Each market can only be traded ONCE")
    print("  [2] Capital is LOCKED until market resolution")
    print("  [3] Only available balance can be used for new trades")
    print("  [4] Time is accelerated 60x (1 min real = 1 hour simulated)")
    print("")
    print(f"Starting Balance: ${starting_balance:.2f}")
    print(f"Min Profit Threshold: {config.trading.min_profit_margin * 100:.3f}%")
    if duration_minutes:
        print(f"Duration: {duration_minutes} minutes")
    else:
        print("Duration: Until Ctrl+C")
    print("")
    print("-" * 70)

    # Initialize simulator
    simulator = RealisticArbitrageSimulator(starting_balance=starting_balance)

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

    # Initialize scanner
    scanner = RealtimeArbitrageScanner()
    scanner.set_execution_callback(simulator.on_opportunity)

    # Load markets
    await scanner.load_markets(markets)

    # Initial scan - process ALL opportunities found (they're unique)
    initial_opps = scanner.do_full_scan()
    if initial_opps:
        logger.info(f"Initial scan found {len(initial_opps)} opportunities - processing all unique ones...")
        for opp in initial_opps:
            await simulator.on_opportunity(opp)
    else:
        logger.info("No opportunities in initial scan - monitoring for new ones...")

    print("")
    logger.info("Starting real-time monitoring... Press Ctrl+C to stop")
    print("")

    # Start tasks
    async def print_status():
        """Print status every 30 seconds."""
        while True:
            await asyncio.sleep(30)
            # Check resolutions
            await simulator.check_resolutions()

            stats = scanner.get_stats()
            logger.info(
                f"[STATUS] Unique Trades: {len(simulator.traded_markets)} | "
                f"Open: {len(simulator.open_positions)} | "
                f"Resolved: {simulator.stats.resolved_trades} | "
                f"Available: ${simulator.available_balance:.2f} | "
                f"Locked: ${simulator.locked_capital:.2f}"
            )

    async def resolution_checker():
        """Check for resolutions every 10 seconds."""
        while True:
            await asyncio.sleep(10)
            await simulator.check_resolutions()

    status_task = asyncio.create_task(print_status())
    resolution_task = asyncio.create_task(resolution_checker())

    # Run scanner with optional timeout
    try:
        if duration_minutes:
            await asyncio.wait_for(
                scanner.run(),
                timeout=duration_minutes * 60
            )
        else:
            await scanner.run()
    except asyncio.TimeoutError:
        logger.info(f"Simulation completed after {duration_minutes} minutes")
    except KeyboardInterrupt:
        logger.info("Simulation stopped by user")
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

        # Final resolution check
        await simulator.check_resolutions()
        simulator.print_summary()


def main():
    parser = argparse.ArgumentParser(description="Polymarket Arbitrage Simulator (Realistic)")
    parser.add_argument(
        "--balance",
        type=float,
        default=100.0,
        help="Starting balance in USDC (default: 100)"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=None,
        help="Run for N minutes (default: runs until Ctrl+C)"
    )

    args = parser.parse_args()

    try:
        asyncio.run(run_simulation(args.balance, args.duration))
    except KeyboardInterrupt:
        print("\nSimulation ended.")


if __name__ == "__main__":
    main()
