#!/usr/bin/env python3
"""
Polymarket Arbitrage Simulator

Simulates trading with virtual money to test the strategy.
Tracks positions, P&L, and provides realistic metrics.

Usage:
    python simulate.py [--balance 100] [--duration 60]

Options:
    --balance   Starting balance in USDC (default: 100)
    --duration  Run for N minutes (default: runs until Ctrl+C)
"""

import asyncio
import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

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
class SimulatedTrade:
    """Record of a simulated trade."""
    timestamp: datetime
    market: str
    condition_id: str
    yes_price: float
    no_price: float
    total_cost: float
    profit: float
    trade_size: float
    status: str = "executed"


@dataclass
class SimulatorStats:
    """Simulation statistics."""
    start_time: datetime = field(default_factory=datetime.now)
    starting_balance: float = 100.0
    current_balance: float = 100.0
    total_trades: int = 0
    winning_trades: int = 0
    total_profit: float = 0.0
    total_volume: float = 0.0
    opportunities_seen: int = 0
    opportunities_traded: int = 0
    largest_profit: float = 0.0
    trades: List[SimulatedTrade] = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return (self.winning_trades / self.total_trades) * 100

    @property
    def roi(self) -> float:
        if self.starting_balance == 0:
            return 0.0
        return ((self.current_balance - self.starting_balance) / self.starting_balance) * 100

    @property
    def runtime_minutes(self) -> float:
        return (datetime.now() - self.start_time).total_seconds() / 60


class ArbitrageSimulator:
    """
    Simulates arbitrage trading with virtual money.

    Tracks:
    - Virtual balance
    - Each trade's P&L
    - Overall statistics
    - Win rate
    """

    def __init__(self, starting_balance: float = 100.0):
        self.stats = SimulatorStats(
            starting_balance=starting_balance,
            current_balance=starting_balance,
        )

        # Position sizing
        self.max_position_pct = 0.10  # Max 10% per trade
        self.min_trade_size = 1.0     # $1 minimum

        # Track active positions
        self.positions: Dict[str, Dict] = {}

    def calculate_trade_size(self, profit_margin: float) -> float:
        """
        Calculate optimal trade size based on Kelly Criterion.

        For arbitrage (guaranteed profit), we can be more aggressive,
        but we still limit position size for safety.
        """
        max_size = self.stats.current_balance * self.max_position_pct

        # Scale with profit margin (larger margin = larger position)
        # But cap at max_position_pct of balance
        if profit_margin > 0.01:  # > 1% profit
            size = max_size
        elif profit_margin > 0.005:  # > 0.5%
            size = max_size * 0.75
        elif profit_margin > 0.001:  # > 0.1%
            size = max_size * 0.5
        else:
            size = max_size * 0.25

        return max(self.min_trade_size, min(size, max_size))

    async def on_opportunity(self, opp: Dict[str, Any]):
        """
        Handle detected arbitrage opportunity.
        Simulates trade execution.
        """
        self.stats.opportunities_seen += 1

        yes_price = opp["yes_price"]
        no_price = opp["no_price"]
        total = opp["total"]
        profit_margin = opp["profit_margin"]
        question = opp["question"]
        condition_id = opp["condition_id"]

        # Calculate trade size
        trade_size = self.calculate_trade_size(profit_margin)

        # Check if we have enough balance
        if trade_size > self.stats.current_balance:
            logger.warning(f"Insufficient balance for trade: ${trade_size:.2f} > ${self.stats.current_balance:.2f}")
            return

        # Calculate costs and profit
        # In arbitrage: buy YES at yes_price, buy NO at no_price
        # Total cost = trade_size (we spend this much total)
        # We get back $1 per share when market resolves
        # Profit = (1 - total) * trade_size

        total_cost = trade_size * total  # What we pay for YES + NO
        payout = trade_size  # We get $1 per share
        profit = payout - total_cost

        # Execute simulated trade
        self.stats.current_balance -= total_cost  # Pay for positions

        # For simulation, we immediately "resolve" and get payout
        # In reality, you'd hold until resolution
        self.stats.current_balance += payout

        # Record trade
        trade = SimulatedTrade(
            timestamp=datetime.now(),
            market=question[:50],
            condition_id=condition_id,
            yes_price=yes_price,
            no_price=no_price,
            total_cost=total_cost,
            profit=profit,
            trade_size=trade_size,
        )
        self.stats.trades.append(trade)

        # Update stats
        self.stats.total_trades += 1
        self.stats.opportunities_traded += 1
        self.stats.total_profit += profit
        self.stats.total_volume += trade_size

        if profit > 0:
            self.stats.winning_trades += 1
        if profit > self.stats.largest_profit:
            self.stats.largest_profit = profit

        # Log the trade
        logger.success(
            f"SIMULATED TRADE #{self.stats.total_trades} | "
            f"Size: ${trade_size:.2f} | "
            f"Cost: ${total_cost:.4f} | "
            f"Profit: ${profit:.4f} ({profit_margin*100:.3f}%) | "
            f"Balance: ${self.stats.current_balance:.2f}"
        )
        logger.info(f"  Market: {question[:60]}")

    def print_summary(self):
        """Print simulation summary."""
        print("")
        print("=" * 70)
        print("SIMULATION SUMMARY")
        print("=" * 70)
        print("")
        print(f"Runtime:              {self.stats.runtime_minutes:.1f} minutes")
        print(f"Starting Balance:     ${self.stats.starting_balance:.2f}")
        print(f"Current Balance:      ${self.stats.current_balance:.2f}")
        print(f"Total Profit/Loss:    ${self.stats.total_profit:.4f}")
        print(f"ROI:                  {self.stats.roi:.2f}%")
        print("")
        print(f"Opportunities Seen:   {self.stats.opportunities_seen}")
        print(f"Trades Executed:      {self.stats.total_trades}")
        print(f"Win Rate:             {self.stats.win_rate:.1f}%")
        print(f"Total Volume:         ${self.stats.total_volume:.2f}")
        print(f"Largest Single Profit: ${self.stats.largest_profit:.4f}")
        print("")

        if self.stats.trades:
            print("Recent Trades:")
            print("-" * 70)
            for trade in self.stats.trades[-10:]:
                print(
                    f"  {trade.timestamp.strftime('%H:%M:%S')} | "
                    f"${trade.trade_size:.2f} | "
                    f"Profit: ${trade.profit:.4f} | "
                    f"{trade.market[:40]}"
                )

        print("")
        print("=" * 70)

        # Extrapolate daily earnings
        if self.stats.runtime_minutes > 0 and self.stats.total_profit > 0:
            hourly_profit = (self.stats.total_profit / self.stats.runtime_minutes) * 60
            daily_profit = hourly_profit * 24
            print("")
            print("PROJECTED EARNINGS (if run 24/7):")
            print(f"  Hourly:  ${hourly_profit:.4f}")
            print(f"  Daily:   ${daily_profit:.4f}")
            print(f"  Monthly: ${daily_profit * 30:.2f}")
            print("")
            print("Note: Actual results may vary. Markets are more active during US hours.")

        print("=" * 70)


async def run_simulation(starting_balance: float, duration_minutes: Optional[int] = None):
    """Run the arbitrage simulation."""
    from src.api.gamma_api import GammaAPI
    from src.monitoring.realtime_scanner import RealtimeArbitrageScanner
    from src.config import config

    print("")
    print("=" * 70)
    print("POLYMARKET ARBITRAGE SIMULATOR")
    print("=" * 70)
    print("")
    print(f"Starting Balance: ${starting_balance:.2f}")
    print(f"Min Profit Threshold: {config.trading.min_profit_margin * 100:.3f}%")
    if duration_minutes:
        print(f"Duration: {duration_minutes} minutes")
    else:
        print("Duration: Until Ctrl+C")
    print("")
    print("This simulation uses VIRTUAL money to test the strategy.")
    print("No real trades will be executed.")
    print("")
    print("-" * 70)

    # Initialize simulator
    simulator = ArbitrageSimulator(starting_balance=starting_balance)

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

    # Initial scan
    initial_opps = scanner.do_full_scan()
    if initial_opps:
        logger.info(f"Initial scan found {len(initial_opps)} opportunities")
        for opp in initial_opps[:3]:
            # Process initial opportunities
            await simulator.on_opportunity(opp)
    else:
        logger.info("No opportunities in initial scan - monitoring for new ones...")

    print("")
    logger.info("Starting real-time monitoring... Press Ctrl+C to stop")
    print("")

    # Run scanner with optional timeout
    try:
        if duration_minutes:
            # Run for specified duration
            await asyncio.wait_for(
                scanner.run(),
                timeout=duration_minutes * 60
            )
        else:
            # Run until interrupted
            await scanner.run()
    except asyncio.TimeoutError:
        logger.info(f"Simulation completed after {duration_minutes} minutes")
    except KeyboardInterrupt:
        logger.info("Simulation stopped by user")
    finally:
        await scanner.stop()
        simulator.print_summary()


def main():
    parser = argparse.ArgumentParser(description="Polymarket Arbitrage Simulator")
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
