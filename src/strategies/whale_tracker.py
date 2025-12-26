"""
Whale/Smart Money Tracking Module.

Monitors top traders' positions and generates copy trading signals.
Based on research showing $10k+/month possible by following multiple
top performers moving in the same direction.

Key features:
- Tracks target wallet positions every ~4 seconds
- Detects position changes and consensus among whales
- Generates trade signals when multiple whales agree
- Supports counter-trading negative P&L wallets
"""

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Set
from loguru import logger

from src.api.gamma_api import GammaAPI
from src.config import config


@dataclass
class WhalePosition:
    """A whale's position in a specific market."""
    wallet: str
    market_id: str
    market_question: str
    outcome: str
    shares: float
    value: float
    avg_price: float
    last_updated: datetime


@dataclass
class WhaleActivity:
    """Detected whale activity (position change)."""
    wallet: str
    market_id: str
    market_question: str
    outcome: str
    action: str  # "BUY" or "SELL"
    size_change: float
    new_size: float
    detected_at: datetime


@dataclass
class ConsensusSignal:
    """Signal generated when multiple whales agree."""
    market_id: str
    market_question: str
    outcome: str
    action: str
    whales_agreeing: List[str]
    consensus_strength: float  # 0-1
    total_value: float
    generated_at: datetime


class WhaleTracker:
    """
    Tracks whale wallets and generates copy trading signals.
    """

    def __init__(
        self,
        target_wallets: Optional[List[str]] = None,
        on_signal: Optional[Callable[[ConsensusSignal], None]] = None,
    ):
        """
        Initialize whale tracker.

        Args:
            target_wallets: List of wallet addresses to track
            on_signal: Callback for consensus signals
        """
        self.target_wallets = target_wallets or config.whale_tracking.get_target_wallet_list()
        self.on_signal = on_signal
        self.poll_interval = config.whale_tracking.poll_interval
        self.min_consensus = config.whale_tracking.min_whale_consensus

        # State
        self._positions: Dict[str, Dict[str, WhalePosition]] = defaultdict(dict)  # wallet -> market_id -> position
        self._activities: List[WhaleActivity] = []
        self._signals: List[ConsensusSignal] = []
        self._running = False

        # API client
        self.gamma_api = GammaAPI()

        logger.info(f"Whale tracker initialized with {len(self.target_wallets)} target wallets")

    async def start(self):
        """Start whale tracking loop."""
        if not self.target_wallets:
            logger.warning("No target wallets configured for whale tracking")
            return

        self._running = True
        logger.info("Starting whale tracker...")

        while self._running:
            try:
                await self._poll_all_wallets()
                await self._detect_consensus()
                await asyncio.sleep(self.poll_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Whale tracker error: {e}")
                await asyncio.sleep(self.poll_interval)

    async def stop(self):
        """Stop whale tracking."""
        self._running = False
        await self.gamma_api.close()
        logger.info("Whale tracker stopped")

    async def _poll_all_wallets(self):
        """Poll positions for all target wallets."""
        tasks = [self._poll_wallet(wallet) for wallet in self.target_wallets]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _poll_wallet(self, wallet: str):
        """Poll and update positions for a single wallet."""
        try:
            positions = await self.gamma_api.get_user_positions(wallet)

            if not positions:
                return

            now = datetime.now()
            previous_positions = self._positions.get(wallet, {})

            for pos in positions:
                market_id = pos.get("conditionId", "")
                outcome = pos.get("outcome", "")
                shares = float(pos.get("shares", 0) or 0)
                value = float(pos.get("value", 0) or 0)
                avg_price = float(pos.get("avgPrice", 0) or 0)
                question = pos.get("question", "Unknown")

                key = f"{market_id}-{outcome}"

                # Check for position change
                prev = previous_positions.get(key)
                if prev:
                    if abs(shares - prev.shares) > 0.01:  # Meaningful change
                        action = "BUY" if shares > prev.shares else "SELL"
                        size_change = abs(shares - prev.shares)

                        activity = WhaleActivity(
                            wallet=wallet,
                            market_id=market_id,
                            market_question=question,
                            outcome=outcome,
                            action=action,
                            size_change=size_change,
                            new_size=shares,
                            detected_at=now,
                        )
                        self._activities.append(activity)
                        logger.info(
                            f"[WHALE] {wallet[:10]}... {action} {size_change:.2f} "
                            f"{outcome} in {question[:40]}..."
                        )

                # Update position
                self._positions[wallet][key] = WhalePosition(
                    wallet=wallet,
                    market_id=market_id,
                    market_question=question,
                    outcome=outcome,
                    shares=shares,
                    value=value,
                    avg_price=avg_price,
                    last_updated=now,
                )

        except Exception as e:
            logger.debug(f"Error polling wallet {wallet[:10]}...: {e}")

    async def _detect_consensus(self):
        """Detect consensus among whales for the same market/direction."""
        # Look at recent activities (last 60 seconds)
        cutoff = datetime.now() - timedelta(seconds=60)
        recent = [a for a in self._activities if a.detected_at > cutoff]

        if not recent:
            return

        # Group by market+outcome+action
        groups: Dict[str, List[WhaleActivity]] = defaultdict(list)
        for activity in recent:
            key = f"{activity.market_id}-{activity.outcome}-{activity.action}"
            groups[key].append(activity)

        # Check for consensus
        for key, activities in groups.items():
            unique_wallets = list(set(a.wallet for a in activities))

            if len(unique_wallets) >= self.min_consensus:
                # Consensus detected!
                first = activities[0]
                total_value = sum(a.size_change * (a.new_size / max(1, a.size_change)) for a in activities)

                signal = ConsensusSignal(
                    market_id=first.market_id,
                    market_question=first.market_question,
                    outcome=first.outcome,
                    action=first.action,
                    whales_agreeing=unique_wallets,
                    consensus_strength=len(unique_wallets) / len(self.target_wallets),
                    total_value=total_value,
                    generated_at=datetime.now(),
                )

                self._signals.append(signal)
                logger.info(
                    f"[CONSENSUS] {len(unique_wallets)} whales {first.action} "
                    f"{first.outcome} in {first.market_question[:40]}..."
                )

                if self.on_signal:
                    try:
                        if asyncio.iscoroutinefunction(self.on_signal):
                            await self.on_signal(signal)
                        else:
                            self.on_signal(signal)
                    except Exception as e:
                        logger.error(f"Signal callback error: {e}")

                # Clear activities that contributed to this signal
                self._activities = [
                    a for a in self._activities
                    if f"{a.market_id}-{a.outcome}-{a.action}" != key
                ]

    def add_target_wallet(self, wallet: str):
        """Add a wallet to track."""
        if wallet not in self.target_wallets:
            self.target_wallets.append(wallet)
            logger.info(f"Added whale wallet: {wallet[:10]}...")

    def remove_target_wallet(self, wallet: str):
        """Remove a wallet from tracking."""
        if wallet in self.target_wallets:
            self.target_wallets.remove(wallet)
            if wallet in self._positions:
                del self._positions[wallet]
            logger.info(f"Removed whale wallet: {wallet[:10]}...")

    def get_whale_positions(self, wallet: str) -> List[WhalePosition]:
        """Get current positions for a whale."""
        return list(self._positions.get(wallet, {}).values())

    def get_all_positions(self) -> Dict[str, List[WhalePosition]]:
        """Get all tracked positions."""
        return {
            wallet: list(positions.values())
            for wallet, positions in self._positions.items()
        }

    def get_recent_activities(self, minutes: int = 5) -> List[WhaleActivity]:
        """Get recent whale activities."""
        cutoff = datetime.now() - timedelta(minutes=minutes)
        return [a for a in self._activities if a.detected_at > cutoff]

    def get_recent_signals(self, minutes: int = 30) -> List[ConsensusSignal]:
        """Get recent consensus signals."""
        cutoff = datetime.now() - timedelta(minutes=minutes)
        return [s for s in self._signals if s.generated_at > cutoff]

    def get_stats(self) -> Dict[str, Any]:
        """Get tracker statistics."""
        return {
            "target_wallets": len(self.target_wallets),
            "tracked_positions": sum(len(p) for p in self._positions.values()),
            "recent_activities": len(self.get_recent_activities()),
            "consensus_signals": len(self.get_recent_signals()),
            "running": self._running,
        }


# Helper functions

async def find_top_traders(
    min_profit: float = 50000,
    min_trades: int = 500,
    limit: int = 20,
) -> List[str]:
    """
    Find top traders suitable for tracking.

    Args:
        min_profit: Minimum all-time profit
        min_trades: Minimum number of trades
        limit: Maximum wallets to return

    Returns:
        List of wallet addresses
    """
    from src.utils.leaderboard import LeaderboardScraper

    scraper = LeaderboardScraper()
    try:
        arb_traders = await scraper.find_arbitrage_traders(
            min_trades=min_trades,
            min_profit=min_profit,
        )
        return [t.address for t in arb_traders[:limit]]
    finally:
        await scraper.close()


async def analyze_whale_strategy(wallet: str) -> Dict[str, Any]:
    """
    Analyze a whale's trading strategy.

    Args:
        wallet: Wallet address to analyze

    Returns:
        Strategy analysis
    """
    from src.utils.leaderboard import LeaderboardScraper

    scraper = LeaderboardScraper()
    try:
        return await scraper.analyze_trader(wallet)
    finally:
        await scraper.close()
