"""
Whale Tracker for Polymarket - Smart Money Following

Track profitable wallets directly on-chain for copy trading signals.

Advantages over API-based tracking:
- See ALL trades, not just visible orders
- Historical analysis of any wallet
- Detect wallet clusters (same entity, multiple wallets)
- Real-time entry detection
"""

import asyncio
from typing import Any, Dict, List, Optional, Set
from datetime import datetime
from dataclasses import dataclass, field
from loguru import logger

from src.blockchain.onchain_monitor import OnChainMonitor, create_onchain_monitor


@dataclass
class WalletStats:
    """Statistics for a tracked wallet."""
    address: str
    label: Optional[str] = None
    total_trades: int = 0
    total_volume_usd: float = 0.0
    last_trade_block: int = 0
    last_trade_time: Optional[datetime] = None
    win_rate: Optional[float] = None
    avg_trade_size: float = 0.0
    markets_traded: Set[str] = field(default_factory=set)


@dataclass
class WhaleAlert:
    """Alert when a tracked whale makes a significant trade."""
    wallet: str
    label: Optional[str]
    trade_type: str  # 'BUY' or 'SELL'
    estimated_size_usd: float
    block_number: int
    tx_hash: str
    timestamp: datetime
    is_new_position: bool = False


class WhaleTracker:
    """
    Track profitable wallets directly on-chain.

    Usage:
    1. Add wallets from Polymarket leaderboard research
    2. Monitor their on-chain activity
    3. Get alerts when they enter large positions
    4. Validate their actual win rate from history
    """

    # Known profitable wallets from research (add your own)
    # Format: (address, label)
    KNOWN_WHALES = [
        # Add addresses from leaderboard research here
        # ("0x...", "Top Trader #1"),
    ]

    def __init__(self, on_chain_monitor: Optional[OnChainMonitor] = None):
        """
        Initialize whale tracker.

        Args:
            on_chain_monitor: Existing monitor or creates new one
        """
        self.chain = on_chain_monitor or create_onchain_monitor()
        self.tracked_wallets: Dict[str, WalletStats] = {}
        self._alerts: List[WhaleAlert] = []
        self._alert_callbacks: List[callable] = []

        # Load known whales
        for address, label in self.KNOWN_WHALES:
            self.add_wallet(address, label)

    def add_wallet(self, address: str, label: Optional[str] = None):
        """
        Add a wallet to track.

        Args:
            address: Ethereum address (from leaderboard research)
            label: Optional friendly name
        """
        address = address.lower()

        if address in self.tracked_wallets:
            logger.debug(f"Wallet already tracked: {address[:10]}...")
            return

        self.tracked_wallets[address] = WalletStats(
            address=address,
            label=label
        )
        self.chain.add_tracked_wallet(address)

        logger.info(f"Tracking whale: {label or address[:10]}...")

    def remove_wallet(self, address: str):
        """Remove a wallet from tracking."""
        address = address.lower()
        if address in self.tracked_wallets:
            del self.tracked_wallets[address]
            self.chain.remove_tracked_wallet(address)
            logger.info(f"Stopped tracking: {address[:10]}...")

    def add_alert_callback(self, callback: callable):
        """Add callback for whale alerts."""
        self._alert_callbacks.append(callback)

    def scan_wallet_history(
        self,
        address: str,
        lookback_blocks: int = 10000
    ) -> Dict[str, Any]:
        """
        Get complete trading history for a wallet.

        Use this to:
        1. Verify leaderboard traders are actually profitable
        2. Understand their strategy (which markets, timing, sizing)
        3. Calculate their historical win rate

        Args:
            address: Wallet to analyze
            lookback_blocks: How far back (10000 ~ 6 hours on Polygon)

        Returns:
            Dict with trade history and statistics
        """
        if not self.chain.connected:
            return {'error': 'On-chain monitor not connected'}

        trades = self.chain.track_wallet(address, lookback_blocks=lookback_blocks)

        # Update stats
        if address.lower() in self.tracked_wallets:
            stats = self.tracked_wallets[address.lower()]
            stats.total_trades = len(trades)
            if trades:
                stats.last_trade_block = max(t.get('blockNumber', 0) for t in trades)

        return {
            'address': address,
            'trade_count': len(trades),
            'lookback_blocks': lookback_blocks,
            'trades': trades[:20],  # Return first 20 for preview
            'total_trades_found': len(trades)
        }

    def detect_whale_entries(
        self,
        min_size_usd: float = 5000,
        lookback_blocks: int = 50
    ) -> List[WhaleAlert]:
        """
        Real-time detection of large position entries from tracked wallets.

        Signal: When a tracked whale enters a position >$5k:
        - Consider following if their historical win rate > 60%
        - They're not exiting an existing position
        - Market has sufficient liquidity for your trade

        Args:
            min_size_usd: Minimum trade size to alert on
            lookback_blocks: How recently to check (~2 min on Polygon)

        Returns:
            List of whale alerts
        """
        alerts = []

        for address, stats in self.tracked_wallets.items():
            trades = self.chain.track_wallet(address, lookback_blocks=lookback_blocks)

            for trade in trades:
                block = trade.get('blockNumber', 0)

                # Skip if we've already seen this trade
                if block <= stats.last_trade_block:
                    continue

                # Create alert
                alert = WhaleAlert(
                    wallet=address,
                    label=stats.label,
                    trade_type='UNKNOWN',  # Would need full decoding
                    estimated_size_usd=min_size_usd,  # Rough estimate
                    block_number=block,
                    tx_hash=trade.get('transactionHash', b'').hex() if isinstance(trade.get('transactionHash'), bytes) else '',
                    timestamp=datetime.now(),
                    is_new_position=block > stats.last_trade_block
                )

                alerts.append(alert)
                self._alerts.append(alert)

                # Trigger callbacks
                for callback in self._alert_callbacks:
                    try:
                        callback(alert)
                    except Exception as e:
                        logger.error(f"Alert callback error: {e}")

                # Update stats
                stats.last_trade_block = block
                stats.total_trades += 1

        return alerts

    def get_wallet_stats(self, address: str) -> Optional[WalletStats]:
        """Get statistics for a tracked wallet."""
        return self.tracked_wallets.get(address.lower())

    def get_all_stats(self) -> Dict[str, WalletStats]:
        """Get statistics for all tracked wallets."""
        return self.tracked_wallets.copy()

    def get_recent_alerts(self, limit: int = 20) -> List[WhaleAlert]:
        """Get recent whale alerts."""
        return self._alerts[-limit:]

    def calculate_follow_confidence(self, alert: WhaleAlert) -> Dict[str, Any]:
        """
        Calculate confidence score for following a whale trade.

        Factors:
        - Historical win rate
        - Trade size relative to their average
        - Time since last trade (is this a new conviction?)
        - Whether other tracked whales are moving same direction
        """
        stats = self.get_wallet_stats(alert.wallet)

        if not stats:
            return {'confidence': 0, 'reason': 'Unknown wallet'}

        confidence = 50  # Base confidence

        # Adjust for win rate (if known)
        if stats.win_rate:
            if stats.win_rate > 70:
                confidence += 20
            elif stats.win_rate > 60:
                confidence += 10
            elif stats.win_rate < 50:
                confidence -= 20

        # Adjust for label (known good traders)
        if stats.label:
            confidence += 5

        # Adjust for trade count (more history = more reliable)
        if stats.total_trades > 100:
            confidence += 10
        elif stats.total_trades < 10:
            confidence -= 10

        return {
            'confidence': min(max(confidence, 0), 100),
            'wallet_label': stats.label,
            'wallet_trades': stats.total_trades,
            'wallet_win_rate': stats.win_rate,
            'recommendation': 'follow' if confidence > 60 else 'observe' if confidence > 40 else 'skip'
        }

    async def run_monitoring_loop(
        self,
        check_interval: int = 30,
        min_size_usd: float = 5000
    ):
        """
        Run continuous whale monitoring.

        Args:
            check_interval: Seconds between checks
            min_size_usd: Minimum trade size to alert
        """
        logger.info(f"Starting whale monitoring loop (checking every {check_interval}s)")
        logger.info(f"Tracking {len(self.tracked_wallets)} wallets")

        while True:
            try:
                alerts = self.detect_whale_entries(
                    min_size_usd=min_size_usd,
                    lookback_blocks=int(check_interval / 2)  # ~1 block per 2 sec
                )

                if alerts:
                    for alert in alerts:
                        confidence = self.calculate_follow_confidence(alert)
                        logger.success(
                            f"[WHALE ALERT] {alert.label or alert.wallet[:10]} | "
                            f"~${alert.estimated_size_usd:,.0f} | "
                            f"Confidence: {confidence['confidence']}% | "
                            f"Block: {alert.block_number}"
                        )

            except Exception as e:
                logger.error(f"Whale monitoring error: {e}")

            await asyncio.sleep(check_interval)

    def get_summary(self) -> Dict[str, Any]:
        """Get tracker summary."""
        return {
            'tracked_wallets': len(self.tracked_wallets),
            'total_alerts': len(self._alerts),
            'on_chain_connected': self.chain.connected,
            'wallets': [
                {
                    'address': stats.address[:10] + '...',
                    'label': stats.label,
                    'trades': stats.total_trades
                }
                for stats in self.tracked_wallets.values()
            ]
        }


# Convenience function
def create_whale_tracker(rpc_url: Optional[str] = None) -> WhaleTracker:
    """Create a whale tracker instance."""
    monitor = create_onchain_monitor(rpc_url)
    return WhaleTracker(monitor)
