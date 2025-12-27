"""
On-Chain Monitor for Polymarket - Direct Blockchain Intelligence

Monitors Polygon blockchain for:
- OrderFilled events (trade execution)
- PositionsSplit events (token minting - new liquidity)
- PositionsMerge events (token burning - positions closing)
- PositionsConverted events (smart money rebalancing)

This gives visibility that API-only bots miss:
- Every single trade (not just visible orders)
- Whale wallet tracking
- Market dynamics through minting/burning patterns
"""

import asyncio
from typing import Any, Callable, Dict, List, Optional, Set
from datetime import datetime
from dataclasses import dataclass, field
from loguru import logger

try:
    from web3 import Web3
    from web3.middleware import ExtraDataToPOAMiddleware
    WEB3_AVAILABLE = True
except ImportError:
    WEB3_AVAILABLE = False
    logger.warning("web3 not installed. On-chain monitoring disabled. Install with: pip install web3")


@dataclass
class PolymarketContracts:
    """Polymarket smart contract addresses on Polygon."""
    # Core trading contracts
    CTF_EXCHANGE: str = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"  # Binary YES/NO
    NEGRISK_EXCHANGE: str = "0xC5d563A36AE78145C45a50134d48A1215220f80a"  # Multi-outcome

    # Token management
    CONDITIONAL_TOKENS: str = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"
    NEGRISK_ADAPTER: str = "0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296"

    # Collateral
    USDC_E: str = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"

    # Oracle
    UMA_ORACLE: str = "0x6A9D222616C90FcA5754cd1333cFD9b7fb6a4F74"


@dataclass
class EventSignatures:
    """Keccak256 hashes of event signatures for log filtering."""
    ORDER_FILLED: str = "0xd0a08e8c493f9c94f29311604c9de1b4e8c8d4c06bd0c789af57f2d65bfec0f6"
    ORDERS_MATCHED: str = "0x63bf4d16b7fa898ef4c4b2b6d90fd201e9c56313b65638af6088d149d2ce956c"
    POSITIONS_SPLIT: str = "0xbbed930dbfb7907ae2d60ddf78345610214f26419a0128df39b6cc3d9e5df9b0"
    POSITIONS_MERGE: str = "0xba33ac50d8894676597e6e35dc09cff59854708b642cd069d21eb9c7ca072a04"
    POSITIONS_CONVERTED: str = "0xb03d19dddbc72a87e735ff0ea3b57bef133ebe44e1894284916a84044deb367e"


@dataclass
class OnChainTrade:
    """Decoded trade from OrderFilled event."""
    order_hash: str
    maker: str
    taker: str
    maker_asset_id: int
    taker_asset_id: int
    maker_amount: float
    taker_amount: float
    fee: float
    trade_type: str  # 'BUY' or 'SELL'
    price: float
    token_amount: float
    block_number: int
    tx_hash: str
    timestamp: Optional[datetime] = None


@dataclass
class MarketDynamics:
    """Market health metrics from on-chain data."""
    total_minted_usdc: float
    total_burned_usdc: float
    net_flow: float
    market_growing: bool
    liquidity_health: str  # 'healthy' or 'contracting'
    lookback_blocks: int


class OnChainMonitor:
    """
    Direct blockchain monitoring for Polymarket intelligence.

    Features:
    - Track all trades on-chain (not just visible orders)
    - Detect whale activity by wallet
    - Monitor minting/burning for market dynamics
    - Identify smart money through position conversions
    """

    # Default RPC endpoints (use private RPC for production)
    DEFAULT_RPCS = [
        "https://polygon-rpc.com/",
        "https://rpc-mainnet.matic.quiknode.pro",
        "https://polygon-mainnet.public.blastapi.io",
    ]

    def __init__(self, rpc_url: Optional[str] = None):
        """
        Initialize on-chain monitor.

        Args:
            rpc_url: Polygon RPC endpoint (private recommended for production)
        """
        if not WEB3_AVAILABLE:
            self.web3 = None
            self.connected = False
            return

        self.rpc_url = rpc_url or self.DEFAULT_RPCS[0]
        self.contracts = PolymarketContracts()
        self.events = EventSignatures()

        # Initialize Web3
        self.web3 = Web3(Web3.HTTPProvider(self.rpc_url))

        # Polygon is a POA chain, need this middleware
        try:
            self.web3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
        except Exception:
            pass  # Already injected or not needed

        self.connected = self.web3.is_connected()

        if self.connected:
            logger.info(f"On-chain monitor connected to Polygon RPC")
            logger.info(f"Current block: {self.web3.eth.block_number}")
        else:
            logger.warning(f"Failed to connect to Polygon RPC: {self.rpc_url}")

        # Cache for decoded events
        self._trade_cache: List[OnChainTrade] = []
        self._whale_wallets: Set[str] = set()

    @property
    def current_block(self) -> int:
        """Get current block number."""
        if not self.connected:
            return 0
        return self.web3.eth.block_number

    def get_recent_trades(
        self,
        from_block: Optional[int] = None,
        to_block: Optional[int] = None,
        market_type: str = "binary",
        lookback_blocks: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Fetch all OrderFilled events for trade analysis.

        Args:
            from_block: Starting block (default: current - lookback_blocks)
            to_block: Ending block (default: current)
            market_type: 'binary' or 'multi' determines which exchange to query
            lookback_blocks: How far back to look if from_block not specified

        Returns:
            List of raw log entries
        """
        if not self.connected:
            return []

        current = self.current_block
        from_block = from_block or (current - lookback_blocks)
        to_block = to_block or current

        target = (
            self.contracts.CTF_EXCHANGE if market_type == "binary"
            else self.contracts.NEGRISK_EXCHANGE
        )

        try:
            logs = self.web3.eth.get_logs({
                'fromBlock': from_block,
                'toBlock': to_block,
                'address': Web3.to_checksum_address(target),
                'topics': [self.events.ORDER_FILLED]
            })
            return logs
        except Exception as e:
            logger.error(f"Error fetching trade logs: {e}")
            return []

    def get_token_minting(
        self,
        from_block: Optional[int] = None,
        to_block: Optional[int] = None,
        lookback_blocks: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Detect new token pairs being created (fresh liquidity entering).

        PositionsSplit = Someone is splitting USDC into YES+NO tokens.
        High minting = New money entering the market.
        """
        if not self.connected:
            return []

        current = self.current_block
        from_block = from_block or (current - lookback_blocks)
        to_block = to_block or current

        try:
            logs = self.web3.eth.get_logs({
                'fromBlock': from_block,
                'toBlock': to_block,
                'address': Web3.to_checksum_address(self.contracts.NEGRISK_ADAPTER),
                'topics': [self.events.POSITIONS_SPLIT]
            })
            return logs
        except Exception as e:
            logger.error(f"Error fetching minting logs: {e}")
            return []

    def get_token_burning(
        self,
        from_block: Optional[int] = None,
        to_block: Optional[int] = None,
        lookback_blocks: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Detect token merges (positions being closed, collateral released).

        PositionsMerge = Someone is combining YES+NO back to USDC.
        High burning = Positions closing, may signal resolution approaching.
        """
        if not self.connected:
            return []

        current = self.current_block
        from_block = from_block or (current - lookback_blocks)
        to_block = to_block or current

        try:
            logs = self.web3.eth.get_logs({
                'fromBlock': from_block,
                'toBlock': to_block,
                'address': Web3.to_checksum_address(self.contracts.NEGRISK_ADAPTER),
                'topics': [self.events.POSITIONS_MERGE]
            })
            return logs
        except Exception as e:
            logger.error(f"Error fetching burning logs: {e}")
            return []

    def get_position_conversions(
        self,
        from_block: Optional[int] = None,
        to_block: Optional[int] = None,
        lookback_blocks: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Track PositionsConverted events for smart money signals.

        Conversions reveal sophisticated trader rebalancing:
        NO_A + NO_B -> YES_C + USDC

        This signals:
        - Informed conviction on outcome C
        - Arbitrage execution
        - Capital efficiency moves
        """
        if not self.connected:
            return []

        current = self.current_block
        from_block = from_block or (current - lookback_blocks)
        to_block = to_block or current

        try:
            logs = self.web3.eth.get_logs({
                'fromBlock': from_block,
                'toBlock': to_block,
                'address': Web3.to_checksum_address(self.contracts.NEGRISK_ADAPTER),
                'topics': [self.events.POSITIONS_CONVERTED]
            })
            return logs
        except Exception as e:
            logger.error(f"Error fetching conversion logs: {e}")
            return []

    def track_wallet(
        self,
        wallet_address: str,
        from_block: Optional[int] = None,
        to_block: Optional[int] = None,
        lookback_blocks: int = 5000
    ) -> List[Dict[str, Any]]:
        """
        Monitor a specific wallet's trading activity on-chain.

        Use this to:
        - Verify leaderboard traders are actually profitable
        - Understand their strategy (which markets, timing, sizing)
        - Detect when they enter new positions

        Args:
            wallet_address: Ethereum address to track
            lookback_blocks: How far back to look (5000 blocks ~ 3 hours)
        """
        if not self.connected:
            return []

        current = self.current_block
        from_block = from_block or (current - lookback_blocks)
        to_block = to_block or current

        # Format wallet address for topic filtering
        wallet_topic = "0x" + wallet_address[2:].lower().zfill(64)

        try:
            # Get trades where wallet was maker
            maker_logs = self.web3.eth.get_logs({
                'fromBlock': from_block,
                'toBlock': to_block,
                'address': Web3.to_checksum_address(self.contracts.NEGRISK_EXCHANGE),
                'topics': [
                    self.events.ORDER_FILLED,
                    None,  # orderHash - any
                    wallet_topic  # maker address
                ]
            })

            # Get trades where wallet was taker
            taker_logs = self.web3.eth.get_logs({
                'fromBlock': from_block,
                'toBlock': to_block,
                'address': Web3.to_checksum_address(self.contracts.NEGRISK_EXCHANGE),
                'topics': [
                    self.events.ORDER_FILLED,
                    None,  # orderHash - any
                    None,  # maker - any
                    wallet_topic  # taker address
                ]
            })

            # Tag logs with role
            all_logs = []
            for log in maker_logs:
                log_dict = dict(log)
                log_dict['wallet_role'] = 'maker'
                all_logs.append(log_dict)
            for log in taker_logs:
                log_dict = dict(log)
                log_dict['wallet_role'] = 'taker'
                all_logs.append(log_dict)

            return sorted(all_logs, key=lambda x: x.get('blockNumber', 0))

        except Exception as e:
            logger.error(f"Error tracking wallet {wallet_address[:10]}...: {e}")
            return []

    def analyze_market_dynamics(self, lookback_blocks: int = 1000) -> MarketDynamics:
        """
        Understand market health through minting/burning patterns.

        Key insights:
        - High minting = New money entering, increased liquidity
        - High burning = Positions closing, resolution may be near
        - Minting > Burning = Market growing
        - Burning > Minting = Market contracting
        """
        mints = self.get_token_minting(lookback_blocks=lookback_blocks)
        burns = self.get_token_burning(lookback_blocks=lookback_blocks)

        # Simple volume estimation (actual decoding would need ABI)
        # Each event represents some amount of value
        total_minted = len(mints) * 1000  # Rough estimate per event
        total_burned = len(burns) * 1000

        net_flow = total_minted - total_burned
        market_growing = total_minted > total_burned

        if total_minted > total_burned * 0.8:
            health = "healthy"
        else:
            health = "contracting"

        return MarketDynamics(
            total_minted_usdc=total_minted,
            total_burned_usdc=total_burned,
            net_flow=net_flow,
            market_growing=market_growing,
            liquidity_health=health,
            lookback_blocks=lookback_blocks
        )

    def detect_whale_trades(
        self,
        min_size_usd: float = 10000,
        lookback_blocks: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Detect large trades (whale activity) in recent blocks.

        Whale activity can signal:
        - Informed trading (they know something)
        - Market manipulation attempt
        - Large position entries to follow

        Args:
            min_size_usd: Minimum trade size to consider "whale"
            lookback_blocks: How far back to look
        """
        if not self.connected:
            return []

        # Get recent trades from both exchanges
        binary_trades = self.get_recent_trades(
            lookback_blocks=lookback_blocks,
            market_type="binary"
        )
        multi_trades = self.get_recent_trades(
            lookback_blocks=lookback_blocks,
            market_type="multi"
        )

        whale_trades = []

        # Filter for large trades (rough estimation without full decoding)
        # In a full implementation, you'd decode the event data
        for trade in binary_trades + multi_trades:
            # The data field contains the trade details
            # This is a simplified check - real implementation would decode
            if len(trade.get('data', '')) > 200:  # Larger data = larger trade
                whale_trades.append({
                    'block': trade.get('blockNumber'),
                    'tx_hash': trade.get('transactionHash', b'').hex() if isinstance(trade.get('transactionHash'), bytes) else str(trade.get('transactionHash', '')),
                    'exchange': 'binary' if trade in binary_trades else 'multi',
                    'estimated_large': True
                })

        return whale_trades

    def add_tracked_wallet(self, address: str):
        """Add a wallet to the tracking set."""
        self._whale_wallets.add(address.lower())
        logger.info(f"Now tracking wallet: {address[:10]}...")

    def remove_tracked_wallet(self, address: str):
        """Remove a wallet from tracking."""
        self._whale_wallets.discard(address.lower())

    def get_tracked_wallets(self) -> Set[str]:
        """Get all tracked wallet addresses."""
        return self._whale_wallets.copy()

    def calculate_trade_flow(self, lookback_blocks: int = 50) -> Dict[str, Any]:
        """
        Analyze recent trade flow for market sentiment.

        Returns:
            Dict with buy/sell volume and flow imbalance
        """
        trades = self.get_recent_trades(lookback_blocks=lookback_blocks)

        # Simplified analysis - count trades as proxy
        # Full implementation would decode amounts
        trade_count = len(trades)

        return {
            'trade_count': trade_count,
            'lookback_blocks': lookback_blocks,
            'blocks_per_trade': lookback_blocks / max(trade_count, 1),
            'activity_level': 'high' if trade_count > 50 else 'medium' if trade_count > 20 else 'low'
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get on-chain monitor statistics."""
        return {
            'connected': self.connected,
            'rpc_url': self.rpc_url,
            'current_block': self.current_block if self.connected else 0,
            'tracked_wallets': len(self._whale_wallets),
            'contracts': {
                'ctf_exchange': self.contracts.CTF_EXCHANGE,
                'negrisk_exchange': self.contracts.NEGRISK_EXCHANGE,
            }
        }


# Convenience function for quick setup
def create_onchain_monitor(rpc_url: Optional[str] = None) -> OnChainMonitor:
    """Create an on-chain monitor instance."""
    return OnChainMonitor(rpc_url=rpc_url)
