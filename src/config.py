"""
Configuration management for Polymarket Trading Bot.
Loads settings from environment variables and provides defaults.
Includes advanced parameters from optimization research.
"""

import os
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv
from loguru import logger

# Load environment variables
load_dotenv()


@dataclass
class WalletConfig:
    """Wallet and authentication configuration."""
    private_key: str = field(default_factory=lambda: os.getenv("PRIVATE_KEY", ""))
    wallet_address: str = field(default_factory=lambda: os.getenv("WALLET_ADDRESS", ""))
    funder_address: Optional[str] = field(
        default_factory=lambda: os.getenv("FUNDER_ADDRESS") or None
    )
    signature_type: int = field(
        default_factory=lambda: int(os.getenv("SIGNATURE_TYPE", "0"))
    )

    def validate(self) -> bool:
        """Validate wallet configuration."""
        if not self.private_key:
            logger.error("PRIVATE_KEY not set in environment")
            return False
        if not self.wallet_address:
            logger.error("WALLET_ADDRESS not set in environment")
            return False
        return True


@dataclass
class APIConfig:
    """API endpoint configuration with rate limits."""
    clob_host: str = field(
        default_factory=lambda: os.getenv("CLOB_HOST", "https://clob.polymarket.com")
    )
    gamma_api_host: str = field(
        default_factory=lambda: os.getenv("GAMMA_API_HOST", "https://gamma-api.polymarket.com")
    )
    chain_id: int = field(
        default_factory=lambda: int(os.getenv("CHAIN_ID", "137"))
    )
    # WebSocket endpoints
    ws_clob_url: str = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    ws_rtds_url: str = "wss://ws-subscriptions-clob.polymarket.com/ws/rtds"

    # Rate limits (documented from research)
    order_burst_limit: int = 240  # orders/second
    price_requests_per_10s: int = 200
    max_ws_instruments: int = 500  # per connection


@dataclass
class TradingConfig:
    """Trading parameters with advanced thresholds."""
    min_profit_margin: float = field(
        default_factory=lambda: float(os.getenv("MIN_PROFIT_MARGIN", "0.00001"))  # 0.001% - capture micro profits
    )
    max_position_size: float = field(
        default_factory=lambda: float(os.getenv("MAX_POSITION_SIZE", "100"))
    )
    min_liquidity: float = field(
        default_factory=lambda: float(os.getenv("MIN_LIQUIDITY", "100"))  # Lower to find more opportunities
    )
    max_slippage: float = field(
        default_factory=lambda: float(os.getenv("MAX_SLIPPAGE", "0.01"))
    )
    dry_run: bool = field(
        default_factory=lambda: os.getenv("DRY_RUN", "true").lower() == "true"
    )

    # Advanced spread thresholds from research
    # <1% skip, 1-2% caution, 2-3% sweet spot, 3-5% verify, 5%+ check rules
    min_spread_threshold: float = field(
        default_factory=lambda: float(os.getenv("MIN_SPREAD_THRESHOLD", "0.01"))  # Skip <1%
    )
    sweet_spot_spread_min: float = 0.02  # 2%
    sweet_spot_spread_max: float = 0.03  # 3%

    # Dynamic tick sizes (price > 96¢ or < 4¢ uses 0.001)
    high_price_threshold: float = 0.96
    low_price_threshold: float = 0.04
    normal_tick_size: float = 0.01
    extreme_tick_size: float = 0.001


@dataclass
class KellyConfig:
    """Kelly Criterion position sizing configuration."""
    # Capital tiers determine Kelly multiplier
    # $50-100: 0.25x (Quarter-Kelly)
    # $100-500: 0.50x (Half-Kelly)
    # $500+: 0.50-0.75x
    default_kelly_fraction: float = field(
        default_factory=lambda: float(os.getenv("KELLY_FRACTION", "0.25"))
    )
    max_position_percent: float = field(
        default_factory=lambda: float(os.getenv("MAX_POSITION_PERCENT", "0.05"))  # 5% max per position
    )
    max_total_exposure: float = field(
        default_factory=lambda: float(os.getenv("MAX_TOTAL_EXPOSURE", "0.20"))  # 20% max total
    )
    min_trade_size: float = field(
        default_factory=lambda: float(os.getenv("MIN_TRADE_SIZE", "1.0"))  # $1 minimum
    )

    def get_kelly_multiplier(self, bankroll: float) -> float:
        """Get Kelly multiplier based on bankroll size."""
        if bankroll < 100:
            return 0.25  # Quarter-Kelly for $50-100
        elif bankroll < 500:
            return 0.50  # Half-Kelly for $100-500
        else:
            return 0.50  # Conservative Half-Kelly for larger amounts


@dataclass
class StrategyConfig:
    """Strategy toggle configuration."""
    enable_binary_arb: bool = field(
        default_factory=lambda: os.getenv("ENABLE_BINARY_ARB", "true").lower() == "true"
    )
    enable_multi_outcome_arb: bool = field(
        default_factory=lambda: os.getenv("ENABLE_MULTI_OUTCOME_ARB", "true").lower() == "true"
    )
    enable_endgame_arb: bool = field(
        default_factory=lambda: os.getenv("ENABLE_ENDGAME_ARB", "true").lower() == "true"
    )
    enable_cross_platform_arb: bool = field(
        default_factory=lambda: os.getenv("ENABLE_CROSS_PLATFORM_ARB", "false").lower() == "true"
    )
    enable_whale_tracking: bool = field(
        default_factory=lambda: os.getenv("ENABLE_WHALE_TRACKING", "false").lower() == "true"
    )
    enable_market_making: bool = field(
        default_factory=lambda: os.getenv("ENABLE_MARKET_MAKING", "false").lower() == "true"
    )


@dataclass
class MonitoringConfig:
    """Monitoring configuration with WebSocket optimizations."""
    poll_interval: int = field(
        default_factory=lambda: int(os.getenv("POLL_INTERVAL", "5"))
    )
    use_websocket: bool = field(
        default_factory=lambda: os.getenv("USE_WEBSOCKET", "true").lower() == "true"
    )
    log_level: str = field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO")
    )

    # WebSocket heartbeat intervals (documented requirements)
    ws_clob_ping_interval: int = 10  # CLOB requires ping every 10 seconds
    ws_rtds_ping_interval: int = 5   # RTDS requires ping every 5 seconds
    ws_ping_timeout: int = 10

    # Reconnection with exponential backoff
    ws_reconnect_base_delay: float = 1.0
    ws_reconnect_max_delay: float = 60.0
    ws_reconnect_jitter: float = 0.1  # 10% random jitter

    # Message batching for high-volume periods
    message_batch_window_ms: int = 100


@dataclass
class RiskConfig:
    """Risk management configuration with adverse selection."""
    max_daily_loss: float = field(
        default_factory=lambda: float(os.getenv("MAX_DAILY_LOSS", "50"))
    )
    max_open_positions: int = field(
        default_factory=lambda: int(os.getenv("MAX_OPEN_POSITIONS", "10"))
    )
    trade_cooldown: int = field(
        default_factory=lambda: int(os.getenv("TRADE_COOLDOWN", "2"))
    )

    # Depth requirements
    min_depth_multiplier: float = 10.0  # Only trade where depth > 10x trade size

    # Spread-based position reduction
    wide_spread_threshold: float = 0.03  # Reduce 50% when spread > 3%
    wide_spread_reduction: float = 0.50

    # Adverse selection avoidance (minutes before/after known news)
    news_blackout_minutes: int = field(
        default_factory=lambda: int(os.getenv("NEWS_BLACKOUT_MINUTES", "30"))
    )

    # Large order detection (signals informed flow)
    whale_trade_threshold: float = field(
        default_factory=lambda: float(os.getenv("WHALE_TRADE_THRESHOLD", "10000"))
    )


@dataclass
class WhaleTrackingConfig:
    """Smart money tracking configuration."""
    enabled: bool = field(
        default_factory=lambda: os.getenv("ENABLE_WHALE_TRACKING", "false").lower() == "true"
    )
    # Poll interval for whale positions (seconds)
    poll_interval: int = field(
        default_factory=lambda: int(os.getenv("WHALE_POLL_INTERVAL", "4"))
    )
    # Target wallets to track (comma-separated addresses)
    target_wallets: str = field(
        default_factory=lambda: os.getenv("WHALE_TARGET_WALLETS", "")
    )
    # Max position per market when copy trading
    max_copy_position_percent: float = field(
        default_factory=lambda: float(os.getenv("WHALE_MAX_COPY_PERCENT", "0.20"))  # 20% max
    )
    # Minimum confidence (multiple whales moving same direction)
    min_whale_consensus: int = field(
        default_factory=lambda: int(os.getenv("WHALE_MIN_CONSENSUS", "2"))
    )

    def get_target_wallet_list(self) -> list:
        """Parse target wallets from comma-separated string."""
        if not self.target_wallets:
            return []
        return [w.strip() for w in self.target_wallets.split(",") if w.strip()]


@dataclass
class NotificationConfig:
    """Notification configuration."""
    telegram_bot_token: Optional[str] = field(
        default_factory=lambda: os.getenv("TELEGRAM_BOT_TOKEN") or None
    )
    telegram_chat_id: Optional[str] = field(
        default_factory=lambda: os.getenv("TELEGRAM_CHAT_ID") or None
    )

    @property
    def telegram_enabled(self) -> bool:
        return bool(self.telegram_bot_token and self.telegram_chat_id)


@dataclass
class LatencyConfig:
    """Expected latency benchmarks for monitoring."""
    # Documented benchmarks
    order_execution_mean_ms: float = 42.3
    order_execution_p95_ms: float = 67.8
    market_data_mean_ms: float = 15.8
    exchange_processing_ms: float = 7.0  # 4-10ms range

    # Alert thresholds
    latency_alert_threshold_ms: float = 100.0


@dataclass
class Config:
    """Main configuration container with all advanced options."""
    wallet: WalletConfig = field(default_factory=WalletConfig)
    api: APIConfig = field(default_factory=APIConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    kelly: KellyConfig = field(default_factory=KellyConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    whale_tracking: WhaleTrackingConfig = field(default_factory=WhaleTrackingConfig)
    notification: NotificationConfig = field(default_factory=NotificationConfig)
    latency: LatencyConfig = field(default_factory=LatencyConfig)

    def validate(self) -> bool:
        """Validate all configuration."""
        return self.wallet.validate()

    def get_tick_size(self, price: float) -> float:
        """Get appropriate tick size based on price level."""
        if price >= self.trading.high_price_threshold or price <= self.trading.low_price_threshold:
            return self.trading.extreme_tick_size
        return self.trading.normal_tick_size

    def log_config(self):
        """Log current configuration (hiding sensitive data)."""
        logger.info("=" * 60)
        logger.info("Polymarket Trading Bot Configuration (Advanced)")
        logger.info("=" * 60)

        if self.wallet.wallet_address:
            logger.info(f"Wallet: {self.wallet.wallet_address[:10]}...{self.wallet.wallet_address[-6:]}")
        else:
            logger.info("Wallet: Not configured")

        logger.info(f"Chain ID: {self.api.chain_id}")
        logger.info(f"Dry Run: {self.trading.dry_run}")

        logger.info("\n[Trading Parameters]")
        logger.info(f"  Min Profit Margin: {self.trading.min_profit_margin * 100:.2f}%")
        logger.info(f"  Max Position Size: ${self.trading.max_position_size}")
        logger.info(f"  Max Slippage: {self.trading.max_slippage * 100:.1f}%")

        logger.info("\n[Kelly Position Sizing]")
        logger.info(f"  Kelly Fraction: {self.kelly.default_kelly_fraction}x")
        logger.info(f"  Max Per Position: {self.kelly.max_position_percent * 100:.0f}%")
        logger.info(f"  Max Total Exposure: {self.kelly.max_total_exposure * 100:.0f}%")

        logger.info("\n[Strategies Enabled]")
        logger.info(f"  Binary Arb: {self.strategy.enable_binary_arb}")
        logger.info(f"  Multi-Outcome Arb: {self.strategy.enable_multi_outcome_arb}")
        logger.info(f"  Endgame Arb: {self.strategy.enable_endgame_arb}")
        logger.info(f"  Cross-Platform Arb: {self.strategy.enable_cross_platform_arb}")
        logger.info(f"  Whale Tracking: {self.strategy.enable_whale_tracking}")

        logger.info("\n[Risk Management]")
        logger.info(f"  Max Daily Loss: ${self.risk.max_daily_loss}")
        logger.info(f"  News Blackout: {self.risk.news_blackout_minutes} min")
        logger.info(f"  Whale Detection Threshold: ${self.risk.whale_trade_threshold}")

        logger.info("\n[Monitoring]")
        logger.info(f"  WebSocket Mode: {self.monitoring.use_websocket}")
        logger.info(f"  WS Ping Interval: {self.monitoring.ws_clob_ping_interval}s")

        logger.info("=" * 60)


# Global config instance
config = Config()
