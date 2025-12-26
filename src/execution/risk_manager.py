"""
Risk Management Module.
Enforces trading limits, position sizes, and daily loss limits.
"""

from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Dict, List, Optional
from loguru import logger

from src.config import config
from src.strategies.base import ArbitrageOpportunity, RiskLevel


@dataclass
class DailyStats:
    """Daily trading statistics."""
    date: date
    total_profit: float = 0.0
    total_loss: float = 0.0
    trades_executed: int = 0
    opportunities_skipped: int = 0

    @property
    def net_pnl(self) -> float:
        return self.total_profit - self.total_loss


@dataclass
class Position:
    """Represents an open position."""
    market_id: str
    token_id: str
    outcome: str
    shares: float
    entry_price: float
    entry_time: datetime
    investment: float

    @property
    def current_value(self) -> float:
        """Would need live price to calculate accurately."""
        return self.shares * self.entry_price


class RiskManager:
    """
    Manages trading risk and enforces limits.
    """

    def __init__(self):
        """Initialize risk manager with config settings."""
        self.max_daily_loss = config.risk.max_daily_loss
        self.max_open_positions = config.risk.max_open_positions
        self.max_position_size = config.trading.max_position_size
        self.trade_cooldown = config.risk.trade_cooldown

        # State
        self.daily_stats: Dict[date, DailyStats] = {}
        self.open_positions: Dict[str, Position] = {}
        self.last_trade_time: Optional[datetime] = None

        logger.info(
            f"Risk manager initialized: max_daily_loss=${self.max_daily_loss}, "
            f"max_positions={self.max_open_positions}, "
            f"max_position_size=${self.max_position_size}"
        )

    def _get_today_stats(self) -> DailyStats:
        """Get or create today's statistics."""
        today = date.today()
        if today not in self.daily_stats:
            self.daily_stats[today] = DailyStats(date=today)
        return self.daily_stats[today]

    def can_trade(self, opportunity: ArbitrageOpportunity) -> tuple[bool, str]:
        """
        Check if we can execute this trade.

        Args:
            opportunity: The opportunity to evaluate

        Returns:
            Tuple of (can_trade, reason)
        """
        # Check dry run mode
        if config.trading.dry_run:
            logger.debug("Dry run mode - would check trade feasibility")
            # In dry run, we still validate but don't actually block

        # Check daily loss limit
        stats = self._get_today_stats()
        if stats.total_loss >= self.max_daily_loss:
            return False, f"Daily loss limit reached (${stats.total_loss:.2f})"

        # Check max open positions
        if len(self.open_positions) >= self.max_open_positions:
            return False, f"Max open positions reached ({self.max_open_positions})"

        # Check position size
        if opportunity.investment_required > self.max_position_size:
            return False, f"Position size ${opportunity.investment_required:.2f} exceeds max ${self.max_position_size:.2f}"

        # Check cooldown
        if self.last_trade_time:
            elapsed = (datetime.now() - self.last_trade_time).total_seconds()
            if elapsed < self.trade_cooldown:
                return False, f"Trade cooldown ({self.trade_cooldown - elapsed:.1f}s remaining)"

        # Check if already have position in this market
        if opportunity.market_id in self.open_positions:
            return False, f"Already have position in market {opportunity.market_id}"

        return True, "Trade allowed"

    def validate_opportunity(self, opportunity: ArbitrageOpportunity) -> tuple[bool, str]:
        """
        Validate an opportunity meets our criteria.

        Args:
            opportunity: The opportunity to validate

        Returns:
            Tuple of (is_valid, reason)
        """
        # Check profit margin
        if opportunity.profit_margin < config.trading.min_profit_margin:
            return False, f"Profit margin {opportunity.profit_margin:.4f} below minimum {config.trading.min_profit_margin}"

        # Check liquidity
        if opportunity.liquidity < config.trading.min_liquidity:
            return False, f"Liquidity ${opportunity.liquidity:.2f} below minimum ${config.trading.min_liquidity:.2f}"

        # Check confidence
        if opportunity.confidence < 0.5:
            return False, f"Confidence {opportunity.confidence:.2f} too low"

        # Check for expired opportunities
        if opportunity.expires_at and opportunity.expires_at < datetime.now():
            return False, "Opportunity has expired"

        return True, "Opportunity valid"

    def calculate_position_size(
        self,
        opportunity: ArbitrageOpportunity,
        available_capital: float,
    ) -> float:
        """
        Calculate optimal position size for an opportunity.

        Args:
            opportunity: The opportunity
            available_capital: Available capital to use

        Returns:
            Recommended position size in USDC
        """
        # Start with max allowed
        size = min(self.max_position_size, available_capital)

        # Scale by risk level
        risk_multipliers = {
            RiskLevel.LOW: 1.0,
            RiskLevel.MEDIUM: 0.5,
            RiskLevel.HIGH: 0.25,
        }
        size *= risk_multipliers.get(opportunity.risk_level, 0.25)

        # Scale by confidence
        size *= opportunity.confidence

        # Ensure minimum viable size
        min_size = 1.0  # $1 minimum
        if size < min_size:
            return 0.0  # Skip if too small

        return round(size, 2)

    def record_trade(
        self,
        opportunity: ArbitrageOpportunity,
        executed_amount: float,
        shares_received: float,
    ):
        """
        Record an executed trade.

        Args:
            opportunity: The executed opportunity
            executed_amount: Amount spent
            shares_received: Shares received
        """
        stats = self._get_today_stats()
        stats.trades_executed += 1
        self.last_trade_time = datetime.now()

        # Create position record
        for trade in opportunity.trades:
            position = Position(
                market_id=opportunity.market_id,
                token_id=trade.get("token_id", ""),
                outcome=trade.get("outcome", ""),
                shares=shares_received / len(opportunity.trades),  # Split evenly
                entry_price=trade.get("price", 0),
                entry_time=datetime.now(),
                investment=executed_amount / len(opportunity.trades),
            )
            key = f"{opportunity.market_id}-{trade.get('outcome', '')}"
            self.open_positions[key] = position

        logger.info(
            f"Recorded trade: ${executed_amount:.2f} -> {shares_received:.4f} shares "
            f"in {opportunity.market_question[:40]}..."
        )

    def record_profit(self, amount: float):
        """Record realized profit."""
        stats = self._get_today_stats()
        if amount >= 0:
            stats.total_profit += amount
        else:
            stats.total_loss += abs(amount)

    def close_position(self, market_id: str, outcome: str, realized_pnl: float):
        """Close a position and record P&L."""
        key = f"{market_id}-{outcome}"
        if key in self.open_positions:
            del self.open_positions[key]
            self.record_profit(realized_pnl)
            logger.info(f"Closed position: {key}, P&L: ${realized_pnl:.2f}")

    def get_stats(self) -> Dict:
        """Get current risk stats."""
        stats = self._get_today_stats()
        return {
            "date": stats.date.isoformat(),
            "net_pnl": stats.net_pnl,
            "total_profit": stats.total_profit,
            "total_loss": stats.total_loss,
            "trades_executed": stats.trades_executed,
            "open_positions": len(self.open_positions),
            "max_positions": self.max_open_positions,
            "daily_loss_remaining": self.max_daily_loss - stats.total_loss,
        }

    def should_stop_trading(self) -> tuple[bool, str]:
        """Check if we should stop trading for the day."""
        stats = self._get_today_stats()

        if stats.total_loss >= self.max_daily_loss:
            return True, f"Daily loss limit reached: ${stats.total_loss:.2f}"

        return False, "Trading allowed"
