"""
Notification Service for alerts.
Supports Telegram notifications for opportunities and trades.
"""

from datetime import datetime
from typing import Any, Dict, Optional
from loguru import logger

from src.config import config
from src.strategies.base import ArbitrageOpportunity


class NotificationService:
    """
    Handles notifications for the trading bot.
    Currently supports Telegram.
    """

    def __init__(self):
        self.telegram_enabled = config.notification.telegram_enabled
        self._bot = None

        if self.telegram_enabled:
            try:
                from telegram import Bot
                self._bot = Bot(token=config.notification.telegram_bot_token)
                logger.info("Telegram notifications enabled")
            except ImportError:
                logger.warning("python-telegram-bot not installed, Telegram disabled")
                self.telegram_enabled = False
            except Exception as e:
                logger.error(f"Failed to initialize Telegram: {e}")
                self.telegram_enabled = False

    async def send_opportunity_alert(self, opportunity: ArbitrageOpportunity):
        """Send alert for detected opportunity."""
        message = self._format_opportunity(opportunity)
        await self._send(message)

    async def send_trade_alert(
        self,
        opportunity: ArbitrageOpportunity,
        success: bool,
        details: Optional[str] = None,
    ):
        """Send alert for executed trade."""
        status = "SUCCESS" if success else "FAILED"
        message = (
            f"TRADE {status}\n"
            f"Strategy: {opportunity.strategy_type.value}\n"
            f"Market: {opportunity.market_question[:50]}...\n"
            f"Investment: ${opportunity.investment_required:.2f}\n"
            f"Expected Profit: ${opportunity.profit_amount:.4f}\n"
        )
        if details:
            message += f"Details: {details}\n"
        message += f"Time: {datetime.now().strftime('%H:%M:%S')}"

        await self._send(message)

    async def send_daily_summary(self, stats: Dict[str, Any]):
        """Send daily trading summary."""
        message = (
            f"DAILY SUMMARY\n"
            f"Date: {stats.get('date', 'N/A')}\n"
            f"Net P&L: ${stats.get('net_pnl', 0):.2f}\n"
            f"Trades: {stats.get('trades_executed', 0)}\n"
            f"Open Positions: {stats.get('open_positions', 0)}\n"
            f"Opportunities Found: {stats.get('opportunities_detected', 0)}\n"
        )
        await self._send(message)

    async def send_error_alert(self, error: str):
        """Send error notification."""
        message = f"ERROR ALERT\n{error}\nTime: {datetime.now().strftime('%H:%M:%S')}"
        await self._send(message)

    def _format_opportunity(self, opp: ArbitrageOpportunity) -> str:
        """Format opportunity for notification."""
        return (
            f"OPPORTUNITY DETECTED\n"
            f"Type: {opp.strategy_type.value.upper()}\n"
            f"Market: {opp.market_question[:60]}...\n"
            f"Profit Margin: {opp.profit_margin * 100:.3f}%\n"
            f"Expected Profit: ${opp.profit_amount:.4f}\n"
            f"ROI: {opp.roi_percent:.2f}%\n"
            f"Risk: {opp.risk_level.value}\n"
            f"Confidence: {opp.confidence * 100:.1f}%\n"
            f"Time: {opp.detected_at.strftime('%H:%M:%S')}"
        )

    async def _send(self, message: str):
        """Send message via configured channels."""
        if self.telegram_enabled and self._bot:
            try:
                await self._bot.send_message(
                    chat_id=config.notification.telegram_chat_id,
                    text=message,
                )
            except Exception as e:
                logger.error(f"Telegram send failed: {e}")

        # Always log the message
        logger.info(f"[NOTIFICATION] {message[:100]}...")
