#!/usr/bin/env python3
"""
Real-Time Arbitrage Scanner for Polymarket.

This is the key script for capturing millisecond-level arbitrage opportunities.
Uses WebSocket feeds instead of REST API for instant price detection.

Usage:
    python realtime.py

The scanner will:
1. Connect to Polymarket WebSocket feed
2. Subscribe to top 500 market tokens (250 markets)
3. Monitor price changes in real-time
4. Detect arbitrage the instant YES + NO < $1.00
5. Log opportunities (DRY_RUN=true) or execute trades

For production:
- Deploy on VPS close to Polymarket servers (Amsterdam/EU)
- Set DRY_RUN=false in .env when ready
- Run 24/7 with systemd or screen
"""

import asyncio
import sys
from pathlib import Path

# Ensure src is in path
sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
    level="DEBUG",
    colorize=True,
)
logger.add(
    "logs/realtime_{time:YYYY-MM-DD}.log",
    rotation="1 day",
    retention="7 days",
    level="INFO",
)


async def main():
    """Main entry point."""
    from src.config import config
    from src.monitoring.realtime_scanner import run_realtime_scanner

    # Log configuration
    logger.info("=" * 70)
    logger.info("POLYMARKET REAL-TIME ARBITRAGE SCANNER")
    logger.info("=" * 70)
    logger.info("")
    logger.info(f"Mode: {'DRY RUN (simulation)' if config.trading.dry_run else 'LIVE TRADING'}")
    logger.info(f"Min Profit Threshold: {config.trading.min_profit_margin * 100:.3f}%")
    logger.info(f"WebSocket: {config.api.ws_clob_url}")
    logger.info("")

    if not config.trading.dry_run:
        logger.warning("!!! LIVE TRADING MODE - Real money at risk !!!")
        logger.warning("Press Ctrl+C within 5 seconds to abort...")
        try:
            await asyncio.sleep(5)
        except KeyboardInterrupt:
            logger.info("Aborted.")
            return

    await run_realtime_scanner()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutdown complete.")
