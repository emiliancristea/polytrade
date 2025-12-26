"""
Polymarket CLOB Client wrapper for trading operations.
Provides authenticated access for order placement and management.
"""

from typing import Any, Dict, List, Optional, Tuple
from decimal import Decimal
from loguru import logger

try:
    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import (
        OrderArgs,
        MarketOrderArgs,
        OrderType,
        BookParams,
    )
    CLOB_AVAILABLE = True
except ImportError:
    CLOB_AVAILABLE = False
    logger.warning("py-clob-client not installed. Install with: pip install py-clob-client")

from src.config import config


class PolymarketClient:
    """
    Wrapper around py-clob-client for Polymarket trading.
    Handles authentication, order creation, and execution.
    """

    def __init__(self, read_only: bool = False):
        """
        Initialize the Polymarket client.

        Args:
            read_only: If True, skip authentication (for market data only)
        """
        if not CLOB_AVAILABLE:
            raise ImportError("py-clob-client is required. Install with: pip install py-clob-client")

        self.read_only = read_only
        self._client: Optional[ClobClient] = None
        self._initialized = False

    def _init_client(self):
        """Initialize the CLOB client with proper authentication."""
        if self._initialized:
            return

        if self.read_only:
            # Read-only client for market data
            self._client = ClobClient(config.api.clob_host)
            self._initialized = True
            logger.info("Initialized read-only CLOB client")
            return

        # Full client with trading capabilities
        if not config.wallet.validate():
            raise ValueError("Invalid wallet configuration")

        self._client = ClobClient(
            host=config.api.clob_host,
            key=config.wallet.private_key,
            chain_id=config.api.chain_id,
            signature_type=config.wallet.signature_type,
            funder=config.wallet.funder_address,
        )

        # Derive or create API credentials
        try:
            creds = self._client.create_or_derive_api_creds()
            self._client.set_api_creds(creds)
            logger.info("Initialized authenticated CLOB client")
        except Exception as e:
            logger.error(f"Failed to derive API credentials: {e}")
            raise

        self._initialized = True

    @property
    def client(self) -> ClobClient:
        """Get the initialized CLOB client."""
        self._init_client()
        return self._client

    # =========================================
    # Market Data Methods (No Auth Required)
    # =========================================

    def get_price(self, token_id: str, side: str = "BUY") -> Optional[float]:
        """
        Get current price for a token.

        Args:
            token_id: The token ID (from market's clobTokenIds)
            side: "BUY" or "SELL"

        Returns:
            Price as float, or None if unavailable
        """
        try:
            price = self.client.get_price(token_id, side=side)
            return float(price) if price else None
        except Exception as e:
            logger.debug(f"Failed to get price for {token_id}: {e}")
            return None

    def get_midpoint(self, token_id: str) -> Optional[float]:
        """Get midpoint price for a token."""
        try:
            mid = self.client.get_midpoint(token_id)
            return float(mid) if mid else None
        except Exception as e:
            logger.debug(f"Failed to get midpoint for {token_id}: {e}")
            return None

    def get_spread(self, token_id: str) -> Optional[Dict[str, float]]:
        """
        Get bid-ask spread for a token.

        Returns:
            Dictionary with 'bid', 'ask', 'spread' keys
        """
        try:
            spread = self.client.get_spread(token_id)
            if spread:
                return {
                    "bid": float(spread.get("bid", 0)),
                    "ask": float(spread.get("ask", 0)),
                    "spread": float(spread.get("spread", 0)),
                }
            return None
        except Exception as e:
            logger.debug(f"Failed to get spread for {token_id}: {e}")
            return None

    def get_order_book(self, token_id: str) -> Optional[Dict[str, Any]]:
        """Get full order book for a token."""
        try:
            return self.client.get_order_book(token_id)
        except Exception as e:
            logger.debug(f"Failed to get order book for {token_id}: {e}")
            return None

    def get_last_trade_price(self, token_id: str) -> Optional[float]:
        """Get the last traded price for a token."""
        try:
            price = self.client.get_last_trade_price(token_id)
            return float(price) if price else None
        except Exception as e:
            logger.debug(f"Failed to get last trade price for {token_id}: {e}")
            return None

    # =========================================
    # Trading Methods (Auth Required)
    # =========================================

    def create_limit_order(
        self,
        token_id: str,
        side: str,
        price: float,
        size: float,
    ) -> Optional[Dict[str, Any]]:
        """
        Create a limit order.

        Args:
            token_id: The token to trade
            side: "BUY" or "SELL"
            price: Limit price (0.00 to 1.00)
            size: Number of shares

        Returns:
            Signed order ready for submission
        """
        if self.read_only:
            raise ValueError("Cannot create orders in read-only mode")

        try:
            order_args = OrderArgs(
                token_id=token_id,
                side=side,
                price=price,
                size=size,
            )
            signed_order = self.client.create_order(order_args)
            return signed_order
        except Exception as e:
            logger.error(f"Failed to create limit order: {e}")
            return None

    def create_market_order(
        self,
        token_id: str,
        side: str,
        amount: float,
    ) -> Optional[Dict[str, Any]]:
        """
        Create a market order (dollar amount based).

        Args:
            token_id: The token to trade
            side: "BUY" or "SELL"
            amount: Dollar amount to spend/receive

        Returns:
            Signed order ready for submission
        """
        if self.read_only:
            raise ValueError("Cannot create orders in read-only mode")

        try:
            order_args = MarketOrderArgs(
                token_id=token_id,
                side=side,
                amount=amount,
            )
            signed_order = self.client.create_market_order(order_args)
            return signed_order
        except Exception as e:
            logger.error(f"Failed to create market order: {e}")
            return None

    def post_order(
        self,
        signed_order: Dict[str, Any],
        order_type: str = "GTC",
    ) -> Optional[Dict[str, Any]]:
        """
        Submit a signed order to the exchange.

        Args:
            signed_order: The signed order from create_*_order
            order_type: "GTC" (Good Till Cancelled) or "FOK" (Fill or Kill)

        Returns:
            Order response from exchange
        """
        if self.read_only:
            raise ValueError("Cannot post orders in read-only mode")

        try:
            ot = OrderType.FOK if order_type == "FOK" else OrderType.GTC
            response = self.client.post_order(signed_order, ot)
            logger.info(f"Order posted: {response}")
            return response
        except Exception as e:
            logger.error(f"Failed to post order: {e}")
            return None

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a specific order."""
        if self.read_only:
            raise ValueError("Cannot cancel orders in read-only mode")

        try:
            self.client.cancel(order_id)
            logger.info(f"Cancelled order: {order_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    def cancel_all_orders(self) -> bool:
        """Cancel all open orders."""
        if self.read_only:
            raise ValueError("Cannot cancel orders in read-only mode")

        try:
            self.client.cancel_all()
            logger.info("Cancelled all orders")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel all orders: {e}")
            return False

    # =========================================
    # Account Methods (Auth Required)
    # =========================================

    def get_open_orders(self) -> List[Dict[str, Any]]:
        """Get all open orders for the authenticated user."""
        if self.read_only:
            return []

        try:
            orders = self.client.get_orders()
            return orders if isinstance(orders, list) else []
        except Exception as e:
            logger.error(f"Failed to get open orders: {e}")
            return []

    def get_trades(self) -> List[Dict[str, Any]]:
        """Get trade history for the authenticated user."""
        if self.read_only:
            return []

        try:
            trades = self.client.get_trades()
            return trades if isinstance(trades, list) else []
        except Exception as e:
            logger.error(f"Failed to get trades: {e}")
            return []

    # =========================================
    # Utility Methods
    # =========================================

    def check_connection(self) -> bool:
        """Verify connection to Polymarket CLOB."""
        try:
            ok = self.client.get_ok()
            return ok == "OK" or ok is True
        except Exception as e:
            logger.error(f"Connection check failed: {e}")
            return False

    def get_server_time(self) -> Optional[int]:
        """Get server time for sync verification."""
        try:
            return self.client.get_server_time()
        except Exception as e:
            logger.error(f"Failed to get server time: {e}")
            return None


# Convenience function for read-only client
def get_readonly_client() -> PolymarketClient:
    """Get a read-only client for market data."""
    return PolymarketClient(read_only=True)
