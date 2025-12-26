"""
WebSocket Monitor for real-time Polymarket price updates.

Implements documented requirements:
- CLOB WebSocket: PING every 10 seconds
- RTDS WebSocket: PING every 5 seconds
- Exponential backoff with 10% jitter for reconnection
- Re-subscribe to all channels after reconnection
- Maximum 500 instruments per connection
"""

import asyncio
import json
import random
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set
import websockets
from websockets.exceptions import ConnectionClosed
from websockets.protocol import State
from loguru import logger

from src.config import config
from src.api.rate_limiter import rate_limiter


def is_ws_open(ws) -> bool:
    """Check if websocket connection is open (compatible with websockets 15.x)."""
    if ws is None:
        return False
    try:
        # websockets 15.x uses state property
        return ws.state == State.OPEN
    except AttributeError:
        # Fallback for older versions
        try:
            return not ws.closed
        except AttributeError:
            return False


class WebSocketMonitor:
    """
    Real-time WebSocket monitor for Polymarket price updates.
    Implements proper heartbeat and reconnection logic.
    """

    def __init__(self, ws_type: str = "clob"):
        """
        Initialize WebSocket monitor.

        Args:
            ws_type: "clob" or "rtds" - determines ping interval
        """
        self.ws_type = ws_type
        self.ws_url = (
            config.api.ws_clob_url if ws_type == "clob"
            else config.api.ws_rtds_url
        )
        self.ping_interval = (
            config.monitoring.ws_clob_ping_interval if ws_type == "clob"
            else config.monitoring.ws_rtds_ping_interval
        )

        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._running = False
        self._subscribed_tokens: Set[str] = set()
        self._price_cache: Dict[str, Dict[str, Any]] = {}
        self._callbacks: List[Callable] = []

        # Reconnection state
        self._reconnect_delay = config.monitoring.ws_reconnect_base_delay
        self._connection_id = f"ws-{ws_type}-{int(time.time())}"

        # Heartbeat task
        self._heartbeat_task: Optional[asyncio.Task] = None

        # Stats
        self._messages_received = 0
        self._reconnections = 0
        self._last_message_time: Optional[datetime] = None

    async def connect(self) -> bool:
        """Establish WebSocket connection with timeout."""
        try:
            self._ws = await asyncio.wait_for(
                websockets.connect(
                    self.ws_url,
                    ping_interval=None,  # We handle pings manually
                    ping_timeout=config.monitoring.ws_ping_timeout,
                    close_timeout=10,
                ),
                timeout=30,
            )
            self._running = True
            self._reconnect_delay = config.monitoring.ws_reconnect_base_delay
            self._connection_id = f"ws-{self.ws_type}-{int(time.time())}"

            # Start heartbeat task
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

            logger.info(f"WebSocket connected: {self.ws_type} ({self._connection_id})")
            return True

        except asyncio.TimeoutError:
            logger.error("WebSocket connection timeout")
            return False
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            return False

    async def disconnect(self):
        """Close WebSocket connection gracefully."""
        self._running = False

        # Cancel heartbeat
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        # Clear subscription tracking
        rate_limiter.clear_ws_connection(self._connection_id)

        # Close WebSocket
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

        logger.info(f"WebSocket disconnected: {self.ws_type}")

    async def _heartbeat_loop(self):
        """Send periodic pings to keep connection alive."""
        consecutive_failures = 0
        max_failures = 3

        while self._running and self._ws:
            try:
                await asyncio.sleep(self.ping_interval)

                if is_ws_open(self._ws):
                    try:
                        # Try ping frame
                        pong_waiter = await self._ws.ping()
                        await asyncio.wait_for(pong_waiter, timeout=config.monitoring.ws_ping_timeout)
                        logger.debug(f"[{self.ws_type}] Heartbeat OK")
                        consecutive_failures = 0
                    except asyncio.TimeoutError:
                        consecutive_failures += 1
                        logger.debug(f"[{self.ws_type}] Ping timeout ({consecutive_failures}/{max_failures})")
                        if consecutive_failures >= max_failures:
                            logger.warning(f"[{self.ws_type}] Too many heartbeat failures")
                            break
                    except Exception as e:
                        # Some servers don't support ping - that's OK
                        logger.debug(f"[{self.ws_type}] Ping not supported: {e}")
                        consecutive_failures = 0  # Reset since connection might still work

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[{self.ws_type}] Heartbeat error: {e}")
                consecutive_failures += 1
                if consecutive_failures >= max_failures:
                    break

    async def subscribe(self, token_ids: List[str]) -> bool:
        """
        Subscribe to price updates for specific tokens.
        Respects the 500 instrument limit per connection.

        Args:
            token_ids: List of token IDs to monitor

        Returns:
            True if subscription successful
        """
        if not is_ws_open(self._ws):
            logger.error("WebSocket not connected")
            return False

        # Filter already subscribed
        new_tokens = [t for t in token_ids if t not in self._subscribed_tokens]
        if not new_tokens:
            return True

        # Check rate limit
        if not rate_limiter.check_ws_subscription_limit(self._connection_id, len(new_tokens)):
            logger.warning(f"Cannot subscribe to {len(new_tokens)} tokens - would exceed limit")
            # Take as many as we can
            remaining = config.api.max_ws_instruments - len(self._subscribed_tokens)
            new_tokens = new_tokens[:remaining]
            if not new_tokens:
                return False

        try:
            # Subscribe in batches of 50 to avoid message size issues
            batch_size = 50
            for i in range(0, len(new_tokens), batch_size):
                batch = new_tokens[i:i + batch_size]

                subscribe_msg = {
                    "type": "subscribe",
                    "channel": "market",
                    "assets_ids": batch,
                }
                await self._ws.send(json.dumps(subscribe_msg))

                for token_id in batch:
                    self._subscribed_tokens.add(token_id)

                rate_limiter.add_ws_subscriptions(self._connection_id, len(batch))

                # Small delay between batches
                if i + batch_size < len(new_tokens):
                    await asyncio.sleep(0.1)

            logger.info(f"Subscribed to {len(new_tokens)} tokens (total: {len(self._subscribed_tokens)})")
            return True

        except Exception as e:
            logger.error(f"Subscribe failed: {e}")
            return False

    async def unsubscribe(self, token_ids: List[str]):
        """Unsubscribe from token updates."""
        if not is_ws_open(self._ws):
            return

        tokens_to_unsub = [t for t in token_ids if t in self._subscribed_tokens]
        if not tokens_to_unsub:
            return

        try:
            unsubscribe_msg = {
                "type": "unsubscribe",
                "channel": "market",
                "assets_ids": tokens_to_unsub,
            }
            await self._ws.send(json.dumps(unsubscribe_msg))

            for token_id in tokens_to_unsub:
                self._subscribed_tokens.discard(token_id)

            rate_limiter.remove_ws_subscriptions(self._connection_id, len(tokens_to_unsub))
            logger.debug(f"Unsubscribed from {len(tokens_to_unsub)} tokens")

        except Exception as e:
            logger.error(f"Unsubscribe failed: {e}")

    def add_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Add a callback for price updates."""
        self._callbacks.append(callback)

    def remove_callback(self, callback: Callable):
        """Remove a callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    async def run(self):
        """
        Main WebSocket event loop with reconnection logic.
        Uses exponential backoff with jitter.
        """
        self._running = True  # Ensure running is set
        logger.debug(f"WebSocket run() started, _running={self._running}")

        while self._running:
            try:
                # Connect if needed
                if not is_ws_open(self._ws):
                    if not await self.connect():
                        await self._wait_with_backoff()
                        continue

                    # Re-subscribe after reconnect
                    if self._subscribed_tokens:
                        tokens = list(self._subscribed_tokens)
                        self._subscribed_tokens.clear()
                        rate_limiter.clear_ws_connection(self._connection_id)
                        await self.subscribe(tokens)

                # Receive message with timeout
                try:
                    message = await asyncio.wait_for(
                        self._ws.recv(),
                        timeout=self.ping_interval * 2,  # 2x ping interval as timeout
                    )
                    await self._process_message(message)
                    self._last_message_time = datetime.now()

                except asyncio.TimeoutError:
                    # No message received - check connection with ping
                    continue

            except ConnectionClosed as e:
                logger.warning(f"WebSocket closed: {e.code} - {e.reason}")
                self._reconnections += 1
                self._ws = None
                await self._wait_with_backoff()

            except asyncio.CancelledError:
                break

            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                self._ws = None
                await self._wait_with_backoff()

        logger.warning(f"WebSocket run loop exited. _running={self._running}")

    async def _wait_with_backoff(self):
        """Wait with exponential backoff and jitter before reconnecting."""
        # Add jitter (10% randomness)
        jitter = self._reconnect_delay * config.monitoring.ws_reconnect_jitter
        wait_time = self._reconnect_delay + random.uniform(-jitter, jitter)

        logger.info(f"Reconnecting in {wait_time:.2f}s...")
        await asyncio.sleep(wait_time)

        # Increase delay for next time (exponential backoff)
        self._reconnect_delay = min(
            self._reconnect_delay * 2,
            config.monitoring.ws_reconnect_max_delay,
        )

    async def _process_message(self, message: str):
        """Process incoming WebSocket message."""
        self._messages_received += 1

        try:
            data = json.loads(message)

            # Polymarket sends messages as a list of updates
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        await self._handle_book_update(item)
                return

            # Handle single dict messages
            if isinstance(data, dict):
                msg_type = data.get("type", "")

                if msg_type == "price_change":
                    await self._handle_price_update(data)
                elif msg_type == "book":
                    await self._handle_book_update(data)
                elif msg_type == "trade":
                    await self._handle_trade(data)
                elif msg_type == "subscribed":
                    logger.debug(f"Subscription confirmed: {len(data.get('assets_ids', []))} tokens")
                elif msg_type == "error":
                    logger.error(f"WebSocket error: {data}")
                elif msg_type == "pong":
                    pass  # Heartbeat response
                elif "asset_id" in data:
                    # Order book update without explicit type
                    await self._handle_book_update(data)
                elif msg_type:
                    # Only log if there's actually a type we don't recognize
                    logger.debug(f"Unknown message type: {msg_type}")

        except json.JSONDecodeError:
            # "INVALID OPERATION" is a known response for invalid subscriptions
            if message != "INVALID OPERATION":
                logger.warning(f"Invalid JSON: {message[:100]}")
        except Exception as e:
            logger.error(f"Message processing error: {e}")

    async def _handle_price_update(self, data: Dict[str, Any]):
        """Handle price update message."""
        asset_id = data.get("asset_id")
        if not asset_id:
            return

        price_data = {
            "token_id": asset_id,
            "price": float(data.get("price", 0)),
            "timestamp": datetime.now(),
            "type": "price_change",
        }

        self._price_cache[asset_id] = price_data
        await self._trigger_callbacks(price_data)

    async def _handle_book_update(self, data: Dict[str, Any]):
        """Handle order book update."""
        asset_id = data.get("asset_id")
        if not asset_id:
            return

        bids = data.get("bids", [])
        asks = data.get("asks", [])

        # Calculate best bid/ask (these are the prices we care about for arbitrage)
        best_bid = 0.0
        best_ask = 1.0

        if bids:
            try:
                best_bid = max(float(b.get("price", 0)) for b in bids)
            except (ValueError, TypeError):
                pass

        if asks:
            try:
                best_ask = min(float(a.get("price", 1)) for a in asks)
            except (ValueError, TypeError):
                pass

        # Calculate mid price for the callback
        mid_price = (best_bid + best_ask) / 2 if best_bid > 0 and best_ask < 1 else None

        book_data = {
            "token_id": str(asset_id),
            "bids": bids,
            "asks": asks,
            "best_bid": best_bid,
            "best_ask": best_ask,
            "price": mid_price,  # Add mid price for easier processing
            "timestamp": datetime.now(),
            "type": "book",
        }

        self._price_cache[str(asset_id)] = book_data
        await self._trigger_callbacks(book_data)

    async def _handle_trade(self, data: Dict[str, Any]):
        """Handle trade message."""
        trade_data = {
            "token_id": data.get("asset_id"),
            "price": float(data.get("price", 0)),
            "size": float(data.get("size", 0)),
            "side": data.get("side"),
            "timestamp": datetime.now(),
            "type": "trade",
        }

        await self._trigger_callbacks(trade_data)

    async def _trigger_callbacks(self, data: Dict[str, Any]):
        """Trigger all registered callbacks."""
        for callback in self._callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
            except Exception as e:
                logger.error(f"Callback error: {e}")

    def get_cached_price(self, token_id: str) -> Optional[Dict[str, Any]]:
        """Get cached price data for a token."""
        return self._price_cache.get(token_id)

    def get_all_cached_prices(self) -> Dict[str, Dict[str, Any]]:
        """Get all cached price data."""
        return self._price_cache.copy()

    @property
    def is_connected(self) -> bool:
        """Check if WebSocket is connected."""
        return is_ws_open(self._ws)

    @property
    def subscribed_count(self) -> int:
        """Number of subscribed tokens."""
        return len(self._subscribed_tokens)

    def get_stats(self) -> Dict[str, Any]:
        """Get WebSocket statistics."""
        return {
            "connected": self.is_connected,
            "ws_type": self.ws_type,
            "subscribed_tokens": len(self._subscribed_tokens),
            "messages_received": self._messages_received,
            "reconnections": self._reconnections,
            "last_message": self._last_message_time.isoformat() if self._last_message_time else None,
            "cached_prices": len(self._price_cache),
        }
