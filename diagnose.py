#!/usr/bin/env python3
"""
Live Diagnostic - Shows exactly what the scanner sees in real-time.
"""

import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

from src.api.gamma_api import GammaAPI
from src.monitoring.websocket_monitor import WebSocketMonitor
from src.config import config


async def run_diagnostic():
    print("=" * 70)
    print("LIVE DIAGNOSTIC - What the scanner actually sees")
    print("=" * 70)
    print()
    print(f"Profit threshold: {config.trading.min_profit_margin * 100:.3f}%")
    print(f"Catching any gap where YES + NO < ${1 - config.trading.min_profit_margin:.4f}")
    print()

    # Get markets
    gamma = GammaAPI()
    markets = await gamma.get_markets(active=True, closed=False, limit=50)
    print(f"Loaded {len(markets)} markets")

    # Build token -> market mapping
    token_to_market = {}
    market_prices = {}  # condition_id -> {yes_price, no_price, question}

    for m in markets:
        cid = m.get("conditionId")
        clob_ids = m.get("clobTokenIds")

        if not cid or not clob_ids:
            continue

        if isinstance(clob_ids, str):
            try:
                clob_ids = json.loads(clob_ids)
            except:
                continue

        if len(clob_ids) >= 2:
            yes_token = str(clob_ids[0])
            no_token = str(clob_ids[1])

            token_to_market[yes_token] = (cid, "YES")
            token_to_market[no_token] = (cid, "NO")

            # Parse initial prices
            prices_raw = m.get("outcomePrices", "[]")
            if isinstance(prices_raw, str):
                try:
                    prices = json.loads(prices_raw)
                except:
                    prices = [0.5, 0.5]
            else:
                prices = prices_raw

            market_prices[cid] = {
                "yes_price": float(prices[0]) if len(prices) > 0 else 0.5,
                "no_price": float(prices[1]) if len(prices) > 1 else 0.5,
                "question": m.get("question", "")[:50],
                "updates": 0,
            }

    print(f"Tracking {len(market_prices)} markets, {len(token_to_market)} tokens")
    print()
    print("-" * 70)
    print("LIVE PRICE UPDATES (showing gaps):")
    print("-" * 70)

    # Stats
    total_updates = 0
    opportunities_found = 0
    smallest_total = 2.0  # Track smallest YES+NO we've seen

    # Connect to WebSocket
    ws = WebSocketMonitor(ws_type="clob")

    async def on_message(data):
        nonlocal total_updates, opportunities_found, smallest_total

        token_id = str(data.get("token_id", ""))
        if token_id not in token_to_market:
            return

        cid, side = token_to_market[token_id]
        if cid not in market_prices:
            return

        # Get ask price (what we'd pay to buy)
        ask_price = data.get("best_ask")
        if ask_price is None:
            ask_price = data.get("price")
        if ask_price is None:
            return

        ask_price = float(ask_price)
        total_updates += 1

        # Update price
        mp = market_prices[cid]
        mp["updates"] += 1

        if side == "YES":
            mp["yes_price"] = ask_price
        else:
            mp["no_price"] = ask_price

        # Calculate total
        total = mp["yes_price"] + mp["no_price"]
        gap = 1.0 - total

        # Track smallest total seen
        if total < smallest_total:
            smallest_total = total

        # Check for opportunity
        if gap > config.trading.min_profit_margin:
            opportunities_found += 1
            print(
                f"\n>>> OPPORTUNITY #{opportunities_found} | "
                f"YES={mp['yes_price']:.4f} + NO={mp['no_price']:.4f} = {total:.4f} | "
                f"GAP: {gap*100:.3f}% | {mp['question']}"
            )

        # Print status every 50 updates
        if total_updates % 50 == 0:
            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] "
                f"Updates: {total_updates} | "
                f"Opportunities: {opportunities_found} | "
                f"Smallest total seen: ${smallest_total:.4f}"
            )

    ws.add_callback(on_message)

    # Subscribe to tokens
    await ws.connect()
    token_list = list(token_to_market.keys())[:500]
    await ws.subscribe(token_list)

    print(f"Subscribed to {len(token_list)} tokens. Monitoring for 60 seconds...")
    print()
    print("NOTE: Updates only happen when prices CHANGE on Polymarket.")
    print("If markets are quiet, you'll see few updates. This is normal.")
    print()

    # Run for 60 seconds
    try:
        start = asyncio.get_event_loop().time()
        while asyncio.get_event_loop().time() - start < 60:
            await asyncio.sleep(0.1)
            # Keep WS alive
            if not ws.is_connected:
                break
    except KeyboardInterrupt:
        pass

    await ws.disconnect()

    # Final summary
    print()
    print("=" * 70)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 70)
    print(f"Total price updates received: {total_updates}")
    print(f"Opportunities found: {opportunities_found}")
    print(f"Smallest YES+NO total seen: ${smallest_total:.4f}")
    print()

    if smallest_total < 1.0:
        print(f"Best gap seen: {(1-smallest_total)*100:.3f}%")
    else:
        print("No gaps observed - all prices summed to >= $1.00")

    print()
    print("If updates are flowing but no opportunities, the market is efficient.")
    print("Run longer (hours) to catch rare opportunities.")


if __name__ == "__main__":
    asyncio.run(run_diagnostic())
