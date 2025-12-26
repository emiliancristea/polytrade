#!/usr/bin/env python3
"""
Latency Test for Polymarket Trading

Measures your actual connection speed to Polymarket servers.
Run this to see if your PC is fast enough for arbitrage.
"""

import asyncio
import time
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import aiohttp
import websockets


async def test_api_latency(url: str, num_tests: int = 10) -> dict:
    """Test REST API latency."""
    latencies = []

    async with aiohttp.ClientSession() as session:
        for i in range(num_tests):
            start = time.perf_counter()
            try:
                async with session.get(url) as response:
                    await response.read()
                    latency = (time.perf_counter() - start) * 1000
                    latencies.append(latency)
                    print(f"  API test {i+1}: {latency:.1f}ms")
            except Exception as e:
                print(f"  API test {i+1}: ERROR - {e}")

            await asyncio.sleep(0.1)  # Small delay between tests

    if latencies:
        return {
            "min": min(latencies),
            "max": max(latencies),
            "avg": statistics.mean(latencies),
            "median": statistics.median(latencies),
            "samples": len(latencies),
        }
    return None


async def test_websocket_latency(url: str, num_tests: int = 10) -> dict:
    """Test WebSocket round-trip latency using ping/pong."""
    latencies = []

    try:
        async with websockets.connect(url) as ws:
            print(f"  WebSocket connected to {url}")

            for i in range(num_tests):
                start = time.perf_counter()
                pong = await ws.ping()
                await pong  # Wait for pong response
                latency = (time.perf_counter() - start) * 1000
                latencies.append(latency)
                print(f"  WS ping {i+1}: {latency:.1f}ms")

                await asyncio.sleep(0.5)  # Wait between pings

    except Exception as e:
        print(f"  WebSocket error: {e}")

    if latencies:
        return {
            "min": min(latencies),
            "max": max(latencies),
            "avg": statistics.mean(latencies),
            "median": statistics.median(latencies),
            "samples": len(latencies),
        }
    return None


async def test_order_simulation() -> dict:
    """
    Simulate order placement timing (without actually placing order).
    Measures: API call to get market data + theoretical order time.
    """
    latencies = []
    url = "https://clob.polymarket.com/markets"

    async with aiohttp.ClientSession() as session:
        for i in range(5):
            start = time.perf_counter()
            try:
                # Simulate: Get orderbook + place order (2 API calls)
                async with session.get(url) as response:
                    await response.read()
                async with session.get(url) as response:
                    await response.read()
                latency = (time.perf_counter() - start) * 1000
                latencies.append(latency)
                print(f"  Order simulation {i+1}: {latency:.1f}ms (2 API calls)")
            except Exception as e:
                print(f"  Order simulation {i+1}: ERROR - {e}")

            await asyncio.sleep(0.2)

    if latencies:
        return {
            "min": min(latencies),
            "max": max(latencies),
            "avg": statistics.mean(latencies),
            "median": statistics.median(latencies),
        }
    return None


async def main():
    print("=" * 60)
    print("POLYMARKET LATENCY TEST")
    print("=" * 60)
    print("")
    print("Testing your connection speed to Polymarket servers...")
    print("This will help determine if your PC is fast enough for arbitrage.")
    print("")

    # Test 1: REST API latency
    print("[1/3] Testing REST API latency (clob.polymarket.com)...")
    api_result = await test_api_latency(
        "https://clob.polymarket.com/markets?limit=1",
        num_tests=10
    )
    print("")

    # Test 2: WebSocket latency
    print("[2/3] Testing WebSocket latency...")
    ws_result = await test_websocket_latency(
        "wss://ws-subscriptions-clob.polymarket.com/ws/market",
        num_tests=10
    )
    print("")

    # Test 3: Order simulation
    print("[3/3] Simulating order round-trip...")
    order_result = await test_order_simulation()
    print("")

    # Results
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print("")

    if api_result:
        print(f"REST API Latency:")
        print(f"  Average: {api_result['avg']:.1f}ms")
        print(f"  Median:  {api_result['median']:.1f}ms")
        print(f"  Min:     {api_result['min']:.1f}ms")
        print(f"  Max:     {api_result['max']:.1f}ms")
        print("")

    if ws_result:
        print(f"WebSocket Latency (ping/pong):")
        print(f"  Average: {ws_result['avg']:.1f}ms")
        print(f"  Median:  {ws_result['median']:.1f}ms")
        print(f"  Min:     {ws_result['min']:.1f}ms")
        print(f"  Max:     {ws_result['max']:.1f}ms")
        print("")

    if order_result:
        print(f"Order Round-Trip (simulated):")
        print(f"  Average: {order_result['avg']:.1f}ms")
        print(f"  Median:  {order_result['median']:.1f}ms")
        print("")

    # Assessment
    print("=" * 60)
    print("ASSESSMENT")
    print("=" * 60)
    print("")

    if ws_result:
        avg_latency = ws_result['avg']

        if avg_latency < 50:
            print(f"[OK] EXCELLENT ({avg_latency:.0f}ms)")
            print("   Your connection is very fast!")
            print("   You can compete with most bots.")
        elif avg_latency < 100:
            print(f"[OK] GOOD ({avg_latency:.0f}ms)")
            print("   Your connection is decent.")
            print("   You'll catch some opportunities.")
        elif avg_latency < 200:
            print(f"[!!] MODERATE ({avg_latency:.0f}ms)")
            print("   You might miss fast opportunities.")
            print("   VPS would help but not required.")
        else:
            print(f"[XX] SLOW ({avg_latency:.0f}ms)")
            print("   You'll likely miss most opportunities.")
            print("   Amsterdam VPS recommended.")

        print("")
        print("Comparison:")
        print(f"  Your PC:        ~{avg_latency:.0f}ms")
        print(f"  Amsterdam VPS:  ~30-50ms (estimated)")
        print(f"  Co-located:     ~5-10ms (pro traders)")

    print("")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
