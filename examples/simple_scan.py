#!/usr/bin/env python3
"""
Simple example: Scan for arbitrage opportunities.
No wallet needed - just fetches market data and analyzes for opportunities.
"""

import asyncio
import sys
sys.path.insert(0, "..")

from src.api.gamma_api import GammaAPI
from src.strategies.binary_arb import BinaryArbitrageStrategy, check_binary_arb


async def main():
    print("=" * 60)
    print("Polymarket Arbitrage Scanner - Simple Example")
    print("=" * 60)

    # Initialize API client
    gamma = GammaAPI()

    # Initialize strategy with 0.1% minimum profit threshold
    strategy = BinaryArbitrageStrategy(
        min_profit_margin=0.001,  # 0.1%
        use_live_prices=False,    # Use API prices (faster, less accurate)
    )

    try:
        # Fetch all active markets
        print("\nFetching markets...")
        markets = await gamma.get_all_active_markets()
        print(f"Found {len(markets)} active markets")

        # Filter to binary markets only
        binary_markets = [m for m in markets if len(m.get("outcomes", [])) == 2]
        print(f"Binary markets: {len(binary_markets)}")

        # Scan for opportunities
        print("\nScanning for arbitrage opportunities...")
        opportunities = await strategy.detect(binary_markets)

        if not opportunities:
            print("\nNo opportunities found. This is normal - arbitrage is competitive!")
            print("Try running during high volatility periods or market events.")
            return

        # Display top opportunities
        print(f"\nFound {len(opportunities)} opportunities:\n")

        for i, opp in enumerate(opportunities[:5], 1):
            print(f"{i}. {opp.market_question[:60]}...")
            print(f"   Prices: YES=${opp.prices.get('Yes', 0):.4f}, NO=${opp.prices.get('No', 0):.4f}")
            print(f"   Total Cost: ${sum(opp.prices.values()):.4f}")
            print(f"   Profit Margin: {opp.profit_margin * 100:.4f}%")
            print(f"   On $100: profit ${opp.profit_amount:.4f}")
            print()

    finally:
        await gamma.close()


# Bonus: Manual calculation example
def calculate_example():
    print("=" * 60)
    print("Manual Calculation Example")
    print("=" * 60)

    # Example: If you see YES at $0.48 and NO at $0.48
    yes_price = 0.48
    no_price = 0.48
    investment = 100  # USDC

    result = check_binary_arb(yes_price, no_price, investment)

    print(f"\nExample: YES=${yes_price}, NO=${no_price}")
    print(f"Total cost per share pair: ${result['total_cost']:.4f}")
    print(f"Profit margin: {result['profit_margin_percent']:.2f}%")
    print(f"Is profitable: {result['is_profitable']}")

    if result['is_profitable']:
        print(f"\nWith ${investment} investment:")
        print(f"  Shares purchased: {result['shares']:.4f}")
        print(f"  Guaranteed return: ${result['return']:.4f}")
        print(f"  Profit: ${result['profit']:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
    print()
    calculate_example()
