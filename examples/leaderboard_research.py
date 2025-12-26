#!/usr/bin/env python3
"""
Example: Research successful Polymarket traders.
Analyzes leaderboard to identify potential arbitrage strategies.
"""

import asyncio
import sys
sys.path.insert(0, "..")

from src.utils.leaderboard import LeaderboardScraper, analyze_top_traders


async def main():
    print("=" * 60)
    print("Polymarket Leaderboard Research")
    print("=" * 60)

    scraper = LeaderboardScraper()

    try:
        # Fetch top traders
        print("\nFetching top 100 traders...")
        traders = await scraper.fetch_top_traders(100)

        if not traders:
            print("Failed to fetch leaderboard")
            return

        # Display top 10
        print("\nTop 10 Traders:")
        print("-" * 80)

        for i, trader in enumerate(traders[:10], 1):
            print(f"{i:2}. {trader}")

        # Find likely arbitrage traders
        print("\n" + "=" * 60)
        print("Identifying Potential Arbitrage Traders")
        print("(High trade count, consistent profits)")
        print("=" * 60)

        arb_traders = await scraper.find_arbitrage_traders(
            min_trades=500,
            min_profit=5000,
        )

        if arb_traders:
            print(f"\nFound {len(arb_traders)} potential arbitrage traders:\n")

            for trader in arb_traders[:5]:
                print(f"  {trader.username or trader.address[:12]}...")
                print(f"    Trades: {trader.trades_count:,}")
                print(f"    Profit: ${trader.profit:,.2f}")
                print(f"    ROI: {trader.roi:.1f}%")
                print()

            # Analyze top arb trader
            if arb_traders:
                top_arb = arb_traders[0]
                print(f"\nDetailed analysis of {top_arb.username or top_arb.address[:12]}...")
                analysis = await scraper.analyze_trader(top_arb.address)

                print(f"  Total positions: {analysis.get('total_positions', 0)}")
                print(f"  Unique markets: {analysis.get('unique_markets', 0)}")
                print(f"  Total value: ${analysis.get('total_value', 0):,.2f}")

                outcomes = analysis.get('outcome_distribution', {})
                print(f"  YES positions: {outcomes.get('Yes', 0)}")
                print(f"  NO positions: {outcomes.get('No', 0)}")

                strategies = analysis.get('detected_strategies', [])
                if strategies:
                    print(f"  Detected strategies: {', '.join(strategies)}")
        else:
            print("No arbitrage traders found matching criteria")

        # Export option
        print("\n" + "-" * 60)
        export = input("Export full leaderboard to CSV? (y/n): ").lower()
        if export == 'y':
            await scraper.export_leaderboard_csv("leaderboard_export.csv", 500)
            print("Exported to leaderboard_export.csv")

    finally:
        await scraper.close()


if __name__ == "__main__":
    asyncio.run(main())
