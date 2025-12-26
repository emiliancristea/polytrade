#!/usr/bin/env python3
"""
Polymarket Arbitrage Bot - Quick Start Script
Run this to quickly scan for opportunities without full setup.
"""

import asyncio
from rich.console import Console
from rich.table import Table

console = Console()


async def quick_scan():
    """Run a quick scan for arbitrage opportunities."""
    console.print("[bold blue]Polymarket Arbitrage Scanner[/bold blue]")
    console.print("[dim]Quick scan mode - no wallet required[/dim]\n")

    # Import here to avoid issues if deps not installed
    try:
        from src.api.gamma_api import GammaAPI
        from src.strategies.binary_arb import BinaryArbitrageStrategy
        from src.strategies.multi_outcome_arb import MultiOutcomeArbitrageStrategy
    except ImportError as e:
        console.print(f"[red]Import error: {e}[/red]")
        console.print("[yellow]Run: pip install -r requirements.txt[/yellow]")
        return

    gamma = GammaAPI()
    binary_strategy = BinaryArbitrageStrategy(min_profit_margin=0.001, use_live_prices=False)
    multi_strategy = MultiOutcomeArbitrageStrategy(min_profit_margin=0.001, use_live_prices=False)

    try:
        # Fetch markets
        console.print("Fetching active markets...")
        markets = await gamma.get_all_active_markets()
        console.print(f"Found {len(markets)} active markets\n")

        # Scan for opportunities
        console.print("Scanning for binary arbitrage...")
        binary_opps = await binary_strategy.detect(markets)

        console.print("Scanning for multi-outcome arbitrage...")
        multi_opps = await multi_strategy.detect(markets)

        all_opps = binary_opps + multi_opps
        all_opps.sort(key=lambda x: x.profit_margin, reverse=True)

        if not all_opps:
            console.print("[yellow]No arbitrage opportunities found at this time.[/yellow]")
            console.print("[dim]This is normal - opportunities are rare and captured quickly by bots.[/dim]")
            return

        # Display results
        table = Table(title=f"Found {len(all_opps)} Opportunities")
        table.add_column("Type", style="cyan")
        table.add_column("Market", style="white", max_width=50)
        table.add_column("Profit %", style="green", justify="right")
        table.add_column("Prices", style="dim")

        for opp in all_opps[:10]:
            prices_str = " / ".join(f"{k}:{v:.3f}" for k, v in opp.prices.items())
            table.add_row(
                opp.strategy_type.value,
                opp.market_question[:50] + "...",
                f"{opp.profit_margin * 100:.4f}%",
                prices_str[:30],
            )

        console.print(table)

        # Show best opportunity details
        if all_opps:
            best = all_opps[0]
            console.print(f"\n[bold green]Best Opportunity:[/bold green]")
            console.print(f"  Market: {best.market_question}")
            console.print(f"  Type: {best.strategy_type.value}")
            console.print(f"  Profit Margin: {best.profit_margin * 100:.4f}%")
            console.print(f"  On $100 investment: ${best.profit_amount:.4f} profit")
            console.print(f"  Prices: {best.prices}")

    finally:
        await gamma.close()


async def demo_calculation():
    """Demonstrate arbitrage calculations."""
    console.print("\n[bold cyan]Arbitrage Calculation Examples[/bold cyan]")

    from src.strategies.binary_arb import check_binary_arb
    from src.strategies.endgame_arb import calculate_endgame_returns

    # Binary example
    console.print("\n[underline]Binary Arbitrage:[/underline]")
    console.print("If YES=$0.45 and NO=$0.50:")
    result = check_binary_arb(0.45, 0.50, 100)
    console.print(f"  Total cost: ${result['total_cost']:.2f}")
    console.print(f"  Profit margin: {result['profit_margin_percent']:.2f}%")
    console.print(f"  On $100: ${result['profit']:.2f} profit")

    # Endgame example
    console.print("\n[underline]Endgame Arbitrage:[/underline]")
    console.print("If 98% probability, resolves in 2 days:")
    result = calculate_endgame_returns(0.98, 2, 100)
    console.print(f"  ROI: {result['roi_percent']:.2f}%")
    console.print(f"  Annualized: {result['annualized_return_percent']:.1f}%")
    console.print(f"  Profit if correct: ${result['profit_if_correct']:.2f}")
    console.print(f"  Loss if wrong: ${result['loss_if_wrong']:.2f}")
    console.print(f"  Expected value: ${result['expected_value']:.2f}")


async def main():
    """Main entry point."""
    await quick_scan()
    await demo_calculation()

    console.print("\n[bold]Next Steps:[/bold]")
    console.print("1. Copy .env.example to .env and add your wallet")
    console.print("2. Run: python main.py scan")
    console.print("3. Run: python main.py run --dry-run")
    console.print("\nFor help: python main.py --help")


if __name__ == "__main__":
    asyncio.run(main())
