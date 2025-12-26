#!/usr/bin/env python3
"""
Polymarket Arbitrage Trading Bot
Main entry point with CLI interface.
"""

import asyncio
import sys
from datetime import datetime
from typing import Optional

import click
from loguru import logger
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel

from src.config import config
from src.api.gamma_api import GammaAPI
from src.strategies.detector import ArbitrageDetector
from src.strategies.base import RiskLevel, ArbitrageOpportunity
from src.execution.executor import OrderExecutor
from src.execution.risk_manager import RiskManager
from src.monitoring.price_tracker import PriceTracker
from src.utils.leaderboard import LeaderboardScraper
from src.utils.notifications import NotificationService

console = Console()


def setup_logging(level: str = "INFO"):
    """Configure logging."""
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
    )
    logger.add(
        "logs/bot_{time:YYYY-MM-DD}.log",
        level="DEBUG",
        rotation="1 day",
        retention="7 days",
    )


@click.group()
@click.option("--debug", is_flag=True, help="Enable debug logging")
def cli(debug: bool):
    """Polymarket Arbitrage Trading Bot"""
    setup_logging("DEBUG" if debug else config.monitoring.log_level)


@cli.command()
@click.option("--min-profit", default=0.5, help="Minimum profit margin (%)")
@click.option("--max-risk", type=click.Choice(["low", "medium", "high"]), default="high")
@click.option("--limit", default=20, help="Maximum opportunities to show")
def scan(min_profit: float, max_risk: str, limit: int):
    """Scan for arbitrage opportunities (one-time)."""
    console.print("[bold blue]Scanning for arbitrage opportunities...[/bold blue]")

    async def run_scan():
        detector = ArbitrageDetector(min_profit_margin=min_profit / 100)
        risk = RiskLevel[max_risk.upper()]

        try:
            opportunities = await detector.scan(max_risk=risk)

            if not opportunities:
                console.print("[yellow]No opportunities found matching criteria.[/yellow]")
                return

            # Display results
            table = Table(title=f"Found {len(opportunities)} Opportunities")
            table.add_column("Type", style="cyan")
            table.add_column("Market", style="white", max_width=40)
            table.add_column("Profit %", style="green", justify="right")
            table.add_column("Est. Profit", style="green", justify="right")
            table.add_column("Risk", justify="center")
            table.add_column("Confidence", justify="right")

            for opp in opportunities[:limit]:
                risk_color = {
                    RiskLevel.LOW: "green",
                    RiskLevel.MEDIUM: "yellow",
                    RiskLevel.HIGH: "red",
                }[opp.risk_level]

                table.add_row(
                    opp.strategy_type.value.upper(),
                    opp.market_question[:40] + "...",
                    f"{opp.profit_margin * 100:.3f}%",
                    f"${opp.profit_amount:.4f}",
                    f"[{risk_color}]{opp.risk_level.value}[/{risk_color}]",
                    f"{opp.confidence * 100:.1f}%",
                )

            console.print(table)

        finally:
            await detector.close()

    asyncio.run(run_scan())


@cli.command()
@click.option("--interval", default=5, help="Scan interval in seconds")
@click.option("--auto-execute", is_flag=True, help="Automatically execute opportunities")
@click.option("--dry-run", is_flag=True, default=True, help="Dry run mode (no real trades)")
def run(interval: int, auto_execute: bool, dry_run: bool):
    """Run the arbitrage bot continuously."""
    console.print(Panel.fit(
        "[bold green]Starting Polymarket Arbitrage Bot[/bold green]\n"
        f"Mode: {'DRY RUN' if dry_run else 'LIVE TRADING'}\n"
        f"Auto-execute: {auto_execute}\n"
        f"Scan interval: {interval}s",
        title="Bot Configuration"
    ))

    if not dry_run and auto_execute:
        console.print("[bold red]WARNING: Live trading with auto-execute enabled![/bold red]")
        if not click.confirm("Are you sure you want to continue?"):
            return

    # Override config
    config.trading.dry_run = dry_run

    async def run_bot():
        detector = ArbitrageDetector()
        executor = OrderExecutor() if auto_execute else None
        notification = NotificationService()

        try:
            iteration = 0
            while True:
                iteration += 1
                console.print(f"\n[dim]Scan #{iteration} at {datetime.now().strftime('%H:%M:%S')}[/dim]")

                opportunities = await detector.scan()

                if opportunities:
                    console.print(f"[green]Found {len(opportunities)} opportunities[/green]")

                    for opp in opportunities[:3]:
                        console.print(
                            f"  [{opp.strategy_type.value}] "
                            f"{opp.profit_margin * 100:.3f}% - "
                            f"{opp.market_question[:50]}..."
                        )

                    if auto_execute and executor:
                        # Execute top opportunity
                        best = opportunities[0]
                        result = await executor.execute(best)
                        console.print(f"  Execution: {result}")

                        if notification.telegram_enabled:
                            await notification.send_trade_alert(
                                best, result.success, str(result)
                            )
                else:
                    console.print("[dim]No opportunities found[/dim]")

                await asyncio.sleep(interval)

        except KeyboardInterrupt:
            console.print("\n[yellow]Shutting down...[/yellow]")
        finally:
            await detector.close()

    asyncio.run(run_bot())


@cli.command()
@click.option("--use-websocket", is_flag=True, help="Use WebSocket for real-time updates")
def monitor(use_websocket: bool):
    """Monitor markets in real-time."""
    console.print("[bold blue]Starting real-time market monitor...[/bold blue]")

    async def run_monitor():
        if use_websocket:
            # WebSocket-based monitoring
            gamma = GammaAPI()
            tracker = PriceTracker(
                on_opportunity=lambda opp: console.print(
                    f"[green]OPPORTUNITY: {opp.profit_margin * 100:.3f}% - {opp.market_question[:50]}...[/green]"
                )
            )

            try:
                markets = await gamma.get_all_active_markets()
                await tracker.add_markets(markets[:100])  # Track top 100
                await tracker.start()
            finally:
                await tracker.stop()
                await gamma.close()
        else:
            # REST API polling
            detector = ArbitrageDetector()
            try:
                await detector.scan_continuous(
                    interval_seconds=5,
                    callback=lambda opps: console.print(
                        f"[cyan]Scan complete: {len(opps)} opportunities[/cyan]"
                    )
                )
            finally:
                await detector.close()

    asyncio.run(run_monitor())


@cli.command()
@click.option("--limit", default=50, help="Number of traders to fetch")
@click.option("--export", is_flag=True, help="Export to CSV")
def leaderboard(limit: int, export: bool):
    """Fetch and analyze the Polymarket leaderboard."""
    console.print("[bold blue]Fetching leaderboard...[/bold blue]")

    async def run_leaderboard():
        scraper = LeaderboardScraper()

        try:
            traders = await scraper.fetch_top_traders(limit)

            if not traders:
                console.print("[yellow]No traders found[/yellow]")
                return

            table = Table(title=f"Top {len(traders)} Polymarket Traders")
            table.add_column("Rank", justify="right")
            table.add_column("Username/Address", max_width=20)
            table.add_column("Profit", style="green", justify="right")
            table.add_column("Volume", justify="right")
            table.add_column("Trades", justify="right")
            table.add_column("ROI %", justify="right")

            for trader in traders[:20]:  # Show top 20
                table.add_row(
                    str(trader.rank),
                    trader.username or trader.address[:12] + "...",
                    f"${trader.profit:,.2f}",
                    f"${trader.volume:,.2f}",
                    f"{trader.trades_count:,}",
                    f"{trader.roi:.1f}%",
                )

            console.print(table)

            if export:
                await scraper.export_leaderboard_csv(f"leaderboard_{limit}.csv", limit)
                console.print(f"[green]Exported to leaderboard_{limit}.csv[/green]")

            # Find potential arb traders
            arb_traders = await scraper.find_arbitrage_traders()
            if arb_traders:
                console.print(f"\n[cyan]Found {len(arb_traders)} potential arbitrage traders[/cyan]")

        finally:
            await scraper.close()

    asyncio.run(run_leaderboard())


@cli.command()
@click.argument("address")
def analyze(address: str):
    """Analyze a specific trader's strategy."""
    console.print(f"[bold blue]Analyzing trader: {address[:20]}...[/bold blue]")

    async def run_analysis():
        scraper = LeaderboardScraper()

        try:
            analysis = await scraper.analyze_trader(address)

            if "error" in analysis:
                console.print(f"[red]{analysis['error']}[/red]")
                return

            table = Table(title="Trader Analysis")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="white")

            table.add_row("Address", address[:20] + "...")
            table.add_row("Total Positions", str(analysis.get("total_positions", 0)))
            table.add_row("Unique Markets", str(analysis.get("unique_markets", 0)))
            table.add_row("Total Value", f"${analysis.get('total_value', 0):,.2f}")
            table.add_row("Avg Position Size", f"${analysis.get('avg_position_size', 0):,.2f}")

            outcomes = analysis.get("outcome_distribution", {})
            table.add_row("YES Positions", str(outcomes.get("Yes", 0)))
            table.add_row("NO Positions", str(outcomes.get("No", 0)))

            if "avg_entry_price" in analysis:
                table.add_row("Avg Entry Price", f"${analysis['avg_entry_price']:.3f}")
                table.add_row("Entry Range", f"${analysis.get('min_entry_price', 0):.3f} - ${analysis.get('max_entry_price', 0):.3f}")

            strategies = analysis.get("detected_strategies", [])
            if strategies:
                table.add_row("Detected Strategies", ", ".join(strategies))

            console.print(table)

        finally:
            await scraper.close()

    asyncio.run(run_analysis())


@cli.command()
def check():
    """Check API connectivity and configuration."""
    console.print("[bold blue]Checking configuration and connectivity...[/bold blue]")

    from src.api.clob_client import PolymarketClient

    # Check config
    console.print("\n[cyan]Configuration:[/cyan]")
    console.print(f"  CLOB Host: {config.api.clob_host}")
    console.print(f"  Gamma API: {config.api.gamma_api_host}")
    console.print(f"  Chain ID: {config.api.chain_id}")
    console.print(f"  Dry Run: {config.trading.dry_run}")

    if config.wallet.wallet_address:
        console.print(f"  Wallet: {config.wallet.wallet_address[:10]}...{config.wallet.wallet_address[-6:]}")
    else:
        console.print("  [yellow]Wallet: Not configured[/yellow]")

    # Check CLOB connectivity
    console.print("\n[cyan]API Connectivity:[/cyan]")
    try:
        client = PolymarketClient(read_only=True)
        if client.check_connection():
            console.print("  [green]CLOB API: Connected[/green]")
            server_time = client.get_server_time()
            if server_time:
                console.print(f"  Server Time: {server_time}")
        else:
            console.print("  [red]CLOB API: Connection failed[/red]")
    except Exception as e:
        console.print(f"  [red]CLOB API: Error - {e}[/red]")

    # Check Gamma API
    async def check_gamma():
        gamma = GammaAPI()
        try:
            markets = await gamma.get_markets(limit=1)
            if markets:
                console.print("  [green]Gamma API: Connected[/green]")
                console.print(f"  Sample Market: {markets[0].get('question', 'N/A')[:50]}...")
            else:
                console.print("  [yellow]Gamma API: Connected but no markets[/yellow]")
        except Exception as e:
            console.print(f"  [red]Gamma API: Error - {e}[/red]")
        finally:
            await gamma.close()

    asyncio.run(check_gamma())

    console.print("\n[green]Configuration check complete![/green]")


@cli.command()
@click.option("--yes-price", type=float, required=True, help="YES share price")
@click.option("--no-price", type=float, required=True, help="NO share price")
@click.option("--investment", default=100.0, help="Investment amount in USDC")
def calculate(yes_price: float, no_price: float, investment: float):
    """Calculate potential arbitrage profit."""
    from src.strategies.binary_arb import check_binary_arb

    result = check_binary_arb(yes_price, no_price, investment)

    table = Table(title="Binary Arbitrage Calculator")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("YES Price", f"${yes_price:.4f}")
    table.add_row("NO Price", f"${no_price:.4f}")
    table.add_row("Total Cost", f"${result['total_cost']:.4f}")
    table.add_row("Investment", f"${investment:.2f}")
    table.add_row("Shares", f"{result['shares']:.4f}")

    if result["is_profitable"]:
        table.add_row("Profit Margin", f"[green]{result['profit_margin_percent']:.3f}%[/green]")
        table.add_row("Expected Profit", f"[green]${result['profit']:.4f}[/green]")
        table.add_row("Total Return", f"[green]${result['return']:.4f}[/green]")
    else:
        table.add_row("Status", "[red]NOT PROFITABLE (cost >= $1.00)[/red]")

    console.print(table)


if __name__ == "__main__":
    cli()
