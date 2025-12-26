"""
Order Executor Module.
Handles safe execution of arbitrage trades with pre-trade checks.
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from loguru import logger

from src.api.clob_client import PolymarketClient
from src.config import config
from src.strategies.base import ArbitrageOpportunity, StrategyType
from .risk_manager import RiskManager


class ExecutionResult:
    """Result of order execution attempt."""

    def __init__(
        self,
        success: bool,
        opportunity: ArbitrageOpportunity,
        orders_placed: List[Dict[str, Any]] = None,
        error: Optional[str] = None,
        spent: float = 0.0,
        shares_received: float = 0.0,
    ):
        self.success = success
        self.opportunity = opportunity
        self.orders_placed = orders_placed or []
        self.error = error
        self.spent = spent
        self.shares_received = shares_received
        self.executed_at = datetime.now()

    def __str__(self) -> str:
        if self.success:
            return (
                f"SUCCESS: Executed {len(self.orders_placed)} orders, "
                f"spent ${self.spent:.2f}, received {self.shares_received:.4f} shares"
            )
        return f"FAILED: {self.error}"


class OrderExecutor:
    """
    Executes trades on Polymarket with safety checks.
    """

    def __init__(self, risk_manager: Optional[RiskManager] = None):
        """
        Initialize order executor.

        Args:
            risk_manager: Optional risk manager instance
        """
        self.risk_manager = risk_manager or RiskManager()
        self._client: Optional[PolymarketClient] = None
        self._initialized = False

        # Execution stats
        self.total_executions = 0
        self.successful_executions = 0
        self.failed_executions = 0

    def _get_client(self) -> PolymarketClient:
        """Get authenticated trading client."""
        if self._client is None:
            self._client = PolymarketClient(read_only=config.trading.dry_run)
        return self._client

    async def execute(
        self,
        opportunity: ArbitrageOpportunity,
        position_size: Optional[float] = None,
    ) -> ExecutionResult:
        """
        Execute an arbitrage opportunity.

        Args:
            opportunity: The opportunity to execute
            position_size: Override position size (uses risk manager if None)

        Returns:
            ExecutionResult with details of the execution
        """
        self.total_executions += 1

        # Validate opportunity
        is_valid, reason = self.risk_manager.validate_opportunity(opportunity)
        if not is_valid:
            self.failed_executions += 1
            return ExecutionResult(
                success=False,
                opportunity=opportunity,
                error=f"Validation failed: {reason}"
            )

        # Check if we can trade
        can_trade, reason = self.risk_manager.can_trade(opportunity)
        if not can_trade:
            self.failed_executions += 1
            return ExecutionResult(
                success=False,
                opportunity=opportunity,
                error=f"Trade blocked: {reason}"
            )

        # Determine position size
        if position_size is None:
            position_size = self.risk_manager.calculate_position_size(
                opportunity,
                config.trading.max_position_size
            )

        if position_size <= 0:
            self.failed_executions += 1
            return ExecutionResult(
                success=False,
                opportunity=opportunity,
                error="Position size too small"
            )

        # Log execution attempt
        logger.info(
            f"Executing {opportunity.strategy_type.value} opportunity: "
            f"${position_size:.2f} -> expected profit ${opportunity.profit_amount:.4f}"
        )

        # Dry run mode
        if config.trading.dry_run:
            return self._simulate_execution(opportunity, position_size)

        # Real execution
        try:
            result = await self._execute_orders(opportunity, position_size)
            if result.success:
                self.successful_executions += 1
                self.risk_manager.record_trade(
                    opportunity,
                    result.spent,
                    result.shares_received
                )
            else:
                self.failed_executions += 1
            return result

        except Exception as e:
            self.failed_executions += 1
            logger.error(f"Execution error: {e}")
            return ExecutionResult(
                success=False,
                opportunity=opportunity,
                error=str(e)
            )

    async def _execute_orders(
        self,
        opportunity: ArbitrageOpportunity,
        position_size: float,
    ) -> ExecutionResult:
        """Execute the actual orders on Polymarket."""
        client = self._get_client()
        orders_placed = []
        total_spent = 0.0
        total_shares = 0.0

        # For arbitrage, we need to execute all trades atomically
        # Using Fill-or-Kill (FOK) orders to ensure all-or-nothing execution

        for trade in opportunity.trades:
            token_id = trade.get("token_id")
            price = trade.get("price", 0)
            side = trade.get("side", "BUY")

            if not token_id or price <= 0:
                continue

            # Calculate shares for this leg
            leg_investment = position_size / len(opportunity.trades)
            shares = leg_investment / price

            try:
                # Create and post order
                if side == "BUY":
                    signed_order = client.create_limit_order(
                        token_id=token_id,
                        side="BUY",
                        price=min(price * (1 + config.trading.max_slippage), 1.0),  # Add slippage buffer
                        size=shares,
                    )
                else:
                    signed_order = client.create_limit_order(
                        token_id=token_id,
                        side="SELL",
                        price=max(price * (1 - config.trading.max_slippage), 0.01),
                        size=shares,
                    )

                if signed_order is None:
                    raise Exception(f"Failed to create order for {token_id}")

                # Post with FOK for immediate fill
                response = client.post_order(signed_order, order_type="FOK")

                if response:
                    orders_placed.append({
                        "token_id": token_id,
                        "side": side,
                        "price": price,
                        "shares": shares,
                        "response": response,
                    })
                    total_spent += leg_investment
                    total_shares += shares
                else:
                    raise Exception(f"Order rejected for {token_id}")

            except Exception as e:
                logger.error(f"Order failed for {token_id}: {e}")
                # If any order fails in an arb, we should cancel others
                # (in production, implement proper rollback)
                return ExecutionResult(
                    success=False,
                    opportunity=opportunity,
                    orders_placed=orders_placed,
                    error=f"Order execution failed: {e}",
                    spent=total_spent,
                    shares_received=total_shares,
                )

        return ExecutionResult(
            success=True,
            opportunity=opportunity,
            orders_placed=orders_placed,
            spent=total_spent,
            shares_received=total_shares,
        )

    def _simulate_execution(
        self,
        opportunity: ArbitrageOpportunity,
        position_size: float,
    ) -> ExecutionResult:
        """Simulate execution for dry run mode."""
        logger.info("[DRY RUN] Simulating execution...")

        simulated_orders = []
        total_shares = 0.0

        for trade in opportunity.trades:
            price = trade.get("price", 0)
            if price <= 0:
                continue

            leg_investment = position_size / len(opportunity.trades)
            shares = leg_investment / price
            total_shares += shares

            simulated_orders.append({
                "token_id": trade.get("token_id"),
                "side": trade.get("side"),
                "price": price,
                "shares": shares,
                "status": "SIMULATED",
            })

        expected_profit = (total_shares * opportunity.profit_margin) if opportunity.strategy_type in [
            StrategyType.BINARY, StrategyType.MULTI_OUTCOME
        ] else opportunity.profit_amount * (position_size / 100)

        logger.info(
            f"[DRY RUN] Would execute {len(simulated_orders)} orders:\n"
            f"  Investment: ${position_size:.2f}\n"
            f"  Shares: {total_shares:.4f}\n"
            f"  Expected Profit: ${expected_profit:.4f}\n"
            f"  ROI: {opportunity.roi_percent:.2f}%"
        )

        self.successful_executions += 1
        return ExecutionResult(
            success=True,
            opportunity=opportunity,
            orders_placed=simulated_orders,
            spent=position_size,
            shares_received=total_shares,
        )

    async def execute_batch(
        self,
        opportunities: List[ArbitrageOpportunity],
        max_concurrent: int = 3,
    ) -> List[ExecutionResult]:
        """
        Execute multiple opportunities (with rate limiting).

        Args:
            opportunities: List of opportunities to execute
            max_concurrent: Maximum concurrent executions

        Returns:
            List of execution results
        """
        results = []

        # Process in batches
        for i in range(0, len(opportunities), max_concurrent):
            batch = opportunities[i:i + max_concurrent]
            tasks = [self.execute(opp) for opp in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Batch execution error: {result}")
                else:
                    results.append(result)

            # Cooldown between batches
            if i + max_concurrent < len(opportunities):
                await asyncio.sleep(config.risk.trade_cooldown)

        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get executor statistics."""
        return {
            "total_executions": self.total_executions,
            "successful": self.successful_executions,
            "failed": self.failed_executions,
            "success_rate": (
                self.successful_executions / self.total_executions * 100
                if self.total_executions > 0 else 0
            ),
            "risk_stats": self.risk_manager.get_stats(),
        }
