"""
Example integration of CircuitBreaker with ResearchWorkflow.

This module demonstrates how to properly integrate the circuit breaker
with the existing ResearchWorkflow, using AgentType enums directly as keys.
"""

import asyncio
from typing import Any

import logfire

from src.agents.base import ResearchDependencies
from src.agents.factory import AgentFactory, AgentType
from src.core.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerError,
    CircuitState,
    MetricsCollector,
)

logger = logfire


class WorkflowMetricsCollector(MetricsCollector):
    """Custom metrics collector for workflow observability."""

    def __init__(self):
        self.state_changes: list[tuple[AgentType, CircuitState, CircuitState]] = []
        self.attempts: list[tuple[AgentType, bool]] = []

    async def record_state_change(
        self, key: AgentType, old_state: CircuitState, new_state: CircuitState
    ) -> None:
        """Record state change for monitoring."""
        self.state_changes.append((key, old_state, new_state))
        logger.info(
            "Circuit breaker state change",
            agent=key.value,
            old_state=old_state.name,
            new_state=new_state.name,
        )

    async def record_attempt(self, key: AgentType, success: bool) -> None:
        """Record attempt for analytics."""
        self.attempts.append((key, success))
        logger.debug(
            "Agent attempt recorded",
            agent=key.value,
            success=success,
        )

    def get_summary(self) -> dict[str, Any]:
        """Get summary of collected metrics."""
        total_attempts = len(self.attempts)
        successful_attempts = sum(1 for _, success in self.attempts if success)
        return {
            "total_attempts": total_attempts,
            "successful_attempts": successful_attempts,
            "success_rate": successful_attempts / total_attempts if total_attempts > 0 else 0,
            "state_changes": len(self.state_changes),
        }


class ResearchWorkflowWithCircuitBreaker:
    """
    Enhanced ResearchWorkflow with proper circuit breaker integration.

    This class demonstrates:
    1. Using AgentType enum directly as dictionary keys
    2. Per-agent circuit breaker configuration
    3. Graceful degradation strategies
    4. Monitoring and observability
    """

    def __init__(
        self,
        default_config: CircuitBreakerConfig | None = None,
    ):
        """
        Initialize workflow with circuit breaker protection.

        Args:
            default_config: Default configuration for circuit breakers
        """
        self.agent_factory = AgentFactory
        self.metrics_collector = WorkflowMetricsCollector()

        # Default configuration
        self.default_config = default_config or CircuitBreakerConfig(
            failure_threshold=3,
            success_threshold=2,
            timeout_seconds=30.0,
            half_open_max_attempts=2,
        )

        # Per-agent circuit breaker configurations
        self.agent_configs = self._create_agent_configs()

        # Create circuit breaker with AgentType as key type
        self.circuit_breaker: CircuitBreaker[AgentType] = CircuitBreaker(
            config=self.default_config,
            metrics_collector=self.metrics_collector,
            fallback_factory=self._create_fallback,
        )

        logger.info(
            "ResearchWorkflow initialized with circuit breaker",
            default_threshold=self.default_config.failure_threshold,
            agents=list(self.agent_configs.keys()),
        )

    def _create_agent_configs(self) -> dict[AgentType, dict[str, Any]]:
        """Create agent-specific configurations."""
        return {
            # Critical agents - more lenient settings
            AgentType.RESEARCH_EXECUTOR: {
                "critical": True,
                "config": CircuitBreakerConfig(
                    failure_threshold=5,
                    success_threshold=2,
                    timeout_seconds=60.0,
                    half_open_max_attempts=3,
                    name="critical_research_executor",
                ),
            },
            # Important agents - balanced settings
            AgentType.REPORT_GENERATOR: {
                "critical": False,
                "config": CircuitBreakerConfig(
                    failure_threshold=3,
                    success_threshold=2,
                    timeout_seconds=45.0,
                    half_open_max_attempts=2,
                    name="important_report_generator",
                ),
            },
            AgentType.BRIEF_GENERATOR: {
                "critical": False,
                "config": CircuitBreakerConfig(
                    failure_threshold=3,
                    success_threshold=2,
                    timeout_seconds=45.0,
                    half_open_max_attempts=2,
                    name="important_brief_generator",
                ),
            },
            # Optional agents - fail fast
            AgentType.QUERY_TRANSFORMATION: {
                "critical": False,
                "config": CircuitBreakerConfig(
                    failure_threshold=2,
                    success_threshold=1,
                    timeout_seconds=30.0,
                    half_open_max_attempts=1,
                    name="optional_query_transformation",
                ),
            },
            # Supporting agents
            AgentType.CLARIFICATION: {
                "critical": False,
                "config": CircuitBreakerConfig(
                    failure_threshold=3,
                    success_threshold=2,
                    timeout_seconds=30.0,
                    half_open_max_attempts=2,
                    name="support_clarification",
                ),
            },
            AgentType.COMPRESSION: {
                "critical": False,
                "config": CircuitBreakerConfig(
                    failure_threshold=3,
                    success_threshold=2,
                    timeout_seconds=30.0,
                    half_open_max_attempts=2,
                    name="support_compression",
                ),
            },
        }

    async def _create_fallback(self, agent_type: AgentType) -> Any:
        """Create fallback response for failed agent."""
        logger.info(f"Using fallback for {agent_type.value}")

        fallback_responses = {
            AgentType.QUERY_TRANSFORMATION: {
                "transformed_query": "original query (transformation unavailable)",
                "confidence": 0.0,
                "fallback": True,
            },
            AgentType.RESEARCH_EXECUTOR: {
                "results": [],
                "from_cache": True,
                "fallback": True,
            },
            AgentType.REPORT_GENERATOR: {
                "report": "Report generation is temporarily unavailable. Please try again later.",
                "fallback": True,
            },
            AgentType.CLARIFICATION: {
                "clarification": None,
                "proceed": True,
                "fallback": True,
            },
            AgentType.COMPRESSION: {
                "compressed": False,
                "data": "uncompressed",
                "fallback": True,
            },
            AgentType.BRIEF_GENERATOR: {
                "brief": "Brief generation unavailable",
                "fallback": True,
            },
        }

        return fallback_responses.get(
            agent_type,
            {"error": f"No fallback for {agent_type.value}", "fallback": True},
        )

    async def execute_agent_with_circuit_breaker(
        self,
        agent_type: AgentType,
        deps: ResearchDependencies,
        **kwargs: Any,
    ) -> Any:
        """
        Execute an agent operation with circuit breaker protection.

        Args:
            agent_type: Type of agent to execute (using enum directly)
            deps: Research dependencies
            **kwargs: Additional arguments for agent

        Returns:
            Result from the agent operation or fallback

        Raises:
            CircuitBreakerError: If circuit is open for critical agents
            Exception: Any exception from the agent operation
        """
        agent_config = self.agent_configs.get(agent_type, {})
        is_critical = agent_config.get("critical", False)

        # Check if circuit is open for critical agents
        if is_critical and self.circuit_breaker.is_open(agent_type):
            raise CircuitBreakerError(
                f"Critical agent {agent_type.value} is unavailable due to circuit breaker"
            )

        try:
            # Create agent instance
            agent = self.agent_factory.create_agent(agent_type, deps)

            # Execute through circuit breaker with enum key
            result = await self.circuit_breaker.call(agent_type, agent.run, deps, **kwargs)

            logger.debug(f"Successfully executed {agent_type.value}")
            return result

        except CircuitBreakerError as e:
            # Circuit is open
            logger.warning(f"Circuit breaker open for {agent_type.value}: {e}")

            if not is_critical:
                # Use fallback for non-critical agents
                return await self._create_fallback(agent_type)
            raise

        except Exception as e:
            logger.error(f"Error executing {agent_type.value}: {e}")
            raise

    async def execute_workflow(
        self,
        user_query: str,
        deps: ResearchDependencies,
    ) -> dict[str, Any]:
        """
        Execute the complete research workflow with circuit breaker protection.

        Args:
            user_query: Research query to process
            deps: Research dependencies

        Returns:
            Workflow results with circuit breaker status
        """
        results = {}

        try:
            # Step 1: Clarification (optional)
            try:
                clarification = await self.execute_agent_with_circuit_breaker(
                    AgentType.CLARIFICATION,
                    deps,
                    query=user_query,
                )
                results["clarification"] = clarification
            except CircuitBreakerError:
                logger.info("Proceeding without clarification due to circuit breaker")
                results["clarification"] = None

            # Step 2: Query Transformation (optional)
            try:
                transformed = await self.execute_agent_with_circuit_breaker(
                    AgentType.QUERY_TRANSFORMATION,
                    deps,
                    query=user_query,
                )
                results["transformed_query"] = transformed
                query_to_use = transformed.get("query", user_query)
            except CircuitBreakerError:
                logger.info("Using original query due to circuit breaker")
                query_to_use = user_query

            # Step 3: Brief Generation (important)
            brief = await self.execute_agent_with_circuit_breaker(
                AgentType.BRIEF_GENERATOR,
                deps,
                query=query_to_use,
            )
            results["brief"] = brief

            # Step 4: Research Execution (critical)
            research_results = await self.execute_agent_with_circuit_breaker(
                AgentType.RESEARCH_EXECUTOR,
                deps,
                brief=brief,
            )
            results["research"] = research_results

            # Step 5: Compression (optional)
            try:
                compressed = await self.execute_agent_with_circuit_breaker(
                    AgentType.COMPRESSION,
                    deps,
                    data=research_results,
                )
                results["compressed"] = compressed
                data_for_report = compressed
            except CircuitBreakerError:
                logger.info("Skipping compression due to circuit breaker")
                data_for_report = research_results

            # Step 6: Report Generation (important)
            report = await self.execute_agent_with_circuit_breaker(
                AgentType.REPORT_GENERATOR,
                deps,
                data=data_for_report,
            )
            results["report"] = report

        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            results["error"] = str(e)

        # Add circuit breaker status to results
        results["circuit_breaker_status"] = self.get_circuit_breaker_status()
        results["metrics_summary"] = self.metrics_collector.get_summary()

        return results

    def get_circuit_breaker_status(self) -> dict[AgentType, dict[str, Any]]:
        """Get status of all circuit breakers in the workflow."""
        status = {}
        all_metrics = self.circuit_breaker.get_all_metrics()

        for agent_type in AgentType:
            state = self.circuit_breaker.get_state(agent_type)
            metrics = all_metrics.get(agent_type)

            status[agent_type] = {
                "state": state.name if state else "NOT_INITIALIZED",
                "is_critical": self.agent_configs.get(agent_type, {}).get("critical", False),
            }

            if metrics:
                status[agent_type].update(
                    {
                        "total_attempts": metrics.total_attempts,
                        "success_rate": metrics.success_rate,
                        "rejected_calls": metrics.rejected_calls,
                        "consecutive_errors": metrics.consecutive_errors,
                    }
                )

        return status

    async def reset_circuit_breakers(self, agent_type: AgentType | None = None):
        """
        Reset circuit breakers.

        Args:
            agent_type: Specific agent to reset, or None to reset all
        """
        if agent_type:
            await self.circuit_breaker.reset(agent_type)
            logger.info(f"Circuit breaker reset for {agent_type.value}")
        else:
            await self.circuit_breaker.reset()
            logger.info("All circuit breakers reset")

    async def health_check(self) -> dict[str, Any]:
        """
        Perform health check on all agents.

        Returns:
            Health status of all agents
        """
        health: dict[str, Any] = {
            "timestamp": asyncio.get_event_loop().time(),
            "agents": {},
        }

        for agent_type in AgentType:
            state = self.circuit_breaker.get_state(agent_type)
            metrics = self.circuit_breaker.get_metrics(agent_type)

            health["agents"][agent_type.value] = {
                "healthy": state != CircuitState.OPEN if state else True,
                "state": state.name if state else "NOT_INITIALIZED",
                "success_rate": metrics.success_rate if metrics else 1.0,
            }

        health["overall_health"] = all(agent["healthy"] for agent in health["agents"].values())

        return health


class AdaptiveCircuitBreakerConfig:
    """
    Adaptive circuit breaker configuration that adjusts based on agent importance.
    """

    @staticmethod
    def get_config_for_agent(agent_type: AgentType) -> CircuitBreakerConfig:
        """
        Get appropriate circuit breaker config based on agent type.

        Critical agents get more lenient settings, while optional agents
        get stricter settings to fail fast.
        """
        configs = {
            # Critical agents - more retries, longer timeout
            AgentType.RESEARCH_EXECUTOR: CircuitBreakerConfig(
                failure_threshold=5,
                success_threshold=2,
                timeout_seconds=60.0,
                half_open_max_attempts=3,
                name=f"critical_{agent_type.value}",
            ),
            # Important agents - balanced settings
            AgentType.REPORT_GENERATOR: CircuitBreakerConfig(
                failure_threshold=3,
                success_threshold=2,
                timeout_seconds=45.0,
                half_open_max_attempts=2,
                name=f"important_{agent_type.value}",
            ),
            AgentType.BRIEF_GENERATOR: CircuitBreakerConfig(
                failure_threshold=3,
                success_threshold=2,
                timeout_seconds=45.0,
                half_open_max_attempts=2,
                name=f"important_{agent_type.value}",
            ),
            # Optional agents - fail fast
            AgentType.QUERY_TRANSFORMATION: CircuitBreakerConfig(
                failure_threshold=2,
                success_threshold=1,
                timeout_seconds=30.0,
                half_open_max_attempts=1,
                name=f"optional_{agent_type.value}",
            ),
            # Supporting agents - balanced
            AgentType.CLARIFICATION: CircuitBreakerConfig(
                failure_threshold=3,
                success_threshold=2,
                timeout_seconds=30.0,
                half_open_max_attempts=2,
                name=f"support_{agent_type.value}",
            ),
            AgentType.COMPRESSION: CircuitBreakerConfig(
                failure_threshold=3,
                success_threshold=2,
                timeout_seconds=30.0,
                half_open_max_attempts=2,
                name=f"support_{agent_type.value}",
            ),
        }

        return configs.get(
            agent_type,
            CircuitBreakerConfig(name=f"default_{agent_type.value}"),
        )


# Example usage
async def example_usage():
    """Demonstrate circuit breaker usage with workflow."""
    from src.models.api_models import APIKeys
    from src.models.core import ResearchState

    # Create workflow with circuit breaker
    workflow = ResearchWorkflowWithCircuitBreaker()

    # Create mock dependencies
    import httpx

    async with httpx.AsyncClient() as http_client:
        deps = ResearchDependencies(
            http_client=http_client,
            api_keys=APIKeys(),
            research_state=ResearchState(
                request_id="test-123",
                user_id="user-1",
                session_id="session-1",
                user_query="What is circuit breaker pattern?",
            ),
        )

        # Execute workflow
        results = await workflow.execute_workflow("What is circuit breaker pattern?", deps)

        # Check status
        print("Circuit Breaker Status:")
        for agent_type, status in results["circuit_breaker_status"].items():
            print(f"  {agent_type.value}: {status}")

        # Health check
        health = await workflow.health_check()
        print(f"\nOverall Health: {health['overall_health']}")

        # Reset if needed
        if not health["overall_health"]:
            await workflow.reset_circuit_breakers()
            print("Circuit breakers reset")


if __name__ == "__main__":
    asyncio.run(example_usage())
