#!/usr/bin/env python
"""Quick test runner for Research Executor Agent evaluation.

This script provides a streamlined way to evaluate the research executor agent
during development with configurable test categories and detailed reporting.
"""

import asyncio
import os
import sys
import json
import time
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Suppress logfire prompts
os.environ['LOGFIRE_IGNORE_NO_CONFIG'] = '1'

import httpx
from pydantic import SecretStr
import logfire
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.panel import Panel
from rich import print as rprint

from src.agents.research_executor import ResearchExecutorAgent
from src.agents.base import ResearchDependencies
from src.models.metadata import ResearchMetadata
from src.models.core import ResearchState, ResearchStage
from src.models.research_executor import ResearchResults
from src.models.api_models import APIKeys
from tests.evals.research_executor_evals import (
    ResearchExecutorInput,
    ResearchExecutorExpectedOutput,
    FindingsRelevanceEvaluator,
    SourceCredibilityEvaluator,
    InsightQualityEvaluator,
    DataGapIdentificationEvaluator,
    ComprehensiveEvaluator,
    ConfidenceCalibrationEvaluator,
    EvidenceSupportEvaluator,
    CategoryCoverageEvaluator,
    CrossReferenceEvaluator
)
from tests.evals.base_multi_judge import BaseMultiJudgeEvaluator, VotingMethod
from tests.evals.research_executor_multi_judge_adapter import ResearchExecutorMultiJudgeAdapter


# Configure Logfire (suppress for cleaner output)
logfire.configure(send_to_logfire=False, console=False)

# Initialize console for rich output
console = Console()


@dataclass
class TestCase:
    """Test case for research executor evaluation."""
    name: str
    category: str
    input: ResearchExecutorInput
    expected: ResearchExecutorExpectedOutput
    evaluators: List[str]
    description: str = ""


@dataclass
class EvaluationResult:
    """Result from evaluating a single test case."""
    case_name: str
    category: str
    passed: bool
    scores: Dict[str, float]
    execution_time: float
    output: Optional[ResearchResults] = None
    error: Optional[str] = None
    multi_judge_result: Optional[Any] = None


class ResearchExecutorEvaluator:
    """Evaluator for research executor agent."""

    def __init__(self, use_multi_judge: bool = False):
        """Initialize the evaluator.

        Args:
            use_multi_judge: Whether to use multi-judge consensus evaluation
        """
        self.agent = ResearchExecutorAgent()
        self.use_multi_judge = use_multi_judge

        # Initialize multi-judge evaluator if needed
        if use_multi_judge:
            adapter = ResearchExecutorMultiJudgeAdapter()
            self.multi_judge_evaluator = BaseMultiJudgeEvaluator(
                adapter=adapter,
                voting_method=VotingMethod.CONFIDENCE_WEIGHTED,
                consensus_threshold=0.7
            )

        # Map evaluator names to instances
        self.evaluator_map = {
            "FindingsRelevanceEvaluator": FindingsRelevanceEvaluator(),
            "SourceCredibilityEvaluator": SourceCredibilityEvaluator(),
            "InsightQualityEvaluator": InsightQualityEvaluator(),
            "DataGapIdentificationEvaluator": DataGapIdentificationEvaluator(),
            "ComprehensiveEvaluator": ComprehensiveEvaluator(),
            "ConfidenceCalibrationEvaluator": ConfidenceCalibrationEvaluator(),
            "EvidenceSupportEvaluator": EvidenceSupportEvaluator(),
            "CategoryCoverageEvaluator": CategoryCoverageEvaluator(),
            "CrossReferenceEvaluator": CrossReferenceEvaluator()
        }

    def load_dataset_from_yaml(self, categories: Optional[List[str]] = None) -> List[TestCase]:
        """Load test cases from YAML dataset.

        Args:
            categories: Optional list of categories to filter

        Returns:
            List of test cases
        """
        yaml_path = Path(__file__).parent / "evaluation_datasets" / "research_executor_dataset.yaml"

        if not yaml_path.exists():
            console.print(f"[yellow]Warning: YAML dataset not found at {yaml_path}[/yellow]")
            return self.get_hardcoded_test_cases(categories)

        try:
            import yaml
            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f)

            test_cases = []
            for category_name, cases in data.get('cases', {}).items():
                if categories and category_name not in categories:
                    continue

                for case_data in cases:
                    # Parse input
                    input_data = case_data.get('inputs', {})
                    test_input = ResearchExecutorInput(
                        query=input_data.get('query', ''),
                        research_brief=input_data.get('research_brief'),
                        methodology=input_data.get('methodology'),
                        domain=input_data.get('domain'),
                        complexity=input_data.get('complexity', 'medium'),
                        temporal_relevance=input_data.get('temporal_relevance')
                    )

                    # Parse expected output
                    expected_data = case_data.get('expected_output', {})
                    expected = ResearchExecutorExpectedOutput(**expected_data)

                    # Get evaluators
                    evaluators = case_data.get('evaluators', ['FindingsRelevanceEvaluator', 'ComprehensiveEvaluator'])

                    test_cases.append(TestCase(
                        name=case_data.get('name', 'unnamed'),
                        category=category_name,
                        input=test_input,
                        expected=expected,
                        evaluators=evaluators,
                        description=case_data.get('description', '')
                    ))

            return test_cases

        except Exception as e:
            console.print(f"[red]Error loading YAML dataset: {e}[/red]")
            return self.get_hardcoded_test_cases(categories)

    def get_hardcoded_test_cases(self, categories: Optional[List[str]] = None) -> List[TestCase]:
        """Get hardcoded test cases for quick evaluation."""

        all_cases = []

        # Golden standard cases
        golden_cases = [
            TestCase(
                name="technical_comparison",
                category="golden_standard",
                input=ResearchExecutorInput(
                    query="Compare React vs Vue.js for building modern web applications",
                    domain="technical",
                    complexity="medium",
                    temporal_relevance=True
                ),
                expected=ResearchExecutorExpectedOutput(
                    min_findings=4,
                    max_findings=12,
                    min_sources=3,
                    expected_categories=["technical"],
                    expected_insights_themes=["performance", "ecosystem", "learning curve"],
                    min_quality_score=0.65
                ),
                evaluators=["FindingsRelevanceEvaluator", "SourceCredibilityEvaluator", "InsightQualityEvaluator"],
                description="Test technical comparison research capabilities"
            ),
            TestCase(
                name="scientific_research",
                category="golden_standard",
                input=ResearchExecutorInput(
                    query="Recent advances in renewable energy storage technologies",
                    domain="scientific",
                    complexity="complex",
                    temporal_relevance=True
                ),
                expected=ResearchExecutorExpectedOutput(
                    min_findings=5,
                    min_sources=4,
                    expected_categories=["scientific", "technical"],
                    expected_gaps=["cost analysis", "scalability"],
                    source_credibility_threshold=0.6
                ),
                evaluators=["FindingsRelevanceEvaluator", "DataGapIdentificationEvaluator"],
                description="Test scientific research with temporal relevance"
            ),
            TestCase(
                name="business_analysis",
                category="golden_standard",
                input=ResearchExecutorInput(
                    query="Impact of AI on customer service industry",
                    domain="business",
                    complexity="medium"
                ),
                expected=ResearchExecutorExpectedOutput(
                    min_findings=4,
                    expected_categories=["business", "technical"],
                    expected_insights_themes=["automation", "efficiency", "customer satisfaction"],
                    confidence_calibration="well-calibrated"
                ),
                evaluators=["InsightQualityEvaluator", "ConfidenceCalibrationEvaluator", "CategoryCoverageEvaluator"],
                description="Test business domain research and insight generation"
            )
        ]

        # Technical cases
        technical_cases = [
            TestCase(
                name="api_design",
                category="technical",
                input=ResearchExecutorInput(
                    query="Best practices for RESTful API versioning strategies",
                    domain="technical",
                    complexity="medium"
                ),
                expected=ResearchExecutorExpectedOutput(
                    min_findings=3,
                    expected_categories=["technical"],
                    expected_insights_themes=["versioning", "backward compatibility", "deprecation"]
                ),
                evaluators=["FindingsRelevanceEvaluator", "EvidenceSupportEvaluator"],
                description="Test technical best practices research"
            ),
            TestCase(
                name="cloud_architecture",
                category="technical",
                input=ResearchExecutorInput(
                    query="Microservices vs monolithic architecture for startups",
                    domain="technical",
                    complexity="complex"
                ),
                expected=ResearchExecutorExpectedOutput(
                    min_findings=5,
                    expected_categories=["technical", "business"],
                    source_credibility_threshold=0.6
                ),
                evaluators=["FindingsRelevanceEvaluator", "CrossReferenceEvaluator"],
                description="Test architectural decision research"
            )
        ]

        # Edge cases
        edge_cases = [
            TestCase(
                name="minimal_query",
                category="edge_cases",
                input=ResearchExecutorInput(
                    query="Python",
                    complexity="simple"
                ),
                expected=ResearchExecutorExpectedOutput(
                    min_findings=2,
                    max_findings=6
                ),
                evaluators=["FindingsRelevanceEvaluator", "ComprehensiveEvaluator"],
                description="Test handling of minimal, ambiguous queries"
            ),
            TestCase(
                name="highly_specific",
                category="edge_cases",
                input=ResearchExecutorInput(
                    query="Performance impact of Python 3.11's specialized adaptive interpreter on NumPy operations",
                    domain="technical",
                    complexity="complex"
                ),
                expected=ResearchExecutorExpectedOutput(
                    min_findings=2,
                    max_findings=5,
                    expected_categories=["technical", "scientific"]
                ),
                evaluators=["FindingsRelevanceEvaluator", "SourceCredibilityEvaluator"],
                description="Test handling of highly specific technical queries"
            )
        ]

        # Performance cases
        performance_cases = [
            TestCase(
                name="simple_factual",
                category="performance",
                input=ResearchExecutorInput(
                    query="What is the population of Tokyo in 2024?",
                    complexity="simple"
                ),
                expected=ResearchExecutorExpectedOutput(
                    min_findings=1,
                    max_findings=3,
                    max_response_time=10.0
                ),
                evaluators=["FindingsRelevanceEvaluator"],
                description="Test simple factual query performance"
            )
        ]

        # Combine all cases
        all_cases.extend(golden_cases)
        all_cases.extend(technical_cases)
        all_cases.extend(edge_cases)
        all_cases.extend(performance_cases)

        # Filter by categories if specified
        if categories:
            all_cases = [case for case in all_cases if case.category in categories]

        return all_cases

    async def evaluate_case(self, test_case: TestCase) -> EvaluationResult:
        """Evaluate a single test case.

        Args:
            test_case: Test case to evaluate

        Returns:
            Evaluation result
        """
        start_time = time.time()

        try:
            # Create dependencies
            async with httpx.AsyncClient() as http_client:
                state = ResearchState(
                    request_id=f"eval-{test_case.name}",
                    user_id="test-user",
                    session_id="test-session",
                    user_query=test_case.input.query,
                    current_stage=ResearchStage.RESEARCH_EXECUTION,
                    metadata=ResearchMetadata()
                )

                # Add research brief and methodology if provided
                if test_case.input.research_brief or test_case.input.methodology:
                    state.metadata.query.transformed_query = {
                        "research_plan": {
                            "brief": test_case.input.research_brief or "",
                            "methodology": test_case.input.methodology or ""
                        }
                    }

                deps = ResearchDependencies(
                    http_client=http_client,
                    api_keys=APIKeys(
                        openai=SecretStr(openai_key) if (openai_key := os.getenv("OPENAI_API_KEY")) else None
                    ),
                    research_state=state
                )

                # Run the agent
                result = await self.agent.agent.run(test_case.input.query, deps=deps)
                output = result.output

                # Evaluate with configured evaluators
                scores = {}
                for evaluator_name in test_case.evaluators:
                    if evaluator_name in self.evaluator_map:
                        evaluator = self.evaluator_map[evaluator_name]
                        # Create mock context for evaluation
                        from types import SimpleNamespace
                        ctx = SimpleNamespace(
                            output=output,
                            expected_output=test_case.expected,
                            input=test_case.input
                        )
                        score = evaluator.evaluate(ctx)
                        scores[evaluator_name] = score

                # Run multi-judge evaluation if enabled
                multi_judge_result = None
                if self.use_multi_judge:
                    context = {
                        "domain": test_case.input.domain,
                        "complexity": test_case.input.complexity,
                        "temporal_relevance": test_case.input.temporal_relevance
                    }
                    multi_judge_result = await self.multi_judge_evaluator.evaluate(
                        test_case.input.query,
                        output,
                        context
                    )
                    scores["MultiJudgeConsensus"] = multi_judge_result.final_score

                # Determine if passed (average score > 0.7)
                avg_score = sum(scores.values()) / len(scores) if scores else 0
                passed = avg_score >= 0.7

                execution_time = time.time() - start_time

                return EvaluationResult(
                    case_name=test_case.name,
                    category=test_case.category,
                    passed=passed,
                    scores=scores,
                    execution_time=execution_time,
                    output=output,
                    multi_judge_result=multi_judge_result
                )

        except Exception as e:
            execution_time = time.time() - start_time
            return EvaluationResult(
                case_name=test_case.name,
                category=test_case.category,
                passed=False,
                scores={},
                execution_time=execution_time,
                error=str(e)
            )

    async def run_evaluation(
        self,
        categories: Optional[List[str]] = None,
        max_concurrency: int = 3
    ) -> Tuple[List[EvaluationResult], Dict[str, Any]]:
        """Run evaluation on test cases.

        Args:
            categories: Optional list of categories to test
            max_concurrency: Maximum concurrent evaluations

        Returns:
            Tuple of results list and summary statistics
        """
        # Load test cases
        test_cases = self.load_dataset_from_yaml(categories)

        if not test_cases:
            console.print("[red]No test cases found![/red]")
            return [], {}

        console.print(f"[cyan]Loaded {len(test_cases)} test cases[/cyan]")

        # Create progress bar
        results = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            task = progress.add_task(
                f"[green]Evaluating {len(test_cases)} cases...",
                total=len(test_cases)
            )

            # Run evaluations with controlled concurrency
            semaphore = asyncio.Semaphore(max_concurrency)

            async def evaluate_with_semaphore(test_case):
                async with semaphore:
                    progress.update(task, description=f"[yellow]Evaluating: {test_case.name}")
                    result = await self.evaluate_case(test_case)
                    progress.update(task, advance=1)
                    return result

            # Execute all evaluations
            results = await asyncio.gather(*[
                evaluate_with_semaphore(case) for case in test_cases
            ])

        # Calculate summary statistics
        summary = self.calculate_summary(results)

        return results, summary

    def calculate_summary(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Calculate summary statistics from results.

        Args:
            results: List of evaluation results

        Returns:
            Summary statistics
        """
        total_cases = len(results)
        passed_cases = sum(1 for r in results if r.passed)
        failed_cases = total_cases - passed_cases

        # Calculate average scores by evaluator
        evaluator_scores = {}
        for result in results:
            for evaluator, score in result.scores.items():
                if evaluator not in evaluator_scores:
                    evaluator_scores[evaluator] = []
                evaluator_scores[evaluator].append(score)

        avg_evaluator_scores = {
            evaluator: sum(scores) / len(scores)
            for evaluator, scores in evaluator_scores.items()
        }

        # Calculate category breakdown
        category_results = {}
        for result in results:
            if result.category not in category_results:
                category_results[result.category] = {"passed": 0, "failed": 0}
            if result.passed:
                category_results[result.category]["passed"] += 1
            else:
                category_results[result.category]["failed"] += 1

        # Calculate research statistics
        total_findings = 0
        total_sources = 0
        total_insights = 0
        total_gaps = 0

        for result in results:
            if result.output:
                total_findings += len(result.output.findings)
                total_sources += len(result.output.sources)
                total_insights += len(result.output.key_insights)
                total_gaps += len(result.output.data_gaps)

        avg_findings = total_findings / total_cases if total_cases > 0 else 0
        avg_sources = total_sources / total_cases if total_cases > 0 else 0
        avg_insights = total_insights / total_cases if total_cases > 0 else 0
        avg_gaps = total_gaps / total_cases if total_cases > 0 else 0

        return {
            "total_cases": total_cases,
            "passed_cases": passed_cases,
            "failed_cases": failed_cases,
            "pass_rate": passed_cases / total_cases if total_cases > 0 else 0,
            "avg_evaluator_scores": avg_evaluator_scores,
            "category_results": category_results,
            "avg_findings": avg_findings,
            "avg_sources": avg_sources,
            "avg_insights": avg_insights,
            "avg_gaps": avg_gaps,
            "total_execution_time": sum(r.execution_time for r in results)
        }

    def print_results(self, results: List[EvaluationResult], summary: Dict[str, Any]):
        """Print evaluation results in a formatted table.

        Args:
            results: List of evaluation results
            summary: Summary statistics
        """
        # Print header
        console.print("\n" + "=" * 80)
        console.print("[bold cyan]RESEARCH EXECUTOR AGENT EVALUATION RESULTS[/bold cyan]")
        console.print("=" * 80)

        # Print summary panel
        summary_text = f"""
[bold green]✓ Passed:[/bold green] {summary['passed_cases']}/{summary['total_cases']} ({summary['pass_rate']:.1%})
[bold red]✗ Failed:[/bold red] {summary['failed_cases']}/{summary['total_cases']}
[bold yellow]⏱ Total Time:[/bold yellow] {summary['total_execution_time']:.2f}s

[bold]Average Research Metrics:[/bold]
• Findings per case: {summary['avg_findings']:.1f}
• Sources per case: {summary['avg_sources']:.1f}
• Insights per case: {summary['avg_insights']:.1f}
• Gaps identified per case: {summary['avg_gaps']:.1f}
"""
        console.print(Panel(summary_text, title="Summary", border_style="cyan"))

        # Print detailed results table
        table = Table(title="Test Case Results", show_header=True, header_style="bold magenta")
        table.add_column("Case", style="cyan", width=25)
        table.add_column("Category", style="yellow", width=15)
        table.add_column("Status", justify="center", width=8)
        table.add_column("Avg Score", justify="right", width=10)
        table.add_column("Time (s)", justify="right", width=8)
        table.add_column("Key Metrics", width=30)

        for result in results:
            # Status emoji
            status = "[bold green]✓[/bold green]" if result.passed else "[bold red]✗[/bold red]"

            # Calculate average score
            avg_score = sum(result.scores.values()) / len(result.scores) if result.scores else 0

            # Format key metrics
            if result.output:
                metrics = f"F:{len(result.output.findings)} S:{len(result.output.sources)} I:{len(result.output.key_insights)}"
            elif result.error:
                metrics = f"[red]Error: {result.error[:25]}...[/red]"
            else:
                metrics = "[dim]No output[/dim]"

            table.add_row(
                result.case_name,
                result.category,
                status,
                f"{avg_score:.2f}",
                f"{result.execution_time:.2f}",
                metrics
            )

        console.print(table)

        # Print evaluator scores breakdown
        if summary['avg_evaluator_scores']:
            console.print("\n[bold]Average Scores by Evaluator:[/bold]")
            for evaluator, score in sorted(summary['avg_evaluator_scores'].items(), key=lambda x: x[1], reverse=True):
                bar_length = int(score * 30)
                bar = "█" * bar_length + "░" * (30 - bar_length)
                color = "green" if score >= 0.7 else "yellow" if score >= 0.5 else "red"
                console.print(f"  {evaluator:35} [{color}]{bar}[/{color}] {score:.2f}")

        # Print category breakdown
        if summary['category_results']:
            console.print("\n[bold]Results by Category:[/bold]")
            for category, results in summary['category_results'].items():
                total = results['passed'] + results['failed']
                pass_rate = results['passed'] / total if total > 0 else 0
                console.print(f"  {category:20} {results['passed']}/{total} passed ({pass_rate:.1%})")

    def save_results(self, results: List[EvaluationResult], summary: Dict[str, Any], filename: str = None):
        """Save evaluation results to JSON file.

        Args:
            results: List of evaluation results
            summary: Summary statistics
            filename: Optional filename (defaults to timestamped file)
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"research_executor_eval_results_{timestamp}.json"

        output_dir = Path(__file__).parent / "evaluation_results"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / filename

        # Convert results to serializable format
        serializable_results = []
        for result in results:
            serializable_result = {
                "case_name": result.case_name,
                "category": result.category,
                "passed": result.passed,
                "scores": result.scores,
                "execution_time": result.execution_time,
                "error": result.error
            }

            # Add output summary if available
            if result.output:
                serializable_result["output_summary"] = {
                    "num_findings": len(result.output.findings),
                    "num_sources": len(result.output.sources),
                    "num_insights": len(result.output.key_insights),
                    "num_gaps": len(result.output.data_gaps),
                    "quality_score": result.output.quality_score
                }

            # Add multi-judge result if available
            if result.multi_judge_result:
                serializable_result["multi_judge"] = {
                    "final_score": result.multi_judge_result.final_score,
                    "consensus_reached": result.multi_judge_result.consensus_reached,
                    "agreement_score": result.multi_judge_result.agreement_score
                }

            serializable_results.append(serializable_result)

        # Save to file
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "summary": summary,
            "results": serializable_results
        }

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)

        console.print(f"\n[green]Results saved to: {output_path}[/green]")


async def main():
    """Main entry point for the evaluation runner."""

    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Research Executor Agent Evaluation Runner")
    parser.add_argument(
        "--categories",
        nargs="+",
        help="Categories to test (e.g., golden_standard technical edge_cases)",
        default=None
    )
    parser.add_argument(
        "--multi-judge",
        action="store_true",
        help="Enable multi-judge consensus evaluation"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=3,
        help="Maximum concurrent evaluations (default: 3)"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save results to JSON file"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick test with only 3 cases"
    )

    args = parser.parse_args()

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        console.print("[red]Error: OPENAI_API_KEY environment variable not set![/red]")
        console.print("Please set your OpenAI API key: export OPENAI_API_KEY='your-key-here'")
        return

    # Override categories for quick test
    if args.quick:
        args.categories = ["golden_standard"]

    # Initialize evaluator
    evaluator = ResearchExecutorEvaluator(use_multi_judge=args.multi_judge)

    # Limit test cases for quick test
    if args.quick:
        test_cases = evaluator.load_dataset_from_yaml(args.categories)
        test_cases = test_cases[:3] if len(test_cases) > 3 else test_cases
        # Manually set the limited test cases
        evaluator.load_dataset_from_yaml = lambda cats: test_cases

    # Run evaluation
    console.print(f"[bold cyan]Starting Research Executor Agent Evaluation[/bold cyan]")
    if args.categories:
        console.print(f"Categories: {', '.join(args.categories)}")
    if args.multi_judge:
        console.print("[yellow]Multi-judge consensus evaluation enabled[/yellow]")
    if args.quick:
        console.print("[yellow]Quick mode: Testing only 3 cases[/yellow]")

    results, summary = await evaluator.run_evaluation(
        categories=args.categories,
        max_concurrency=args.concurrency
    )

    # Print results
    evaluator.print_results(results, summary)

    # Save results if requested
    if args.save:
        evaluator.save_results(results, summary)

    # Exit with appropriate code
    exit_code = 0 if summary['pass_rate'] >= 0.7 else 1
    console.print(f"\n[{'green' if exit_code == 0 else 'red'}]Evaluation {'passed' if exit_code == 0 else 'failed'}![/]")
    return exit_code


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
