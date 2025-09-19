"""
Enhanced Evaluation Runner with Comprehensive Reporting

This module provides a unified evaluation system that orchestrates all testing and
evaluation components, generates comprehensive reports, and provides dashboard
capabilities for the clarification agent.
"""

import argparse
import asyncio
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

# Testing imports
# Evaluation system imports
from tests.evals.clarification_evals import (
    ClarificationAgent,
    create_clarification_dataset,
    run_clarification_evaluation,
)
from tests.evals.domain_specific_evals import DomainEvaluationOrchestrator
from tests.evals.multi_judge_evaluation import AdvancedMultiJudgeEvaluator
from tests.evals.regression_tracker import PerformanceMetrics, PerformanceTracker, RegressionAlert


class EvaluationSuite(Enum):
    """Available evaluation suites."""

    UNIT_TESTS = "unit_tests"
    INTEGRATION_TESTS = "integration_tests"
    LLM_EVALUATIONS = "llm_evaluations"
    MULTI_JUDGE = "multi_judge"
    DOMAIN_SPECIFIC = "domain_specific"
    REGRESSION_TRACKING = "regression_tracking"
    PERFORMANCE_BENCHMARKS = "performance_benchmarks"
    ALL = "all"


class OutputFormat(Enum):
    """Available output formats."""

    CONSOLE = "console"
    MARKDOWN = "markdown"
    HTML = "html"
    JSON = "json"
    DASHBOARD = "dashboard"


@dataclass
class EvaluationConfig:
    """Configuration for evaluation runs."""

    # Suite selection
    suites: list[EvaluationSuite]

    # Output options
    output_formats: list[OutputFormat]
    output_dir: str

    # Performance options
    include_regression_tracking: bool = True
    baseline_days: int = 30

    # Reporting options
    include_trends: bool = True
    include_detailed_results: bool = True
    include_recommendations: bool = True

    # CI/CD options
    fail_on_regression: bool = True
    fail_on_accuracy_drop: float = 0.05  # 5% accuracy drop fails build

    # Git integration
    git_commit: str | None = None
    git_branch: str | None = None
    model_version: str | None = None


@dataclass
class EvaluationResults:
    """Comprehensive evaluation results."""

    # Metadata
    timestamp: datetime
    config: EvaluationConfig

    # Test results
    unit_test_results: dict[str, Any] | None = None
    integration_test_results: dict[str, Any] | None = None

    # LLM evaluation results
    llm_evaluation_results: dict[str, Any] | None = None
    multi_judge_results: dict[str, Any] | None = None
    domain_specific_results: dict[str, Any] | None = None

    # Performance results
    performance_metrics: PerformanceMetrics | None = None
    regression_alerts: list[RegressionAlert] | None = None

    # Summary
    overall_success: bool = False
    critical_issues: list[str] = None
    recommendations: list[str] = None

    def __post_init__(self):
        if self.critical_issues is None:
            self.critical_issues = []
        if self.recommendations is None:
            self.recommendations = []


class TestRunner:
    """Runner for pytest-based tests."""

    def run_unit_tests(self) -> dict[str, Any]:
        """Run unit tests."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "tests/unit/agents/test_clarification_agent_unit.py",
                "-v",
                "--tb=short",
                "--json-report",
                "--json-report-file=/tmp/unit_test_results.json",
            ],
            capture_output=True,
            text=True,
        )

        # Load JSON report if available
        json_path = Path("/tmp/unit_test_results.json")
        if json_path.exists():
            with open(json_path) as f:
                json_data = json.load(f)
        else:
            json_data = {}

        return {
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "success": result.returncode == 0,
            "json_report": json_data,
        }

    def run_integration_tests(self) -> dict[str, Any]:
        """Run integration tests."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "tests/integration/test_clarification_workflows.py",
                "-v",
                "--tb=short",
                "--json-report",
                "--json-report-file=/tmp/integration_test_results.json",
            ],
            capture_output=True,
            text=True,
        )

        # Load JSON report if available
        json_path = Path("/tmp/integration_test_results.json")
        if json_path.exists():
            with open(json_path) as f:
                json_data = json.load(f)
        else:
            json_data = {}

        return {
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "success": result.returncode == 0,
            "json_report": json_data,
        }


class ComprehensiveEvaluationRunner:
    """Main evaluation runner that orchestrates all evaluation components."""

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.test_runner = TestRunner()
        self.performance_tracker = PerformanceTracker()

        # Ensure output directory exists
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    async def run_evaluation(self) -> EvaluationResults:
        """Run comprehensive evaluation based on configuration."""

        print("🚀 Starting comprehensive evaluation...")
        print(f"   Suites: {[s.value for s in self.config.suites]}")
        print(f"   Output formats: {[f.value for f in self.config.output_formats]}")

        results = EvaluationResults(timestamp=datetime.now(UTC), config=self.config)

        # Run selected evaluation suites
        if (
            EvaluationSuite.UNIT_TESTS in self.config.suites
            or EvaluationSuite.ALL in self.config.suites
        ):
            print("📋 Running unit tests...")
            results.unit_test_results = self.test_runner.run_unit_tests()

        if (
            EvaluationSuite.INTEGRATION_TESTS in self.config.suites
            or EvaluationSuite.ALL in self.config.suites
        ):
            print("🔗 Running integration tests...")
            results.integration_test_results = self.test_runner.run_integration_tests()

        if (
            EvaluationSuite.LLM_EVALUATIONS in self.config.suites
            or EvaluationSuite.ALL in self.config.suites
        ):
            print("🤖 Running LLM evaluations...")
            results.llm_evaluation_results = await self._run_llm_evaluations()

        if (
            EvaluationSuite.MULTI_JUDGE in self.config.suites
            or EvaluationSuite.ALL in self.config.suites
        ):
            print("⚖️  Running multi-judge evaluations...")
            results.multi_judge_results = await self._run_multi_judge_evaluations()

        if (
            EvaluationSuite.DOMAIN_SPECIFIC in self.config.suites
            or EvaluationSuite.ALL in self.config.suites
        ):
            print("🎯 Running domain-specific evaluations...")
            results.domain_specific_results = await self._run_domain_specific_evaluations()

        if (
            EvaluationSuite.REGRESSION_TRACKING in self.config.suites
            or EvaluationSuite.ALL in self.config.suites
            or self.config.include_regression_tracking
        ):
            print("📊 Running regression tracking...")
            (
                performance_metrics,
                regression_alerts,
            ) = await self.performance_tracker.run_performance_evaluation(
                git_commit=self.config.git_commit, model_version=self.config.model_version
            )
            results.performance_metrics = performance_metrics
            results.regression_alerts = regression_alerts

        # Analyze results and determine overall success
        results.overall_success, results.critical_issues, results.recommendations = (
            self._analyze_results(results)
        )

        # Generate reports
        await self._generate_reports(results)

        print(f"✅ Evaluation completed. Overall success: {results.overall_success}")

        return results

    async def _run_llm_evaluations(self) -> dict[str, Any]:
        """Run LLM-based evaluations."""
        try:
            report = await run_clarification_evaluation()
            return {
                "success": True,
                "report": str(report) if report else "No report generated",
                "error": None,
            }
        except Exception as e:
            return {"success": False, "report": None, "error": str(e)}

    async def _run_multi_judge_evaluations(self) -> dict[str, Any]:
        """Run multi-judge consensus evaluations."""
        try:
            evaluator = AdvancedMultiJudgeEvaluator()
            dataset = create_clarification_dataset()
            agent = ClarificationAgent()

            results = []
            for case in dataset.cases[:5]:  # Sample evaluation for performance
                try:
                    agent_result = await agent.agent.run(case.inputs.query)
                    multi_judge_result = await evaluator.evaluate_consensus(
                        case.inputs.query,
                        agent_result.data if hasattr(agent_result, "data") else agent_result,
                    )
                    results.append(
                        {"case_name": case.name, "success": True, "result": multi_judge_result}
                    )
                except Exception as e:
                    results.append({"case_name": case.name, "success": False, "error": str(e)})

            success_rate = sum(1 for r in results if r["success"]) / len(results) if results else 0

            return {
                "success": success_rate > 0.8,
                "success_rate": success_rate,
                "total_cases": len(results),
                "results": results,
                "error": None,
            }

        except Exception as e:
            return {"success": False, "error": str(e), "results": []}

    async def _run_domain_specific_evaluations(self) -> dict[str, Any]:
        """Run domain-specific evaluations."""
        try:
            orchestrator = DomainEvaluationOrchestrator()

            # Sample queries for different domains
            test_queries = [
                ("How do I optimize my code?", "technical"),
                ("Explain quantum computing", "scientific"),
                ("Analyze the market for our product", "business"),
            ]

            results = []
            for query, expected_domain in test_queries:
                try:
                    domain_result = await orchestrator.evaluate_query_with_domain_detection(query)
                    results.append(
                        {
                            "query": query,
                            "expected_domain": expected_domain,
                            "detected_domain": domain_result.get("detected_domain"),
                            "success": True,
                            "result": domain_result,
                        }
                    )
                except Exception as e:
                    results.append(
                        {
                            "query": query,
                            "expected_domain": expected_domain,
                            "success": False,
                            "error": str(e),
                        }
                    )

            success_rate = sum(1 for r in results if r["success"]) / len(results) if results else 0

            return {
                "success": success_rate > 0.7,
                "success_rate": success_rate,
                "total_cases": len(results),
                "results": results,
                "error": None,
            }

        except Exception as e:
            return {"success": False, "error": str(e), "results": []}

    def _analyze_results(self, results: EvaluationResults) -> tuple[bool, list[str], list[str]]:
        """Analyze all results to determine overall success and generate recommendations."""

        overall_success = True
        critical_issues = []
        recommendations = []

        # Check unit tests
        if results.unit_test_results and not results.unit_test_results["success"]:
            overall_success = False
            critical_issues.append("Unit tests failed")
            recommendations.append("Fix failing unit tests before deployment")

        # Check integration tests
        if results.integration_test_results and not results.integration_test_results["success"]:
            overall_success = False
            critical_issues.append("Integration tests failed")
            recommendations.append("Investigate integration test failures")

        # Check performance metrics
        if results.performance_metrics:
            metrics = results.performance_metrics

            # Check accuracy drop
            if metrics.overall_accuracy < (1.0 - self.config.fail_on_accuracy_drop):
                if self.config.fail_on_regression:
                    overall_success = False
                critical_issues.append(f"Low accuracy: {metrics.overall_accuracy:.3f}")
                recommendations.append("Investigate causes of accuracy degradation")

            # Check response time
            if metrics.avg_response_time > 5.0:
                critical_issues.append(f"Slow response time: {metrics.avg_response_time:.2f}s")
                recommendations.append("Optimize response time performance")

            # Check domain-specific performance
            domain_accuracies = [
                metrics.technical_domain_accuracy,
                metrics.scientific_domain_accuracy,
                metrics.business_domain_accuracy,
            ]
            min_domain_accuracy = min(domain_accuracies)
            if min_domain_accuracy < 0.7:
                critical_issues.append(f"Low domain accuracy: {min_domain_accuracy:.3f}")
                recommendations.append("Improve domain-specific performance")

        # Check regression alerts
        if results.regression_alerts:
            critical_alerts = [a for a in results.regression_alerts if a.severity == "critical"]
            warning_alerts = [a for a in results.regression_alerts if a.severity == "warning"]

            if critical_alerts:
                if self.config.fail_on_regression:
                    overall_success = False
                critical_issues.extend(
                    [f"Critical regression in {a.metric_name}" for a in critical_alerts]
                )
                recommendations.append("Address critical performance regressions immediately")

            if warning_alerts:
                recommendations.extend(
                    [f"Monitor {a.metric_name} performance" for a in warning_alerts]
                )

        # Add general recommendations
        if not critical_issues:
            recommendations.append("Performance is within acceptable bounds")

        if results.performance_metrics and results.performance_metrics.avg_response_time < 2.0:
            recommendations.append("Consider response time optimization opportunities")

        return overall_success, critical_issues, recommendations

    async def _generate_reports(self, results: EvaluationResults):
        """Generate reports in all requested formats."""

        for output_format in self.config.output_formats:
            if output_format == OutputFormat.CONSOLE:
                self._print_console_report(results)
            elif output_format == OutputFormat.MARKDOWN:
                await self._generate_markdown_report(results)
            elif output_format == OutputFormat.HTML:
                await self._generate_html_report(results)
            elif output_format == OutputFormat.JSON:
                await self._generate_json_report(results)
            elif output_format == OutputFormat.DASHBOARD:
                await self._generate_dashboard(results)

    def _print_console_report(self, results: EvaluationResults):
        """Print comprehensive console report."""

        print("\n" + "=" * 80)
        print("🔍 COMPREHENSIVE EVALUATION REPORT")
        print("=" * 80)

        print(f"📅 Timestamp: {results.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"🎯 Overall Success: {'✅ PASS' if results.overall_success else '❌ FAIL'}")

        if results.critical_issues:
            print(f"🚨 Critical Issues ({len(results.critical_issues)}):")
            for issue in results.critical_issues:
                print(f"   • {issue}")

        # Performance summary
        if results.performance_metrics:
            metrics = results.performance_metrics
            print("\n📊 Performance Summary:")
            print(
                f"   • Overall Accuracy: {metrics.overall_accuracy:.3f} ({metrics.overall_accuracy * 100:.1f}%)"
            )
            print(f"   • Average Response Time: {metrics.avg_response_time:.3f}s")
            print(f"   • F1 Score: {metrics.f1_score:.3f}")
            print(
                f"   • Test Cases: {metrics.total_test_cases} ({metrics.failed_test_cases} failed)"
            )

        # Regression alerts
        if results.regression_alerts:
            critical_alerts = [a for a in results.regression_alerts if a.severity == "critical"]
            warning_alerts = [a for a in results.regression_alerts if a.severity == "warning"]

            if critical_alerts:
                print(f"\n🚨 Critical Regressions ({len(critical_alerts)}):")
                for alert in critical_alerts:
                    print(f"   • {alert.metric_name}: {alert.change_percent:+.1f}% change")

            if warning_alerts:
                print(f"\n⚠️  Warning Regressions ({len(warning_alerts)}):")
                for alert in warning_alerts:
                    print(f"   • {alert.metric_name}: {alert.change_percent:+.1f}% change")

        # Recommendations
        if results.recommendations:
            print(f"\n💡 Recommendations ({len(results.recommendations)}):")
            for rec in results.recommendations:
                print(f"   • {rec}")

        print("\n" + "=" * 80)

    async def _generate_markdown_report(self, results: EvaluationResults):
        """Generate comprehensive markdown report."""

        report = f"""# Clarification Agent Evaluation Report

**Generated:** {results.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")}
**Overall Status:** {"✅ PASS" if results.overall_success else "❌ FAIL"}
**Git Commit:** {self.config.git_commit or "Unknown"}
**Model Version:** {self.config.model_version or "Unknown"}

## Executive Summary

"""

        if results.overall_success:
            report += "🎉 All evaluation criteria passed successfully. The clarification agent is performing within acceptable parameters.\n\n"
        else:
            report += f"⚠️  Evaluation identified {len(results.critical_issues)} critical issues that need immediate attention.\n\n"

        # Critical Issues
        if results.critical_issues:
            report += "## 🚨 Critical Issues\n\n"
            for issue in results.critical_issues:
                report += f"- {issue}\n"
            report += "\n"

        # Performance Metrics
        if results.performance_metrics:
            metrics = results.performance_metrics
            report += f"""## 📊 Performance Metrics

### Accuracy Metrics
- **Overall Accuracy:** {metrics.overall_accuracy:.3f} ({metrics.overall_accuracy * 100:.1f}%)
- **Precision:** {metrics.precision:.3f}
- **Recall:** {metrics.recall:.3f}
- **F1 Score:** {metrics.f1_score:.3f}

### Response Time Performance
- **Average:** {metrics.avg_response_time:.3f}s
- **Median:** {metrics.median_response_time:.3f}s
- **95th Percentile:** {metrics.p95_response_time:.3f}s
- **Maximum:** {metrics.max_response_time:.3f}s

### Domain-Specific Performance
- **Technical Domain:** {metrics.technical_domain_accuracy:.3f} ({metrics.technical_domain_accuracy * 100:.1f}%)
- **Scientific Domain:** {metrics.scientific_domain_accuracy:.3f} ({metrics.scientific_domain_accuracy * 100:.1f}%)
- **Business Domain:** {metrics.business_domain_accuracy:.3f} ({metrics.business_domain_accuracy * 100:.1f}%)

### Quality Metrics
- **Average Confidence:** {metrics.avg_confidence_score:.3f}
- **Question Relevance:** {metrics.question_relevance_score:.3f}
- **Dimension Coverage:** {metrics.dimension_coverage_score:.3f}

### Robustness
- **Edge Case Handling:** {metrics.edge_case_handling_score:.3f} ({metrics.edge_case_handling_score * 100:.1f}%)
- **Multilingual Support:** {metrics.multilingual_handling_score:.3f} ({metrics.multilingual_handling_score * 100:.1f}%)

### Resource Usage
- **Peak Memory:** {metrics.peak_memory_usage_mb:.1f} MB
- **Average CPU:** {metrics.avg_cpu_usage_percent:.1f}%

### Test Results
- **Total Test Cases:** {metrics.total_test_cases}
- **Failed Cases:** {metrics.failed_test_cases}
- **Success Rate:** {((metrics.total_test_cases - metrics.failed_test_cases) / metrics.total_test_cases * 100):.1f}%

"""

        # Regression Alerts
        if results.regression_alerts:
            critical_alerts = [a for a in results.regression_alerts if a.severity == "critical"]
            warning_alerts = [a for a in results.regression_alerts if a.severity == "warning"]

            report += "## 🔍 Regression Analysis\n\n"

            if critical_alerts:
                report += "### 🚨 Critical Regressions\n\n"
                for alert in critical_alerts:
                    report += f"- **{alert.metric_name}**: {alert.change_percent:+.1f}% change (Current: {alert.current_value:.3f}, Baseline: {alert.baseline_value:.3f})\n"
                report += "\n"

            if warning_alerts:
                report += "### ⚠️ Warning Regressions\n\n"
                for alert in warning_alerts:
                    report += f"- **{alert.metric_name}**: {alert.change_percent:+.1f}% change (Current: {alert.current_value:.3f}, Baseline: {alert.baseline_value:.3f})\n"
                report += "\n"
        else:
            report += "## ✅ No Regressions Detected\n\nAll metrics are performing within expected ranges.\n\n"

        # Test Results Summary
        if results.unit_test_results or results.integration_test_results:
            report += "## 🧪 Test Results Summary\n\n"

            if results.unit_test_results:
                status = "✅ PASS" if results.unit_test_results["success"] else "❌ FAIL"
                report += f"- **Unit Tests:** {status}\n"

            if results.integration_test_results:
                status = "✅ PASS" if results.integration_test_results["success"] else "❌ FAIL"
                report += f"- **Integration Tests:** {status}\n"

            report += "\n"

        # Recommendations
        if results.recommendations:
            report += "## 💡 Recommendations\n\n"
            for i, rec in enumerate(results.recommendations, 1):
                report += f"{i}. {rec}\n"
            report += "\n"

        # Save report
        report_path = Path(self.config.output_dir) / "evaluation_report.md"
        report_path.write_text(report)
        print(f"📋 Markdown report saved to {report_path}")

    async def _generate_json_report(self, results: EvaluationResults):
        """Generate JSON report for programmatic consumption."""

        # Convert results to serializable format
        json_data = {
            "timestamp": results.timestamp.isoformat(),
            "overall_success": results.overall_success,
            "critical_issues": results.critical_issues,
            "recommendations": results.recommendations,
            "config": {
                "suites": [s.value for s in results.config.suites],
                "output_formats": [f.value for f in results.config.output_formats],
                "git_commit": results.config.git_commit,
                "model_version": results.config.model_version,
            },
        }

        # Add performance metrics if available
        if results.performance_metrics:
            json_data["performance_metrics"] = asdict(results.performance_metrics)
            json_data["performance_metrics"]["timestamp"] = (
                results.performance_metrics.timestamp.isoformat()
            )

        # Add regression alerts if available
        if results.regression_alerts:
            json_data["regression_alerts"] = [
                {
                    "metric_name": alert.metric_name,
                    "current_value": alert.current_value,
                    "baseline_value": alert.baseline_value,
                    "change_percent": alert.change_percent,
                    "severity": alert.severity,
                    "timestamp": alert.timestamp.isoformat(),
                }
                for alert in results.regression_alerts
            ]

        # Add test results
        json_data["test_results"] = {
            "unit_tests": results.unit_test_results,
            "integration_tests": results.integration_test_results,
            "llm_evaluations": results.llm_evaluation_results,
            "multi_judge": results.multi_judge_results,
            "domain_specific": results.domain_specific_results,
        }

        # Save JSON report
        json_path = Path(self.config.output_dir) / "evaluation_report.json"
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2, default=str)

        print(f"📊 JSON report saved to {json_path}")

    async def _generate_html_report(self, results: EvaluationResults):
        """Generate HTML dashboard report."""

        # Create a simple HTML report
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Clarification Agent Evaluation Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 20px; }}
        .header {{ background: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
        .status-pass {{ color: #28a745; font-weight: bold; }}
        .status-fail {{ color: #dc3545; font-weight: bold; }}
        .metric-card {{ background: white; border: 1px solid #dee2e6; border-radius: 8px; padding: 15px; margin: 10px 0; }}
        .critical {{ background-color: #f8d7da; border-color: #f5c6cb; }}
        .warning {{ background-color: #fff3cd; border-color: #ffeaa7; }}
        .success {{ background-color: #d4edda; border-color: #c3e6cb; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #dee2e6; }}
        th {{ background-color: #f8f9fa; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔍 Clarification Agent Evaluation Report</h1>
        <p><strong>Generated:</strong> {results.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")}</p>
        <p><strong>Status:</strong> <span class="{"status-pass" if results.overall_success else "status-fail"}">{"✅ PASS" if results.overall_success else "❌ FAIL"}</span></p>
        <p><strong>Git Commit:</strong> {self.config.git_commit or "Unknown"}</p>
        <p><strong>Model Version:</strong> {self.config.model_version or "Unknown"}</p>
    </div>"""

        # Critical Issues
        if results.critical_issues:
            html_content += """
    <div class="metric-card critical">
        <h2>🚨 Critical Issues</h2>
        <ul>"""
            for issue in results.critical_issues:
                html_content += f"<li>{issue}</li>"
            html_content += "</ul></div>"

        # Performance Metrics
        if results.performance_metrics:
            metrics = results.performance_metrics
            html_content += f"""
    <div class="metric-card">
        <h2>📊 Performance Metrics</h2>
        <div class="metrics-grid">
            <div>
                <h3>Accuracy</h3>
                <p>Overall: {metrics.overall_accuracy:.3f} ({metrics.overall_accuracy * 100:.1f}%)</p>
                <p>Precision: {metrics.precision:.3f}</p>
                <p>Recall: {metrics.recall:.3f}</p>
                <p>F1 Score: {metrics.f1_score:.3f}</p>
            </div>
            <div>
                <h3>Response Time</h3>
                <p>Average: {metrics.avg_response_time:.3f}s</p>
                <p>Median: {metrics.median_response_time:.3f}s</p>
                <p>95th Percentile: {metrics.p95_response_time:.3f}s</p>
                <p>Maximum: {metrics.max_response_time:.3f}s</p>
            </div>
            <div>
                <h3>Domain Accuracy</h3>
                <p>Technical: {metrics.technical_domain_accuracy:.3f}</p>
                <p>Scientific: {metrics.scientific_domain_accuracy:.3f}</p>
                <p>Business: {metrics.business_domain_accuracy:.3f}</p>
            </div>
        </div>
    </div>"""

        # Regression Alerts
        if results.regression_alerts:
            html_content += """
    <div class="metric-card">
        <h2>🔍 Regression Alerts</h2>
        <table>
            <tr><th>Metric</th><th>Change</th><th>Current</th><th>Baseline</th><th>Severity</th></tr>"""

            for alert in results.regression_alerts:
                severity_class = alert.severity.lower()
                html_content += f"""
            <tr class="{severity_class}">
                <td>{alert.metric_name}</td>
                <td>{alert.change_percent:+.1f}%</td>
                <td>{alert.current_value:.3f}</td>
                <td>{alert.baseline_value:.3f}</td>
                <td>{alert.severity.upper()}</td>
            </tr>"""

            html_content += "</table></div>"

        # Recommendations
        if results.recommendations:
            html_content += """
    <div class="metric-card">
        <h2>💡 Recommendations</h2>
        <ol>"""
            for rec in results.recommendations:
                html_content += f"<li>{rec}</li>"
            html_content += "</ol></div>"

        html_content += """
</body>
</html>"""

        # Save HTML report
        html_path = Path(self.config.output_dir) / "evaluation_report.html"
        html_path.write_text(html_content)
        print(f"🌐 HTML report saved to {html_path}")

    async def _generate_dashboard(self, results: EvaluationResults):
        """Generate interactive dashboard (placeholder for future implementation)."""
        # This would integrate with tools like Streamlit, Dash, or custom web dashboard
        print(
            "📈 Dashboard generation not yet implemented - consider integrating with Streamlit or similar"
        )


def get_git_info() -> tuple[str | None, str | None]:
    """Get current git commit and branch."""
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True
        ).strip()
        return commit, branch
    except:
        return None, None


def create_default_config() -> EvaluationConfig:
    """Create default evaluation configuration."""
    git_commit, git_branch = get_git_info()

    return EvaluationConfig(
        suites=[EvaluationSuite.ALL],
        output_formats=[OutputFormat.CONSOLE, OutputFormat.MARKDOWN, OutputFormat.JSON],
        output_dir="tests/evals/reports",
        git_commit=git_commit,
        git_branch=git_branch,
        model_version="pydantic-ai-1.0",
    )


async def main():
    """CLI entry point for evaluation runner."""

    parser = argparse.ArgumentParser(description="Comprehensive Clarification Agent Evaluation")

    parser.add_argument(
        "--suites",
        nargs="+",
        choices=[s.value for s in EvaluationSuite],
        default=["all"],
        help="Evaluation suites to run",
    )

    parser.add_argument(
        "--output-formats",
        nargs="+",
        choices=[f.value for f in OutputFormat],
        default=["console", "markdown", "json"],
        help="Output formats to generate",
    )

    parser.add_argument(
        "--output-dir", default="tests/evals/reports", help="Output directory for reports"
    )

    parser.add_argument(
        "--fail-on-regression", action="store_true", help="Fail build on critical regressions"
    )

    parser.add_argument(
        "--model-version", default="pydantic-ai-1.0", help="Model version identifier"
    )

    args = parser.parse_args()

    # Create configuration
    git_commit, git_branch = get_git_info()

    config = EvaluationConfig(
        suites=[EvaluationSuite(s) for s in args.suites],
        output_formats=[OutputFormat(f) for f in args.output_formats],
        output_dir=args.output_dir,
        fail_on_regression=args.fail_on_regression,
        git_commit=git_commit,
        git_branch=git_branch,
        model_version=args.model_version,
    )

    # Run evaluation
    runner = ComprehensiveEvaluationRunner(config)
    results = await runner.run_evaluation()

    # Exit with appropriate code
    sys.exit(0 if results.overall_success else 1)


if __name__ == "__main__":
    asyncio.run(main())
