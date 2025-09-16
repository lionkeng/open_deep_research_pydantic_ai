"""
Regression Testing and Performance Tracking System for Agents

This module provides comprehensive regression testing and performance tracking
capabilities to monitor agent performance over time and detect regressions.
"""

import json
import asyncio
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pydantic import BaseModel
import sqlite3
import sys
import os
import time

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

import httpx
from pydantic import SecretStr
from agents.clarification import ClarificationAgent
from agents.base import ResearchDependencies
from models.core import ResearchState, ResearchStage
from models.metadata import ResearchMetadata
from models.api_models import APIKeys

# Import evaluation datasets
from tests.evals.run_clarification_eval import MultiQuestionClarificationEvaluator


@dataclass
class PerformanceMetrics:
    """Core performance metrics for regression tracking."""

    # Accuracy Metrics
    overall_accuracy: float
    precision: float
    recall: float
    f1_score: float

    # Performance Metrics
    avg_response_time: float
    median_response_time: float
    p95_response_time: float
    max_response_time: float

    # Quality Metrics
    avg_confidence_score: float

    # Resource Usage
    peak_memory_usage_mb: float
    avg_cpu_usage_percent: float

    # Evaluation Metadata
    total_test_cases: int
    failed_test_cases: int
    timestamp: datetime
    git_commit: Optional[str]
    model_version: Optional[str]


@dataclass
class RegressionAlert:
    """Alert for detected regression."""

    metric_name: str
    current_value: float
    baseline_value: float
    change_percent: float
    severity: str  # "critical", "warning", "info"
    threshold_type: str  # "absolute", "relative"
    timestamp: datetime


class PerformanceDatabase:
    """SQLite database for storing performance metrics and tracking regressions."""

    def __init__(self, db_path: str = "tests/evals/performance_history.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    def _init_database(self):
        """Initialize the performance tracking database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create performance_metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                git_commit TEXT,
                model_version TEXT,
                overall_accuracy REAL NOT NULL,
                precision_score REAL NOT NULL,
                recall_score REAL NOT NULL,
                f1_score REAL NOT NULL,
                avg_response_time REAL NOT NULL,
                median_response_time REAL NOT NULL,
                p95_response_time REAL NOT NULL,
                max_response_time REAL NOT NULL,
                avg_confidence_score REAL NOT NULL,
                peak_memory_usage_mb REAL NOT NULL,
                avg_cpu_usage_percent REAL NOT NULL,
                total_test_cases INTEGER NOT NULL,
                failed_test_cases INTEGER NOT NULL
            )
        """)

        # Create regression_alerts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS regression_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                current_value REAL NOT NULL,
                baseline_value REAL NOT NULL,
                change_percent REAL NOT NULL,
                severity TEXT NOT NULL,
                threshold_type TEXT NOT NULL,
                acknowledged BOOLEAN DEFAULT FALSE
            )
        """)

        conn.commit()
        conn.close()

    def save_metrics(self, metrics: PerformanceMetrics) -> int:
        """Save performance metrics to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO performance_metrics (
                timestamp, git_commit, model_version, overall_accuracy, precision_score,
                recall_score, f1_score, avg_response_time, median_response_time,
                p95_response_time, max_response_time, avg_confidence_score,
                peak_memory_usage_mb, avg_cpu_usage_percent,
                total_test_cases, failed_test_cases
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            metrics.timestamp.isoformat(),
            metrics.git_commit,
            metrics.model_version,
            metrics.overall_accuracy,
            metrics.precision,
            metrics.recall,
            metrics.f1_score,
            metrics.avg_response_time,
            metrics.median_response_time,
            metrics.p95_response_time,
            metrics.max_response_time,
            metrics.avg_confidence_score,
            metrics.peak_memory_usage_mb,
            metrics.avg_cpu_usage_percent,
            metrics.total_test_cases,
            metrics.failed_test_cases
        ))

        run_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return run_id

    def get_baseline_metrics(self, days_back: int = 30) -> Optional[PerformanceMetrics]:
        """Get baseline metrics from recent successful runs."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM performance_metrics
            WHERE timestamp > datetime('now', '-{} days')
            ORDER BY overall_accuracy DESC, avg_response_time ASC
            LIMIT 10
        """.format(days_back))

        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return None

        # Calculate median baseline from top performing runs
        metrics_list = []
        for row in rows:
            metrics_list.append({
                'overall_accuracy': row[4],
                'precision': row[5],
                'recall': row[6],
                'f1_score': row[7],
                'avg_response_time': row[8],
                'median_response_time': row[9],
                'p95_response_time': row[10],
                'max_response_time': row[11],
                'avg_confidence_score': row[12],
                'peak_memory_usage_mb': row[13],
                'avg_cpu_usage_percent': row[14],
                'total_test_cases': row[15],
                'failed_test_cases': row[16]
            })

        # Calculate median values for baseline
        baseline = PerformanceMetrics(
            overall_accuracy=statistics.median([m['overall_accuracy'] for m in metrics_list]),
            precision=statistics.median([m['precision'] for m in metrics_list]),
            recall=statistics.median([m['recall'] for m in metrics_list]),
            f1_score=statistics.median([m['f1_score'] for m in metrics_list]),
            avg_response_time=statistics.median([m['avg_response_time'] for m in metrics_list]),
            median_response_time=statistics.median([m['median_response_time'] for m in metrics_list]),
            p95_response_time=statistics.median([m['p95_response_time'] for m in metrics_list]),
            max_response_time=statistics.median([m['max_response_time'] for m in metrics_list]),
            avg_confidence_score=statistics.median([m['avg_confidence_score'] for m in metrics_list]),
            peak_memory_usage_mb=statistics.median([m['peak_memory_usage_mb'] for m in metrics_list]),
            avg_cpu_usage_percent=statistics.median([m['avg_cpu_usage_percent'] for m in metrics_list]),
            total_test_cases=int(statistics.median([m['total_test_cases'] for m in metrics_list])),
            failed_test_cases=int(statistics.median([m['failed_test_cases'] for m in metrics_list])),
            timestamp=datetime.now(timezone.utc),
            git_commit=None,
            model_version=None
        )

        return baseline

    def save_alert(self, alert: RegressionAlert):
        """Save regression alert to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO regression_alerts (
                timestamp, metric_name, current_value, baseline_value,
                change_percent, severity, threshold_type
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            alert.timestamp.isoformat(),
            alert.metric_name,
            alert.current_value,
            alert.baseline_value,
            alert.change_percent,
            alert.severity,
            alert.threshold_type
        ))

        conn.commit()
        conn.close()


class RegressionDetector:
    """Detect regressions by comparing current metrics against baselines."""

    def __init__(self, db: PerformanceDatabase):
        self.db = db

        # Define regression thresholds
        self.thresholds = {
            # Accuracy metrics (lower is worse)
            "overall_accuracy": {"critical": -0.05, "warning": -0.02},
            "precision": {"critical": -0.05, "warning": -0.02},
            "recall": {"critical": -0.05, "warning": -0.02},
            "f1_score": {"critical": -0.05, "warning": -0.02},

            # Performance metrics (higher is worse)
            "avg_response_time": {"critical": 0.5, "warning": 0.2},
            "p95_response_time": {"critical": 1.0, "warning": 0.3},
            "max_response_time": {"critical": 2.0, "warning": 0.5},

            # Quality metrics (lower is worse)
            "avg_confidence_score": {"critical": -0.1, "warning": -0.05},

            # Resource usage (higher is worse)
            "peak_memory_usage_mb": {"critical": 0.3, "warning": 0.15},
            "avg_cpu_usage_percent": {"critical": 0.2, "warning": 0.1}
        }

    def detect_regressions(self, current: PerformanceMetrics, baseline: PerformanceMetrics) -> List[RegressionAlert]:
        """Detect regressions by comparing current metrics against baseline."""
        alerts = []

        for metric_name, thresholds in self.thresholds.items():
            if not hasattr(current, metric_name) or not hasattr(baseline, metric_name):
                continue

            current_value = getattr(current, metric_name)
            baseline_value = getattr(baseline, metric_name)

            if baseline_value == 0:
                continue  # Skip division by zero

            # Calculate change
            if metric_name in ["avg_response_time", "p95_response_time", "max_response_time",
                              "peak_memory_usage_mb", "avg_cpu_usage_percent"]:
                # For these metrics, higher is worse
                change = (current_value - baseline_value) / baseline_value
                is_regression = change > 0
            else:
                # For accuracy metrics, lower is worse
                change = (current_value - baseline_value) / baseline_value
                is_regression = change < 0

            if not is_regression:
                continue

            change_abs = abs(change)

            # Determine severity
            severity = "info"
            if change_abs >= abs(thresholds["critical"]):
                severity = "critical"
            elif change_abs >= abs(thresholds["warning"]):
                severity = "warning"
            else:
                continue  # Below threshold, no alert needed

            alert = RegressionAlert(
                metric_name=metric_name,
                current_value=current_value,
                baseline_value=baseline_value,
                change_percent=change * 100,
                severity=severity,
                threshold_type="relative",
                timestamp=datetime.now(timezone.utc)
            )

            alerts.append(alert)
            self.db.save_alert(alert)

        return alerts


class PerformanceTracker:
    """Main performance tracking and regression detection system."""

    def __init__(self, db_path: str = "tests/evals/performance_history.db"):
        self.db = PerformanceDatabase(db_path)
        self.detector = RegressionDetector(self.db)

    async def run_performance_evaluation(self,
                                       git_commit: Optional[str] = None,
                                       model_version: Optional[str] = None) -> Tuple[PerformanceMetrics, List[RegressionAlert]]:
        """Run complete performance evaluation and regression detection."""

        # Use the MultiQuestionClarificationEvaluator to run evaluation
        evaluator = MultiQuestionClarificationEvaluator()

        # Load dataset from YAML
        yaml_path = Path(__file__).parent / "evaluation_datasets" / "clarification_dataset.yaml"
        dataset = evaluator.load_dataset_from_yaml(yaml_path)

        # Track detailed results
        results = []
        response_times = []
        confidence_scores = []

        # Resource monitoring
        import psutil
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        cpu_samples = []

        start_time = time.time()

        # Initialize agent
        agent = ClarificationAgent()

        async with httpx.AsyncClient(timeout=30.0) as http_client:
            # Run evaluation on each test case
            for case in dataset.cases:
                case_start_time = time.time()

                try:
                    # Create dependencies for this run
                    state = ResearchState(
                        request_id=f"regression-{abs(hash(case.name))}",
                        user_query=case.inputs.query,
                        current_stage=ResearchStage.CLARIFICATION,
                        metadata=ResearchMetadata()
                    )
                    deps = ResearchDependencies(
                        http_client=http_client,
                        api_keys=APIKeys(
                            openai=SecretStr(openai_key) if (openai_key := os.getenv("OPENAI_API_KEY")) else None,
                            anthropic=SecretStr(anthropic_key) if (anthropic_key := os.getenv("ANTHROPIC_API_KEY")) else None
                        ),
                        research_state=state
                    )

                    # Run agent
                    result = await agent.agent.run(case.inputs.query, deps=deps)
                    response_time = time.time() - case_start_time
                    response_times.append(response_time)

                    # Extract actual output
                    actual_output = result.output if hasattr(result, 'output') else result

                    # Calculate accuracy
                    expected = case.expected_output.needs_clarification if case.expected_output else True
                    actual = actual_output.needs_clarification if hasattr(actual_output, 'needs_clarification') else False
                    is_correct = expected == actual

                    # Track confidence (use a default for now)
                    confidence_scores.append(0.8)  # Placeholder

                    # Sample CPU usage
                    cpu_samples.append(process.cpu_percent())

                    results.append({
                        'case_name': case.name,
                        'expected': expected,
                        'actual': actual,
                        'correct': is_correct,
                        'response_time': response_time
                    })

                except Exception as e:
                    results.append({
                        'case_name': case.name,
                        'expected': case.expected_output.needs_clarification if case.expected_output else True,
                        'actual': None,
                        'correct': False,
                        'response_time': time.time() - case_start_time,
                        'error': str(e)
                    })

        # Calculate final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        peak_memory = max(initial_memory, final_memory)

        # Calculate metrics
        correct_predictions = sum(1 for r in results if r['correct'])
        total_predictions = len(results)
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0

        # Calculate precision, recall, F1
        true_positives = sum(1 for r in results if r['expected'] and r['actual'] and r['correct'])
        false_positives = sum(1 for r in results if not r['expected'] and r['actual'])
        false_negatives = sum(1 for r in results if r['expected'] and not r['actual'])

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        # Create performance metrics
        metrics = PerformanceMetrics(
            overall_accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            avg_response_time=statistics.mean(response_times) if response_times else 0.0,
            median_response_time=statistics.median(response_times) if response_times else 0.0,
            p95_response_time=self._percentile(response_times, 95) if response_times else 0.0,
            max_response_time=max(response_times) if response_times else 0.0,
            avg_confidence_score=statistics.mean(confidence_scores) if confidence_scores else 0.0,
            peak_memory_usage_mb=peak_memory,
            avg_cpu_usage_percent=statistics.mean(cpu_samples) if cpu_samples else 0.0,
            total_test_cases=total_predictions,
            failed_test_cases=total_predictions - correct_predictions,
            timestamp=datetime.now(timezone.utc),
            git_commit=git_commit,
            model_version=model_version
        )

        # Save metrics
        self.db.save_metrics(metrics)

        # Detect regressions
        baseline = self.db.get_baseline_metrics()
        alerts = []

        if baseline:
            alerts = self.detector.detect_regressions(metrics, baseline)

        return metrics, alerts

    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile of a list."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = (percentile / 100.0) * (len(sorted_data) - 1)
        lower = int(index)
        upper = min(lower + 1, len(sorted_data) - 1)
        weight = index - lower
        return sorted_data[lower] * (1 - weight) + sorted_data[upper] * weight

    def generate_performance_report(self, metrics: PerformanceMetrics, alerts: List[RegressionAlert]) -> str:
        """Generate a human-readable performance report."""
        report = f"""
# Agent Performance Report
**Generated:** {metrics.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}
**Git Commit:** {metrics.git_commit or 'Unknown'}
**Model Version:** {metrics.model_version or 'Unknown'}

## Overall Performance
- **Accuracy:** {metrics.overall_accuracy:.3f} ({metrics.overall_accuracy*100:.1f}%)
- **Precision:** {metrics.precision:.3f}
- **Recall:** {metrics.recall:.3f}
- **F1 Score:** {metrics.f1_score:.3f}

## Response Time Performance
- **Average:** {metrics.avg_response_time:.3f}s
- **Median:** {metrics.median_response_time:.3f}s
- **95th Percentile:** {metrics.p95_response_time:.3f}s
- **Maximum:** {metrics.max_response_time:.3f}s

## Resource Usage
- **Peak Memory:** {metrics.peak_memory_usage_mb:.1f} MB
- **Average CPU:** {metrics.avg_cpu_usage_percent:.1f}%

## Test Results
- **Total Test Cases:** {metrics.total_test_cases}
- **Failed Cases:** {metrics.failed_test_cases}
- **Success Rate:** {((metrics.total_test_cases - metrics.failed_test_cases) / metrics.total_test_cases * 100):.1f}%
"""

        if alerts:
            report += "\n## Regression Alerts\n"
            for alert in alerts:
                emoji = "🚨" if alert.severity == "critical" else "⚠️" if alert.severity == "warning" else "ℹ️"
                report += f"- {emoji} **{alert.metric_name}**: {alert.change_percent:+.1f}% change (Current: {alert.current_value:.3f}, Baseline: {alert.baseline_value:.3f})\n"
        else:
            report += "\n## ✅ No Regressions Detected\n"

        return report


async def main():
    """Main function for running performance tracking."""
    import subprocess

    # Get current git commit
    try:
        git_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()[:8]
    except:
        git_commit = None

    # Initialize tracker
    tracker = PerformanceTracker()

    # Run evaluation
    print("🚀 Starting performance evaluation...")
    metrics, alerts = await tracker.run_performance_evaluation(
        git_commit=git_commit,
        model_version="pydantic-ai-1.0"
    )

    # Generate report
    report = tracker.generate_performance_report(metrics, alerts)
    print(report)

    # Save report to file
    report_path = Path("tests/evals/performance_report.md")
    report_path.write_text(report)
    print(f"📊 Performance report saved to {report_path}")

    # Alert on critical regressions
    critical_alerts = [a for a in alerts if a.severity == "critical"]
    if critical_alerts:
        print(f"🚨 {len(critical_alerts)} critical regressions detected!")
        for alert in critical_alerts:
            print(f"   - {alert.metric_name}: {alert.change_percent:+.1f}% change")
        return 1  # Exit with error code

    return 0


if __name__ == "__main__":
    asyncio.run(main())
