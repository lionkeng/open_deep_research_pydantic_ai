#!/usr/bin/env python3
"""Demo showing how regression tracking integrates with the evaluation framework."""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from regression_tracker_fixed import PerformanceTracker


async def demo_regression_tracking():
    """Demonstrate regression tracking in action."""

    print("=" * 80)
    print("REGRESSION TRACKING DEMO")
    print("=" * 80)
    print("\nThis demo shows how the regression tracker:")
    print("1. Runs performance evaluations")
    print("2. Stores metrics in a database")
    print("3. Compares against historical baselines")
    print("4. Detects and alerts on regressions")
    print("=" * 80)

    # Initialize tracker
    tracker = PerformanceTracker(db_path="demo_performance.db")

    print("\n📊 Running performance evaluation...")
    print("   (This would normally run the full test suite)")

    # Run a minimal evaluation (in practice, this runs the full suite)
    try:
        metrics, alerts = await tracker.run_performance_evaluation(
            git_commit="demo-commit",
            model_version="v1.0-demo"
        )

        print("\n✅ Evaluation complete!")
        print(f"\nMetrics Summary:")
        print(f"  - Accuracy: {metrics.overall_accuracy:.2%}")
        print(f"  - Precision: {metrics.precision:.2f}")
        print(f"  - Recall: {metrics.recall:.2f}")
        print(f"  - F1 Score: {metrics.f1_score:.2f}")
        print(f"  - Avg Response Time: {metrics.avg_response_time:.3f}s")
        print(f"  - Test Cases: {metrics.total_test_cases} total, {metrics.failed_test_cases} failed")

        if alerts:
            print(f"\n⚠️  {len(alerts)} regression alerts detected:")
            for alert in alerts:
                severity_icon = "🚨" if alert.severity == "critical" else "⚠️"
                print(f"  {severity_icon} {alert.metric_name}: {alert.change_percent:+.1f}% change")
        else:
            print("\n✅ No regressions detected!")

        # Generate full report
        report = tracker.generate_performance_report(metrics, alerts)

        # Save report
        report_path = Path("demo_performance_report.md")
        report_path.write_text(report)
        print(f"\n📄 Full report saved to: {report_path}")

    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        print("\nNote: This demo requires:")
        print("  1. API keys (OPENAI_API_KEY) to be set")
        print("  2. The evaluation dataset to be present")
        return False

    print("\n" + "=" * 80)
    print("HOW IT WORKS IN THE EVALUATION FRAMEWORK:")
    print("=" * 80)
    print("""
1. CONTINUOUS MONITORING:
   - Run on every commit in CI/CD
   - Track performance trends over time
   - Build historical baseline from best runs

2. REGRESSION DETECTION:
   - Compare current metrics against baseline
   - Alert on significant degradations
   - Configurable thresholds per metric

3. INTEGRATION POINTS:
   - Works with existing evaluation datasets
   - Can incorporate multi-judge consensus scores
   - Tracks resource usage alongside accuracy

4. USE CASES:
   - Pre-merge validation in pull requests
   - Nightly regression tests
   - Model version comparisons
   - Performance profiling
    """)

    return True


async def main():
    """Main function."""
    success = await demo_regression_tracking()

    if success:
        print("\n✅ Demo completed successfully!")
        print("\nThe regression tracker is now ready to:")
        print("  • Monitor agent performance over time")
        print("  • Detect regressions automatically")
        print("  • Generate performance reports")
        print("  • Integrate with CI/CD pipelines")
    else:
        print("\n⚠️  Demo encountered issues. See notes above.")


if __name__ == "__main__":
    asyncio.run(main())
