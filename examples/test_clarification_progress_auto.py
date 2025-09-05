"""Non-interactive test script for the clarification progress indicator.

This script demonstrates the progress indicator without requiring user input.
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from src.interfaces.clarification_progress import ClarificationProgressIndicator


async def simulate_clarification_analysis(query: str, duration: float = 6.0):
    """Simulate a clarification analysis process.
    
    Args:
        query: The research query to analyze
        duration: How long to simulate the analysis
    """
    # Simulate some async work
    await asyncio.sleep(duration)
    
    # Return mock result
    return {
        "needs_clarification": True,
        "questions": [
            "What is your target audience for this research?",
            "What specific aspects are you most interested in?",
            "What time period should the research cover?",
        ],
        "confidence": 0.85,
    }


async def main():
    """Main test function."""
    console = Console()
    
    # Single test query
    query = "What is machine learning and its applications in healthcare?"
    
    console.print("\n[bold cyan]Clarification Progress Indicator Demo[/bold cyan]\n")
    console.print("Demonstrating the new progress indicator that provides")
    console.print("rich visual feedback during clarification analysis.\n")
    console.print(f"Query: [italic]{query}[/italic]\n")
    
    # Create and run progress indicator
    progress = ClarificationProgressIndicator(console)
    progress.start(query)
    
    try:
        # Simulate the analysis with progress
        result = await progress.run_with_callback(
            simulate_clarification_analysis(query, duration=6.0)
        )
        
        # Show results
        console.print("\n[bold green]Analysis Complete![/bold green]\n")
        console.print(f"• Needs clarification: {result['needs_clarification']}")
        console.print(f"• Confidence level: {result['confidence']:.0%}")
        
        if result["needs_clarification"] and result["questions"]:
            console.print("\n[bold]Generated Clarification Questions:[/bold]")
            for j, question in enumerate(result["questions"], 1):
                console.print(f"  {j}. {question}")
                
    except Exception as e:
        progress.stop()
        console.print(f"\n[red]Error: {e}[/red]")
    
    console.print("\n[dim]Demo complete. The progress indicator shows:[/dim]")
    console.print("• Real-time analysis phases")
    console.print("• Current activity descriptions")
    console.print("• Progress bars with time tracking")
    console.print("• Smooth animations and transitions\n")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nDemo cancelled by user.")
        sys.exit(0)