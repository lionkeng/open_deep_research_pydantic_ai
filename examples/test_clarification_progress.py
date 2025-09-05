"""Test script for the clarification progress indicator.

This script demonstrates the new progress indicator that provides
rich visual feedback during the clarification analysis phase.
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from src.interfaces.clarification_progress import ClarificationProgressIndicator


async def simulate_clarification_analysis(query: str, duration: float = 8.0):
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
    
    # Test queries to demonstrate
    test_queries = [
        "What is machine learning?",
        "Research climate change impacts on agriculture in Southeast Asia",
        "Compare Python vs JavaScript for web development",
    ]
    
    console.print("\n[bold cyan]Clarification Progress Indicator Demo[/bold cyan]\n")
    console.print("This demo shows the new progress indicator that displays")
    console.print("while the AI analyzes your query for clarification needs.\n")
    
    for i, query in enumerate(test_queries, 1):
        console.print(f"\n[bold]Test {i} of {len(test_queries)}:[/bold]")
        console.print(f"Query: [italic]{query}[/italic]\n")
        
        # Create and run progress indicator
        progress = ClarificationProgressIndicator(console)
        progress.start(query)
        
        try:
            # Simulate the analysis with progress
            result = await progress.run_with_callback(
                simulate_clarification_analysis(query, duration=6.0)
            )
            
            # Show mock results
            console.print("\n[bold green]Analysis Results:[/bold green]")
            console.print(f"Needs clarification: {result['needs_clarification']}")
            console.print(f"Confidence: {result['confidence']:.0%}")
            
            if result["needs_clarification"] and result["questions"]:
                console.print("\n[bold]Generated Questions:[/bold]")
                for j, question in enumerate(result["questions"], 1):
                    console.print(f"  {j}. {question}")
                    
        except KeyboardInterrupt:
            progress.stop()
            console.print("\n[yellow]Demo interrupted by user[/yellow]")
            break
        except Exception as e:
            progress.stop()
            console.print(f"\n[red]Error: {e}[/red]")
            
        if i < len(test_queries):
            console.print("\n[dim]Press Enter to continue to next test...[/dim]")
            input()
    
    console.print("\n[bold green]Demo complete![/bold green]\n")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nDemo cancelled by user.")
        sys.exit(0)