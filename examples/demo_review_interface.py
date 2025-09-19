"""Demo script to showcase the review interface for clarification answers.

This demonstrates the complete flow from initial questions to review and confirmation.
"""

import asyncio

from rich.console import Console

from interfaces.clarification_flow import handle_clarification_with_review
from models.clarification import (
    ClarificationAnswer,
    ClarificationQuestion,
    ClarificationRequest,
    ClarificationResponse,
)


def create_demo_request() -> ClarificationRequest:
    """Create a demo clarification request with various question types."""
    questions = [
        ClarificationQuestion(
            question="Who is your target audience for this research?",
            is_required=True,
            question_type="choice",
            choices=[
                "Academic researchers",
                "Business executives",
                "General public",
                "Students",
                "Industry professionals",
            ],
            context="Understanding your audience helps tailor the depth and style of the research.",
            order=1,
        ),
        ClarificationQuestion(
            question="What is the primary purpose of this research?",
            is_required=True,
            question_type="text",
            context="This helps focus the research on the most relevant aspects.",
            order=2,
        ),
        ClarificationQuestion(
            question="Select the geographical regions of interest",
            is_required=False,
            question_type="multi_choice",
            choices=[
                "North America",
                "Europe",
                "Asia-Pacific",
                "Latin America",
                "Middle East & Africa",
                "Global",
            ],
            order=3,
        ),
        ClarificationQuestion(
            question="What time period should the research cover?",
            is_required=True,
            question_type="choice",
            choices=[
                "Last 1 year",
                "Last 3 years",
                "Last 5 years",
                "Last 10 years",
                "All time",
                "Custom range",
            ],
            order=4,
        ),
        ClarificationQuestion(
            question="Are there specific sources you prefer?",
            is_required=False,
            question_type="multi_choice",
            choices=[
                "Academic journals",
                "Industry reports",
                "News articles",
                "Government databases",
                "Company reports",
                "Social media",
                "Books",
            ],
            context="Select all that apply. Leave blank for all sources.",
            order=5,
        ),
        ClarificationQuestion(
            question="What level of technical detail do you need?",
            is_required=True,
            question_type="choice",
            choices=[
                "High-level overview",
                "Moderate detail",
                "In-depth technical analysis",
                "Executive summary",
            ],
            order=6,
        ),
        ClarificationQuestion(
            question="Do you have any specific constraints or requirements?",
            is_required=False,
            question_type="text",
            context="E.g., word count, specific frameworks, methodologies, etc.",
            order=7,
        ),
        ClarificationQuestion(
            question="What format would you prefer for the research output?",
            is_required=False,
            question_type="choice",
            choices=[
                "Detailed report with sections",
                "Executive summary with key points",
                "Comparative analysis table",
                "Visual presentation with charts",
                "Academic paper format",
            ],
            order=8,
        ),
    ]

    return ClarificationRequest(
        questions=questions,
        context="This research will be conducted using AI-powered analysis tools.",
    )


def create_demo_response(request: ClarificationRequest) -> ClarificationResponse:
    """Create a demo response with some answers for testing the review interface."""
    # Simulate partial answers
    answers = [
        ClarificationAnswer(
            question_id=request.questions[0].id,
            answer="Academic researchers",
            skipped=False,
        ),
        ClarificationAnswer(
            question_id=request.questions[1].id,
            answer="Comparative analysis of machine learning frameworks for production deployment",
            skipped=False,
        ),
        ClarificationAnswer(
            question_id=request.questions[2].id,
            skipped=True,  # Optional question skipped
        ),
        ClarificationAnswer(
            question_id=request.questions[3].id,
            answer="Last 3 years",
            skipped=False,
        ),
        ClarificationAnswer(
            question_id=request.questions[4].id,
            answer="Academic journals, Industry reports, Company reports",
            skipped=False,
        ),
        ClarificationAnswer(
            question_id=request.questions[5].id,
            answer="In-depth technical analysis",
            skipped=False,
        ),
        ClarificationAnswer(
            question_id=request.questions[6].id,
            skipped=True,  # Optional question skipped
        ),
        ClarificationAnswer(
            question_id=request.questions[7].id,
            skipped=True,  # Optional question skipped
        ),
    ]

    return ClarificationResponse(
        request_id=request.id,
        answers=answers,
    )


async def demo_full_flow():
    """Demonstrate the full clarification flow with review."""
    console = Console()

    console.print("\n[bold cyan]DEMO: Clarification with Review Interface[/bold cyan]\n")
    console.print("This demo shows the complete clarification flow including the review step.\n")

    # Create demo request
    request = create_demo_request()
    original_query = "Compare different machine learning frameworks for production use"

    # Run the full flow
    final_response = await handle_clarification_with_review(
        request=request,
        original_query=original_query,
        console=console,
        auto_review=True,
        allow_skip_review=True,
    )

    if final_response:
        console.print("\n[green]✓ Demo completed successfully![/green]")
        console.print(f"Final response contains {len(final_response.answers)} answers.")
    else:
        console.print("\n[yellow]Demo cancelled by user.[/yellow]")


async def demo_review_only():
    """Demonstrate just the review interface with pre-populated answers."""
    console = Console()

    console.print("\n[bold cyan]DEMO: Review Interface Only[/bold cyan]\n")
    console.print("This demo shows just the review interface with pre-populated answers.\n")

    # Create demo data
    request = create_demo_request()
    response = create_demo_response(request)
    original_query = "Compare different machine learning frameworks for production use"

    # Show initial state
    console.print("[dim]Initial answers have been pre-populated for this demo.[/dim]")
    console.print("[dim]You can review and edit them using the interface.[/dim]\n")

    input("Press Enter to start the review interface...")

    # Run just the review interface
    from interfaces.review_interface import handle_review_interface

    final_response = await handle_review_interface(
        request=request,
        response=response,
        original_query=original_query,
        console=console,
    )

    if final_response:
        console.print("\n[green]✓ Review completed successfully![/green]")

        # Show what changed
        original_answers = {a.question_id: a for a in response.answers}
        final_answers = {a.question_id: a for a in final_response.answers}

        changes = []
        for qid, final_answer in final_answers.items():
            orig_answer = original_answers.get(qid)
            if orig_answer is None or orig_answer.answer != final_answer.answer:
                question = request.get_question_by_id(qid)
                if question:
                    changes.append(question.question[:50])

        if changes:
            console.print(f"\n[blue]You edited {len(changes)} answer(s):[/blue]")
            for change in changes:
                console.print(f"  • {change}")
    else:
        console.print("\n[yellow]Review cancelled by user.[/yellow]")


async def main():
    """Main demo entry point."""
    console = Console()

    console.print("\n[bold]Clarification Review Interface Demo[/bold]\n")
    console.print("Choose a demo mode:\n")
    console.print("1. Full flow (questions + review)")
    console.print("2. Review only (with pre-populated answers)")
    console.print("3. Exit\n")

    choice = input("Enter your choice (1-3): ").strip()

    if choice == "1":
        await demo_full_flow()
    elif choice == "2":
        await demo_review_only()
    else:
        console.print("[dim]Exiting demo.[/dim]")


if __name__ == "__main__":
    asyncio.run(main())
