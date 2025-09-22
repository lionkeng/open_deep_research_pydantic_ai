"""Report display/save helpers for CLI (direct + HTTP)."""

from __future__ import annotations

from typing import Any, cast

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from models.report_generator import ResearchReport

console = Console(force_terminal=True)


def display_report_object(report: ResearchReport) -> None:
    console.print("\n")
    console.print(
        Panel(
            Markdown(f"# {report.title}\n\n{report.executive_summary}"),
            title="Research Report Summary",
        )
    )
    if report.metadata.source_summary:
        console.print("\n[bold magenta]Sources:[/bold magenta]")
        for source in report.metadata.source_summary[:10]:
            line = f"{source.get('id', '?')}: {source.get('title', '')}"
            if source.get("url"):
                line += f" ({source['url']})"
            console.print(f"  - {line}")
    for section in report.sections[:3]:
        content = section.content
        if len(content) > 500:
            content = content[:500] + "..."
        console.print(f"\n[bold cyan]{section.title}[/bold cyan]")
        console.print(content)


def display_report_dict(report_dict: dict[str, Any]) -> None:
    title = str(report_dict.get("title", "Research Report"))
    executive_summary = str(report_dict.get("executive_summary", ""))
    console.print("\n")
    console.print(
        Panel(
            Markdown(f"# {title}\n\n{executive_summary}"),
            title="Research Report Summary",
        )
    )

    sections_raw = report_dict.get("sections", [])
    sections = cast(list[Any], sections_raw) if isinstance(sections_raw, list) else []
    for section_raw in sections[:3]:
        title_val = ""
        content_val = ""
        if isinstance(section_raw, dict):
            title_val = str(section_raw.get("title") or "")
            content_val = str(section_raw.get("content") or "")
        console.print(f"\n[bold cyan]{title_val}[/bold cyan]")
        display_content = content_val[:500] + "..." if len(content_val) > 500 else content_val
        console.print(display_content)


def display_http_report(report_data: dict[str, Any]) -> None:
    display_report_dict(report_data)


def save_report_object(report: ResearchReport, filename: str) -> None:
    content: list[str] = []
    content.append(f"# {report.title}\n")
    content.append(f"*Generated: {report.metadata.created_at}*\n")
    content.append(f"\n## Executive Summary\n\n{report.executive_summary}\n")
    content.append(f"\n## Introduction\n\n{report.introduction}\n")
    for section in report.sections:
        content.append(f"\n## {section.title}\n\n{section.content}\n")
        for subsection in section.subsections:
            content.append(f"\n### {subsection.title}\n\n{subsection.content}\n")
    content.append(f"\n## Conclusions\n\n{report.conclusions}\n")
    if report.recommendations:
        content.append("\n## Recommendations\n")
        for rec in report.recommendations:
            content.append(f"- {rec}\n")
    if report.references:
        content.append("\n## Footnotes\n")
        for reference in report.references:
            content.append(f"{reference}\n")
    with open(filename, "w") as f:
        f.write("\n".join(content))


def save_http_report(report_data: dict[str, Any], filename: str) -> None:
    content: list[str] = []
    title = str(report_data.get("title", "Research Report"))
    generated_at_val = report_data.get("generated_at")
    if not generated_at_val:
        metadata = report_data.get("metadata")
        if isinstance(metadata, dict):
            generated_at_val = metadata.get("created_at")
    generated_at = str(generated_at_val) if generated_at_val is not None else "N/A"
    executive_summary = str(report_data.get("executive_summary", ""))
    introduction = str(report_data.get("introduction", ""))
    methodology = str(report_data.get("methodology", ""))
    conclusion = str(report_data.get("conclusion", ""))

    content.append(f"# {title}\n")
    content.append(f"*Generated: {generated_at}*\n")
    content.append(f"\n## Executive Summary\n\n{executive_summary}\n")
    content.append(f"\n## Introduction\n\n{introduction}\n")
    content.append(f"\n## Methodology\n\n{methodology}\n")

    sections_raw = report_data.get("sections", [])
    if isinstance(sections_raw, list):
        for section_raw in cast(list[Any], sections_raw):
            if isinstance(section_raw, dict):
                sec_title = str(section_raw.get("title", ""))
                sec_content = str(section_raw.get("content", ""))
                content.append(f"\n## {sec_title}\n\n{sec_content}\n")
                subsections_raw = section_raw.get("subsections")
                if isinstance(subsections_raw, list):
                    for subsection_raw in cast(list[Any], subsections_raw):
                        if isinstance(subsection_raw, dict):
                            sub_title = str(subsection_raw.get("title", ""))
                            sub_content = str(subsection_raw.get("content", ""))
                            content.append(f"\n### {sub_title}\n\n{sub_content}\n")

    content.append(f"\n## Conclusion\n\n{conclusion}\n")

    references_raw = report_data.get("references")
    citations_raw = report_data.get("citations") if not references_raw else None
    entries_raw = references_raw if isinstance(references_raw, list) else citations_raw
    if isinstance(entries_raw, list) and entries_raw:
        content.append("\n## References\n")
        for entry in cast(list[Any], entries_raw):
            content.append(f"- {str(entry)}\n")

    with open(filename, "w") as f:
        f.write("\n".join(content))
