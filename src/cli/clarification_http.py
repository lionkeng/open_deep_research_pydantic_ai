"""HTTP-mode clarification orchestration for CLI."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import logfire
from rich.console import Console

from models.clarification import ClarificationRequest

from .http_client import HTTPResearchClient
from .stream import CLIStreamHandler


async def handle_http_clarification_flow(
    client: HTTPResearchClient,
    request_id: str,
    console: Console,
    stream_task: asyncio.Task[None],
    handler: CLIStreamHandler,
) -> None:
    """Poll server for clarification, prompt user, and submit answers.

    Minimal post-submit behavior: stop polling after successful submission.
    """
    try:
        from interfaces.cli_multi_clarification import handle_multi_clarification_cli
    except ImportError:
        return

    backoff = 1.0
    max_backoff = 5.0
    handled_request_ids: set[str] = set()

    while not stream_task.done():
        try:
            status = await client.get_clarification(request_id)
        except Exception as e:
            logfire.debug(f"Clarification poll error: {e}")
            await asyncio.sleep(backoff)
            backoff = min(backoff + 1.0, max_backoff)
            continue

        if not bool(status.get("awaiting_response")):
            await asyncio.sleep(backoff)
            backoff = min(backoff + 1.0, max_backoff)
            continue

        backoff = 1.0
        req_data: Any = status.get("clarification_request")
        original_query = str(status.get("original_query") or "")
        if not isinstance(req_data, dict):
            await asyncio.sleep(1.0)
            continue
        try:
            request_model = ClarificationRequest.model_validate(req_data)
        except Exception as e:
            logfire.warning(f"Failed to parse ClarificationRequest: {e}")
            await asyncio.sleep(1.0)
            continue

        if request_model.id in handled_request_ids:
            await asyncio.sleep(1.0)
            continue

        try:
            qs = [
                {
                    "id": q.id,
                    "order": q.order,
                    "required": q.is_required,
                    "type": q.question_type,
                    "text": (q.question[:120] if isinstance(q.question, str) else ""),
                }
                for q in request_model.questions
            ]
            logfire.info(
                "Clarification prompt received",
                request_id=request_id,
                clar_request_id=request_model.id,
                num_questions=len(request_model.questions),
                questions=qs,
            )
        except Exception:
            pass

        try:
            handler.progress_manager.stop()
        except Exception:
            pass

        try:
            response_model = await handle_multi_clarification_cli(
                request_model, original_query, console
            )
        except KeyboardInterrupt:
            console.print("\n[yellow]Clarification cancelled by user[/yellow]")
            await asyncio.sleep(1.0)
            continue
        except Exception as e:
            console.print(f"[red]Clarification UI error: {e}[/red]")
            await asyncio.sleep(1.0)
            continue

        if response_model is None:
            await asyncio.sleep(1.0)
            continue

        try:
            _ = await client.submit_clarification(request_id, response_model)
            console.print("[green]Clarification submitted. Resuming research...[/green]")
            handled_request_ids.add(request_model.id)
            logfire.info(
                "Clarification submitted", request_id=request_id, clar_request_id=request_model.id
            )
        except Exception as e:
            if isinstance(e, httpx.HTTPStatusError):
                if e.response.status_code == 400:
                    err = {}
                    try:
                        err = e.response.json()
                    except Exception:
                        pass
                    console.print(
                        f"[yellow]Validation error submitting clarification: {err}[/yellow]"
                    )
                elif e.response.status_code == 409:
                    console.print("[yellow]Clarification already processed. Continuing...[/yellow]")
                    handled_request_ids.add(request_model.id)
                else:
                    console.print(f"[red]Failed to submit clarification: {e}[/red]")
            else:
                console.print(f"[red]Failed to submit clarification: {e}[/red]")

        # Brief settle loop then exit (minimal change)
        settle_backoff = 0.5
        for _ in range(10):
            if stream_task.done():
                break
            try:
                post_status = await client.get_clarification(request_id)
                if not bool(post_status.get("awaiting_response")):
                    break
                new_req = post_status.get("clarification_request") or {}
                new_id = str(new_req.get("id") or "")
                if new_id and new_id != request_model.id:
                    break
            except Exception:
                pass
            await asyncio.sleep(settle_backoff)
            settle_backoff = min(settle_backoff + 0.5, 2.0)

        return
