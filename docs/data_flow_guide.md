# Data Flow Guide - Deep Research System

This guide traces the complete data flow through the Deep Research system, from user input to final output. It provides developers with specific code references to understand how data moves through the system's components.

## Table of Contents
1. [System Architecture Overview](#system-architecture-overview)
2. [Entry Points](#entry-points)
3. [Data Models](#data-models)
4. [Workflow Orchestration](#workflow-orchestration)
5. [Agent System](#agent-system)
6. [Event-Driven Architecture](#event-driven-architecture)
7. [Response Flow](#response-flow)
8. [Code Flow Example](#code-flow-example)
9. [Developer Quick Reference](#developer-quick-reference)

## System Architecture Overview

The Deep Research system operates in two modes:

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER INPUT                               │
└────────────────────┬────────────────┬────────────────────────────┘
                     │                │
              DIRECT MODE         HTTP MODE
                     │                │
                     ▼                ▼
        ┌──────────────────┐  ┌──────────────────┐
        │   CLI (cli.py)   │  │  FastAPI Server  │
        │                  │  │    (main.py)     │
        └─────────┬────────┘  └──────┬───────────┘
                  │                   │
                  └──────┬────────────┘
                         │
                         ▼
            ┌────────────────────────────┐
            │   ResearchWorkflow         │
            │   (workflow.py:30-317)     │
            └────────────┬───────────────┘
                         │
                ┌────────┴────────┐
                ▼                 ▼
        ┌──────────────┐  ┌──────────────┐
        │  5 Research  │  │  Event Bus   │
        │    Agents    │  │ (events.py)  │
        └──────┬───────┘  └──────┬───────┘
                │                 │
                └────────┬────────┘
                         │
                         ▼
            ┌────────────────────────────┐
            │    ResearchReport          │
            │    (Final Output)          │
            └────────────────────────────┘
```

### Key Components

| Component | Location | Purpose |
|-----------|----------|---------|
| CLI Interface | `src/.../cli.py` | Command-line entry point |
| FastAPI Server | `src/.../api/main.py` | HTTP API server |
| Workflow Orchestrator | `src/.../core/workflow.py` | Manages research pipeline |
| Agent System | `src/.../agents/` | Specialized AI agents |
| Event Bus | `src/.../core/events.py` | Async event coordination |
| SSE Handler | `src/.../api/sse_handler.py` | Real-time streaming |

## Entry Points

### 1. CLI Entry Points

The CLI provides three main commands:

#### Research Command (`cli.py:570-641`)
```python
@cli.command()
@click.argument("query")
@click.option("--mode", "-m", type=click.Choice(["direct", "http"]))
def research(query: str, api_key: tuple, verbose: bool, mode: str, server_url: str):
```

**Data Flow:**
1. Parse command-line arguments
2. Create `APIKeys` model (`cli.py:615-619`)
3. Call `run_research()` (`cli.py:633`)
4. Execute in either Direct or HTTP mode

#### Interactive Command (`cli.py:644-701`)
```python
@cli.command()
def interactive(mode: str, server_url: str):
```

Provides a REPL-like interface for multiple research queries.

### 2. FastAPI Entry Points

#### Start Research Endpoint (`api/main.py:69-115`)
```python
@app.post("/research", response_model=ResearchResponse)
async def start_research(request: ResearchRequest):
```

**Request Processing:**
1. Validate request with `ResearchRequest` model (`main.py:34-42`)
2. Generate unique `request_id` (`main.py:80`)
3. Initialize `ResearchState` (`main.py:83-88`)
4. Launch background task (`main.py:91-98`)

#### Stream Updates Endpoint (`api/main.py:179-192`)
```python
@app.get("/research/{request_id}/stream")
async def stream_research_updates(request_id: str, request: Request):
```

Returns Server-Sent Events (SSE) stream for real-time updates.

## Data Models

### Core State Management

#### ResearchState (`models/research.py:112-161`)
```python
class ResearchState(BaseModel):
    request_id: str  # Unique identifier
    user_query: str  # Original query
    current_stage: ResearchStage  # Pipeline stage
    clarified_query: Optional[str]
    research_brief: Optional[ResearchBrief]
    findings: List[ResearchFinding]
    compressed_findings: Optional[str]
    final_report: Optional[ResearchReport]
    error_message: Optional[str]
```

**Key Methods:**
- `advance_stage()` (`research.py:136-141`) - Progress through pipeline
- `start_research()` (`research.py:143-146`) - Initialize research
- `is_completed()` (`research.py:148-150`) - Check completion
- `set_error()` (`research.py:156-160`) - Handle errors

### Research Pipeline Stages (`models/research.py:10-20`)
```python
class ResearchStage(str, Enum):
    PENDING = "pending"
    CLARIFICATION = "clarification"
    BRIEF_GENERATION = "brief_generation"
    RESEARCH_EXECUTION = "research_execution"
    COMPRESSION = "compression"
    REPORT_GENERATION = "report_generation"
    COMPLETED = "completed"
```

### Data Structures

#### ResearchBrief (`models/research.py:30-55`)
Structured research plan with objectives, questions, and constraints.

#### ResearchFinding (`models/research.py:57-72`)
Individual research result with source attribution and confidence scoring.

#### ResearchReport (`models/research.py:91-110`)
Final output with sections, citations, and recommendations.

## Workflow Orchestration

### ResearchWorkflow Class (`core/workflow.py:30-317`)

The central orchestrator managing the 5-stage research pipeline.

#### Execute Research Method (`workflow.py:48-187`)
```python
async def execute_research(
    self,
    user_query: str,
    api_keys: Optional[APIKeys] = None,
    stream_callback: Optional[Any] = None,
    request_id: Optional[str] = None,
) -> ResearchState:
```

**Pipeline Execution:**

1. **Initialization** (`workflow.py:66-86`)
   - Create `ResearchState` with request_id
   - Initialize HTTP client
   - Create `ResearchDependencies`

2. **Stage 1: Clarification** (`workflow.py:95-109`)
   ```python
   clarification_result = await clarification_agent.clarify_query(user_query, deps)
   ```
   - Validates and refines user query
   - May return clarifying questions

3. **Stage 2: Brief Generation** (`workflow.py:112-119`)
   ```python
   research_brief = await brief_generator_agent.generate_brief(...)
   ```
   - Creates structured research plan

4. **Stage 3: Research Execution** (`workflow.py:122-129`)
   ```python
   findings = await research_executor_agent.execute_research(...)
   ```
   - Parallel information gathering

5. **Stage 4: Compression** (`workflow.py:132-144`)
   ```python
   compressed_findings = await compression_agent.compress_findings(...)
   ```
   - Synthesizes findings

6. **Stage 5: Report Generation** (`workflow.py:147-158`)
   ```python
   final_report = await report_generator_agent.generate_report(...)
   ```
   - Creates final deliverable

### Error Handling (`workflow.py:169-187`)
- Catches exceptions at any stage
- Emits error events
- Updates state with error information

## Agent System

### Base Architecture

#### BaseResearchAgent (`agents/base.py:45-99`)
```python
class BaseResearchAgent[DepsT: ResearchDependencies, OutputT: BaseModel](ABC):
    def __init__(self, name: str, model: str, output_type: type[OutputT]):
        # Initialize Pydantic AI agent
        self.agent = Agent(
            model=self.model,
            deps_type=ResearchDependencies,
            output_type=actual_output_type,
        )
```

#### ResearchDependencies (`agents/base.py:24-39`)
Dependency injection container shared across all agents:
```python
@dataclass
class ResearchDependencies:
    http_client: httpx.AsyncClient
    api_keys: APIKeys
    research_state: ResearchState
    metadata: ResearchMetadata
    usage: Usage
    stream_callback: Optional[Any]
```

### Specialized Agents

| Agent | File | Lines | Purpose |
|-------|------|-------|---------|
| ClarificationAgent | `agents/clarification.py` | 34-230 | Query validation & refinement |
| BriefGeneratorAgent | `agents/brief_generator.py` | 26-156 | Research plan creation |
| ResearchExecutorAgent | `agents/research_executor.py` | 95-201 | Information gathering |
| CompressionAgent | `agents/compression.py` | 35-165 | Finding synthesis |
| ReportGeneratorAgent | `agents/report_generator.py` | 24-180 | Final report creation |

#### Agent Registration (`agents/base.py:200-250`)
```python
coordinator = AgentCoordinator()
coordinator.register_agent(clarification_agent)
# ... register other agents
```

## Event-Driven Architecture

### Event Bus (`core/events.py:144-270`)

Provides asynchronous, lock-free coordination between components.

#### Key Components:

1. **ResearchEventBus Class** (`events.py:144-270`)
   ```python
   class ResearchEventBus:
       def subscribe(self, event_type: type[T], handler: EventHandler[T])
       async def emit(self, event: ResearchEvent)
   ```

2. **Event Types** (`events.py:39-137`)
   - `ResearchStartedEvent` - Research initiated
   - `StageCompletedEvent` - Stage finished
   - `StreamingUpdateEvent` - Real-time updates
   - `ErrorEvent` - Error occurred
   - `ResearchCompletedEvent` - Research finished

3. **Event Emission** (`events.py:176-231`)
   ```python
   async def emit(self, event: ResearchEvent):
       # Process handlers concurrently
       for handler in handlers:
           task = asyncio.create_task(self._safe_call_handler(handler, event))
   ```

### SSE Handler (`api/sse_handler.py:32-216`)

Converts internal events to Server-Sent Events for HTTP streaming.

#### Event Generator (`sse_handler.py:67-216`)
```python
async def event_generator(self, active_sessions: dict[str, ResearchState]):
    while True:
        event = await asyncio.wait_for(self.event_queue.get(), timeout=30.0)
        # Convert to SSE format based on event type
```

## Response Flow

### Direct Mode (`cli.py:432-466`)

1. **Event Subscription** (`cli.py:434-437`)
   ```python
   research_event_bus.subscribe(StreamingUpdateEvent, handler.handle_streaming_update)
   research_event_bus.subscribe(StageCompletedEvent, handler.handle_stage_completed)
   ```

2. **Progress Display** (`cli.py:87-143`)
   - `CLIStreamHandler` manages terminal output
   - Real-time progress updates with Rich library

3. **Report Display** (`cli.py:317-374`)
   ```python
   def display_report(report_data: dict | Any, is_dict: bool = False):
       # Format and display report in terminal
   ```

### HTTP Mode (`cli.py:473-519`)

1. **HTTP Client** (`cli.py:145-315`)
   ```python
   class HTTPResearchClient:
       async def start_research(query, api_keys) -> str
       async def stream_events(request_id, handler) -> None
       async def get_report(request_id) -> dict
   ```

2. **SSE Stream Processing** (`cli.py:199-269`)
   ```python
   async def stream_events(self, request_id: str, handler: CLIStreamHandler):
       async with aconnect_sse(...) as event_source:
           async for sse in event_source.aiter_sse():
               await self._process_sse_event(sse, handler)
   ```

3. **Event Conversion** (`cli.py:212-269`)
   - Parse SSE messages with Pydantic models
   - Convert to internal events
   - Update UI through handler

## Code Flow Example

Here's a complete trace of a research query through the system:

### Direct Mode Execution

```python
# 1. User runs: deep-research "What is quantum computing?"

# 2. CLI Entry (cli.py:587)
def research(query="What is quantum computing?", mode="direct"):

    # 3. Create API keys (cli.py:615-619)
    api_keys = APIKeys(openai=..., anthropic=..., tavily=...)

    # 4. Run research (cli.py:633)
    asyncio.run(run_research(query, api_keys, mode="direct"))

# 5. Execute Research (cli.py:415-466)
async def run_research(query, api_keys, mode="direct"):

    # 6. Subscribe to events (cli.py:434-437)
    research_event_bus.subscribe(StreamingUpdateEvent, handler.handle_streaming_update)

    # 7. Execute workflow (cli.py:442)
    state = await workflow.execute_research(user_query=query, api_keys=api_keys)

# 8. Workflow Execution (workflow.py:48-187)
async def execute_research(user_query, api_keys):

    # 9. Create state (workflow.py:71-74)
    research_state = ResearchState(request_id=request_id, user_query=user_query)

    # 10. Stage 1: Clarification (workflow.py:95-109)
    clarification_result = await clarification_agent.clarify_query(user_query, deps)

    # 11. Stage 2: Brief Generation (workflow.py:112-119)
    research_brief = await brief_generator_agent.generate_brief(...)

    # 12. Stage 3: Research Execution (workflow.py:122-129)
    findings = await research_executor_agent.execute_research(...)

    # 13. Stage 4: Compression (workflow.py:132-144)
    compressed_findings = await compression_agent.compress_findings(...)

    # 14. Stage 5: Report Generation (workflow.py:147-158)
    final_report = await report_generator_agent.generate_report(...)

    # 15. Return completed state
    return research_state

# 16. Display Results (cli.py:449-456)
if state.final_report:
    display_report(state.final_report)
```

### HTTP Mode Execution

```python
# 1. Start FastAPI server
uvicorn open_deep_research_with_pydantic_ai.api.main:app

# 2. User runs: deep-research "What is quantum computing?" --mode http

# 3. HTTP Client initiates (cli.py:481)
async with HTTPResearchClient(server_url) as client:

    # 4. POST to /research endpoint (cli.py:484)
    request_id = await client.start_research(query, api_keys)

# 5. Server processes request (main.py:69-115)
@app.post("/research")
async def start_research(request: ResearchRequest):

    # 6. Create background task (main.py:91-98)
    asyncio.create_task(execute_research_background(...))

# 7. Background execution (main.py:118-153)
async def execute_research_background(...):
    state = await workflow.execute_research(...)  # Same workflow as Direct mode

# 8. Client streams events (cli.py:487)
await client.stream_events_with_retry(request_id, handler)

# 9. SSE Handler streams updates (sse_handler.py:67-216)
async def event_generator(self, active_sessions):
    while True:
        event = await self.event_queue.get()
        yield ServerSentEvent(data=...)

# 10. Client processes SSE events (cli.py:212-269)
async def _process_sse_event(self, sse, handler):
    msg = parse_sse_message(sse.data)
    await handler.handle_streaming_update(event)

# 11. Fetch final report (cli.py:492)
report_data = await client.get_report(request_id)
display_http_report(report_data)
```

## Developer Quick Reference

### File Structure Map

```
src/open_deep_research_with_pydantic_ai/
├── cli.py                    # CLI interface (720 lines)
├── api/
│   ├── main.py              # FastAPI server (266 lines)
│   └── sse_handler.py       # SSE streaming (250 lines)
├── core/
│   ├── workflow.py          # Orchestration (318 lines)
│   ├── events.py            # Event bus (306 lines)
│   ├── config.py            # Configuration
│   └── sse_models.py        # SSE data models
├── agents/
│   ├── base.py              # Base agent class (250 lines)
│   ├── clarification.py     # Query clarification (231 lines)
│   ├── brief_generator.py   # Brief generation (156 lines)
│   ├── research_executor.py # Research execution (201 lines)
│   ├── compression.py       # Finding compression (165 lines)
│   └── report_generator.py  # Report generation (180 lines)
├── models/
│   ├── research.py          # Core data models (164 lines)
│   ├── api_models.py        # API models
│   └── tool_models.py       # Tool models
└── services/
    └── search.py            # Search service (200+ lines)
```

### Key Functions Index

| Function | Location | Purpose |
|----------|----------|---------|
| `execute_research()` | `workflow.py:48` | Main orchestration |
| `clarify_query()` | `clarification.py:190` | Query validation |
| `generate_brief()` | `brief_generator.py:87` | Plan creation |
| `execute_research()` | `research_executor.py:156` | Data gathering |
| `compress_findings()` | `compression.py:105` | Synthesis |
| `generate_report()` | `report_generator.py:135` | Report creation |
| `emit()` | `events.py:176` | Event emission |
| `stream_events()` | `cli.py:199` | SSE streaming |

### Common Modification Scenarios

#### Adding a New Agent
1. Create agent class in `agents/new_agent.py`
2. Extend `BaseResearchAgent` (`base.py:45`)
3. Implement `_get_default_system_prompt()`
4. Register with coordinator (`base.py:200+`)
5. Add to workflow pipeline (`workflow.py`)

#### Adding a New Event Type
1. Define event class in `events.py`
2. Extend `ResearchEvent` base class
3. Add emission helper function
4. Update SSE handler for streaming (`sse_handler.py`)

#### Modifying the Pipeline
1. Edit `execute_research()` in `workflow.py:48`
2. Update `ResearchStage` enum (`research.py:10`)
3. Modify state transitions (`research.py:136`)
4. Update event emissions

#### Adding a New API Endpoint
1. Add endpoint in `api/main.py`
2. Define request/response models
3. Update CORS settings if needed (`main.py:24`)
4. Add CLI support if applicable (`cli.py`)

### Testing Entry Points

- Unit tests: `tests/test_basic.py`
- CLI tests: `tests/test_cli_http_mode.py`
- Manual testing: `test_cli_modes.py`

### Environment Variables

Configure in `.env` file:
- `OPENAI_API_KEY` - OpenAI API key
- `ANTHROPIC_API_KEY` - Anthropic API key
- `TAVILY_API_KEY` - Tavily search API key
- `LOGFIRE_TOKEN` - Logfire logging token

### Debugging Tips

1. Enable verbose logging: `--verbose` flag
2. Check Logfire dashboard for detailed traces
3. Monitor event bus stats: `research_event_bus.get_stats()`
4. Use interactive mode for testing: `deep-research interactive`
5. Inspect SSE stream: `curl http://localhost:8000/research/{id}/stream`

## Extension Points

The system is designed for extensibility at multiple levels:

1. **Agent Level**: Add specialized research agents
2. **Tool Level**: Extend agent tools (`base.py:_register_tools`)
3. **Search Level**: Add search providers (`services/search.py`)
4. **Event Level**: Custom event handlers
5. **Output Level**: Custom report formats

This modular architecture allows developers to enhance functionality without modifying core components.
