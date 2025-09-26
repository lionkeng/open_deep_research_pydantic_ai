# Deep Research with Pydantic AI

A comprehensive AI-powered deep research system built with Pydantic-AI using an event-driven architecture. This is an AI-assisted project built with the help of claude-code and codex.

## Features

### Multi-Agent Research System

- **4-Stage Research Workflow**:
  1. Clarification – Validates and refines the research query (two‑phase flow)
  2. Query Transformation – Produces a TransformedQuery (SearchQueryBatch + ResearchPlan)
  3. Research Execution – Executes prioritized search batch and synthesizes findings
  4. Report Generation – Composes the final ResearchReport with citations

### Event-Driven Architecture

- Lock-free Event Bus implementation for async coordination
- Immutable events prevent deadlocks
- Support for streaming updates and progress tracking

### Multiple Interfaces

- **FastAPI Web API** with Server-Sent Events (SSE) support
- **Command-Line Interface** with rich formatting and streaming
- Real-time progress updates and interactive sessions

## Installation

This project uses `uv` for dependency management. First, ensure you have `uv` installed:

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then install the project:

```bash
# Clone the repository
git clone <your-repo-url>
cd open_deep_research_pydantic_ai

# Install dependencies
uv sync

# Install with development dependencies
uv sync --all-extras

# Or install specific extras
uv sync --extra dev  # Development tools
uv sync --extra docs  # Documentation tools
uv sync --extra research  # Research/data science tools
```

## Development

### Running Tests

```bash
uv run pytest
```

### Code Quality

```bash
# Lint
uv run ruff check src tests

# Format (uses ruff format)
uv run ruff format src tests

# Type checking (strict)
uv run pyright src
```

### Building Documentation

```bash
uv run mkdocs serve
```

## Project Structure

```
open_deep_research_pydantic_ai/
├── src/
│   ├── api/           # FastAPI app and HTTP endpoints (SSE streaming)
│   ├── agents/        # Agent implementations (clarification, transformation, execution, report)
│   ├── cli/           # Modular CLI (app, runner, http_client, stream, clarification_http, report_io)
│   ├── core/          # Workflow orchestrator, events, logging bootstrap
│   ├── models/        # Pydantic models for state, reports, queries
│   ├── services/      # Search orchestrator, source repository, validation, tools
│   ├── utils/         # Utilities
│   └── __init__.py
├── tests/
├── docs/
├── pyproject.toml
└── README.md
```

## Configuration

Set your API keys as environment variables:

```bash
export OPENAI_API_KEY="your-openai-key"
export TAVILY_API_KEY="your-tavily-key"  # Optional for search
export LOGFIRE_TOKEN="your-logifire-token" # For observability
```

### Optional Synthesis Enhancements (Feature Flags)

Two opt-in improvements are available for synthesis and reporting. They are disabled by default.

- Embedding-based semantic grouping
  - Enable: `ENABLE_EMBEDDING_SIMILARITY=1`
  - Optional threshold: `EMBEDDING_SIMILARITY_THRESHOLD` (default 0.55)
  - Backend: if `OPENAI_API_KEY` is set, OpenAI embeddings are used automatically in the workflow.

## Usage

### Command-Line Interface

```bash
# Run a single research query
uv run deep-research research "What are the latest advances in quantum computing?"

# Start interactive in direct more
uv run deep-research interactive --mode direct

# With custom API keys
uv run deep-research research "Your query" -k openai:sk-... -k tavily:tvly-...

# Enable verbose logging
uv run deep-research research "Your query" -v

# HTTP mode (client/server)
uv run deep-research-server
uv run deep-research research "Your query" --mode http --server-url http://localhost:8000

```

### Web API

Start the FastAPI server:

```bash
# Recommended script entry
uv run deep-research-server

# Or directly with uvicorn
uv run uvicorn api.main:app --host 0.0.0.0 --port 8000
```

API Endpoints:

- `POST /research` - Start a new research task
- `GET /research/{request_id}` - Get research status
- `GET /research/{request_id}/stream` - Stream updates via SSE
- `GET /research/{request_id}/report` - Get final report

### Python API

```python
from open_deep_research_pydantic_ai import ResearchWorkflow

# Execute research (async)
state = await ResearchWorkflow().run(
    user_query="What are the applications of AI in healthcare?",
)

if state.final_report:
    print(state.final_report.title)
    print(state.final_report.executive_summary)
```

## Architecture

### Core Components

- **Event Bus** (`src/core/events.py`) – Async event-driven coordination via immutable events
- **Workflow Orchestrator** (`src/core/workflow.py`) – Manages the 4-stage research pipeline
- **Models** (`src/models/`) – Typed Pydantic models for state, queries, and reports
- **Base Agent** (`src/agents/base.py`) – Typed agent base with DI, metrics, tools

### Agents

1. Clarification Agent – Evaluates readiness and triggers interactive clarification when needed
2. Query Transformation Agent – Emits TransformedQuery with SearchQueryBatch + ResearchPlan
3. Research Executor Agent – Executes prioritized search and synthesizes findings
4. Report Generator Agent – Composes the final report with citations

### Patterns

- Dependency Injection: Shared resources passed via `ResearchDependencies`
- Event-Driven Coordination: Lock-free async communication through the event bus
- Streaming Support: Real-time updates via SSE (HTTP) and Rich UI (CLI)

## Key Differences from Original

- **Framework**: Pydantic-AI instead of LangGraph
- **Communication**: Event Bus pattern for agent coordination
- **Streaming**: Native SSE and CLI streaming support
- **Type Safety**: Full Pydantic validation throughout

## Todo Items

- Features

  - [ ] Generate charts and relevant graphics in the report
  - [ ] Option to use other search engines, such as Brave Search
  - [ ] Configurable writing style
  - [ ] Durable execution with temporal.io

- Enhancements
  - [ ] Fix broken tests
  - [ ] Fix Pyright warnings and errors
  - [ ] Fix and improve agent evals

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## License

MIT License

## Acknowledgments

Inspired by the [Open Deep Research](https://github.com/langchain-ai/open_deep_research) project from LangChain.
