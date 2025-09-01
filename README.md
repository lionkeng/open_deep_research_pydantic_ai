# Deep Research with Pydantic AI

A comprehensive AI-powered deep research system built with Pydantic-AI, reimplementing the concepts from the Open Deep Research project using a modern, event-driven architecture.

## Features

### Multi-Agent Research System

- **5-Stage Research Workflow**:
  1. User Clarification - Validates and refines research queries
  2. Research Brief Generation - Creates structured research plans
  3. Research Execution - Parallel information gathering with delegation
  4. Compression - Synthesizes findings into insights
  5. Report Generation - Creates comprehensive research reports

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

### Code Formatting and Linting

```bash
# Format code
uv run black src tests

# Lint code
uv run ruff check src tests

# Type checking
uv run mypy src
```

### Building Documentation

```bash
uv run mkdocs serve
```

## Project Structure

```
open_deep_research_pydantic_ai/
├── src/
│   └── open_deep_research_pydantic_ai/
│       ├── __init__.py
│       ├── cli.py           # Command-line interface
│       ├── core/            # Core functionality
│       ├── models/          # Pydantic models
│       └── utils/           # Utility functions
├── tests/                   # Test files
├── docs/                    # Documentation
├── pyproject.toml          # Project configuration
└── README.md               # This file
```

## Configuration

Set your API keys as environment variables:

```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"  # Optional
export TAVILY_API_KEY="your-tavily-key"  # Optional for search
```

## Usage

### Command-Line Interface

```bash
# Run a single research query
uv run deep-research research "What are the latest advances in quantum computing?"

# Start interactive mode
uv run deep-research interactive

# With custom API keys
uv run deep-research research "Your query" -k openai:sk-... -k tavily:tvly-...

# Enable verbose logging
uv run deep-research research "Your query" -v
```

### Web API

Start the FastAPI server:

```bash
uv run uvicorn open_deep_research_pydantic_ai.api.main:app --reload
```

API Endpoints:

- `POST /research` - Start a new research task
- `GET /research/{request_id}` - Get research status
- `GET /research/{request_id}/stream` - Stream updates via SSE
- `GET /research/{request_id}/report` - Get final report

### Python API

```python
from open_deep_research_pydantic_ai import workflow

# Execute research
state = await workflow.execute_research(
    user_query="What are the applications of AI in healthcare?",
    api_keys={"openai": "sk-..."},
)

# Access results
if state.final_report:
    print(state.final_report.title)
    print(state.final_report.executive_summary)
```

## Architecture

### Core Components

- **Event Bus** (`core/events.py`) - Async event-driven coordination
- **Workflow Orchestrator** (`core/workflow.py`) - Manages research pipeline
- **Research Models** (`models/research.py`) - Pydantic models for data validation
- **Base Agent** (`agents/base.py`) - Dependency injection and delegation support

### Agents

1. **Clarification Agent** - Validates research scope and clarity
2. **Brief Generator Agent** - Creates structured research plans
3. **Research Executor Agent** - Parallel information gathering with sub-agent delegation
4. **Compression Agent** - Synthesizes and organizes findings
5. **Report Generator Agent** - Creates comprehensive reports with citations

### Key Design Patterns

- **Agent Delegation**: Agents can delegate tasks to specialized sub-agents
- **Dependency Injection**: Shared resources passed via `ResearchDependencies`
- **Event-Driven Coordination**: Lock-free async communication
- **Streaming Support**: Real-time updates via SSE and CLI progress

## Key Differences from Original

- **Framework**: Pydantic-AI instead of LangGraph
- **Architecture**: Event-driven instead of graph-based state machines
- **Communication**: Event Bus pattern for agent coordination
- **Streaming**: Native SSE and CLI streaming support
- **Type Safety**: Full Pydantic validation throughout

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
