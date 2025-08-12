# CLI Dual-Mode Implementation

## Overview

The Deep Research CLI now supports two execution modes:
1. **Direct Mode** (default): Executes research workflow in-process
2. **HTTP Mode**: Connects to FastAPI server via Server-Sent Events (SSE)

## Installation

### Basic Installation (Direct Mode Only)
```bash
uv sync
```

### Full Installation (With HTTP Mode Support)
```bash
uv add --optional cli
# or
pip install -e ".[cli]"
```

This installs the `httpx-sse` dependency required for HTTP mode.

## Usage

### Direct Mode (Default)
```bash
# Execute research directly without a server
deep-research research "What is quantum computing?"

# With verbose logging
deep-research research "What is quantum computing?" -v

# With custom API keys
deep-research research "What is quantum computing?" -k openai:sk-xxx -k tavily:tvly-xxx
```

### HTTP Mode
```bash
# Start the server first (in another terminal)
uvicorn open_deep_research_with_pydantic_ai.api.main:app

# Execute research via HTTP/SSE
deep-research research "What is quantum computing?" --mode http

# With custom server URL
deep-research research "What is quantum computing?" --mode http --server-url http://api.example.com:8000

# Interactive mode with HTTP
deep-research interactive --mode http
```

## Architecture

### Direct Mode Flow
```
CLI → Import Workflow → Execute In-Process → Event Bus → Progress Display
```

### HTTP Mode Flow
```
CLI → HTTP POST → FastAPI Server → Background Task → SSE Stream → Progress Display
```

## Key Components

### 1. HTTPResearchClient (`cli.py`)
- Manages HTTP connections to the FastAPI server
- Converts SSE events to CLI event handlers
- Implements retry logic for connection failures

### 2. Dual-Mode run_research Function
```python
async def run_research(
    query: str,
    api_keys: dict[str, str] | None = None,
    mode: str = "direct",
    server_url: str = "http://localhost:8000",
) -> None
```

### 3. SSE Event Processing
- Converts JSON SSE events to Python event objects
- Maintains same progress display for both modes
- Handles connection/disconnection gracefully

## Command-Line Options

### research Command
- `--mode, -m`: Choose execution mode (`direct` or `http`)
- `--server-url, -s`: Server URL for HTTP mode
- `--api-key, -k`: API keys in format `service:key`
- `--verbose, -v`: Enable verbose logging

### interactive Command
- `--mode, -m`: Execution mode for the session
- `--server-url, -s`: Server URL if using HTTP mode

## Benefits

### Direct Mode
- No server required
- Lower latency
- Simpler deployment
- Good for development/testing

### HTTP Mode
- Client-server architecture
- Multiple clients can connect
- Server can be deployed separately
- Good for production/multi-user scenarios
- Enables remote execution

## Error Handling

### Connection Failures
- Automatic retry with exponential backoff
- Maximum 3 retry attempts
- Clear error messages if server unavailable

### Graceful Degradation
- If httpx-sse not installed, HTTP mode shows helpful error
- Direct mode always available as fallback

## Testing

Run the test script to verify installation:
```bash
python test_cli_modes.py
```

This tests:
- HTTP mode dependency availability
- HTTPResearchClient initialization
- CLI command-line options

## Implementation Details

### Dependencies
- Core: `httpx`, `click`, `rich`
- HTTP Mode: `httpx-sse>=0.4.0` (optional)

### Event Types Supported
- `StreamingUpdateEvent`: Progress updates
- `StageCompletedEvent`: Stage completion notifications
- `ErrorEvent`: Error notifications
- `ResearchCompletedEvent`: Final results

### SSE Features
- Automatic reconnection (5-second retry)
- Heartbeat/ping support
- Clean disconnection handling
- Event ID tracking

## Future Enhancements

1. **WebSocket Support**: Alternative to SSE for bidirectional communication
2. **Authentication**: Add API key/token support for HTTP mode
3. **Session Management**: Save/resume research sessions
4. **Batch Processing**: Submit multiple queries
5. **Result Caching**: Cache results on server side
6. **Progress Persistence**: Resume interrupted research

## Troubleshooting

### HTTP Mode Not Available
```
Error: HTTP mode requires additional dependencies.
Install with: uv add --optional cli
```

### Server Connection Failed
```
Connection failed, retrying 1/3...
```
Ensure server is running: `uvicorn open_deep_research_with_pydantic_ai.api.main:app`

### Port Already in Use
```
ERROR: [Errno 48] Address already in use
```
Kill existing process: `lsof -i :8000 | grep LISTEN | awk '{print $2}' | xargs kill -9`
