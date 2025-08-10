# Open Deep Research with Pydantic AI

A Python project for conducting deep research using Pydantic AI.

## Features

- Built with modern Python tooling using `uv` for package management
- Structured with Pydantic for data validation
- Integrated with Pydantic AI for AI-powered research capabilities
- Comprehensive testing setup with pytest
- Code quality tools: ruff, mypy, black
- Documentation support with MkDocs

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
cd open_deep_research_with_pydantic_ai

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
open_deep_research_with_pydantic_ai/
├── src/
│   └── open_deep_research_with_pydantic_ai/
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

## Usage

```bash
# Run the CLI
uv run open-deep-research --help
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## License

[Add your license here]
