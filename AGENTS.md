# AI Assistant Instructions

This document provides context and instructions for AI assistants working on the Open Deep Research with Pydantic AI project.

## Project Overview

This is a Python project for conducting deep research using Pydantic AI. The project:

- Uses modern Python 3.12+ with type hints
- Leverages Pydantic for data validation and Pydantic AI for AI-powered research
- Follows strict code quality standards with ruff, pyright, and pytest

## Development Environment

### Package Management

- **Tool**: `uv` (not pip, poetry, or conda)
- **Install dependencies**: `uv sync`
- **Run commands**: Always prefix with `uv run` (e.g., `uv run pytest`)
- **Add packages**: `uv add package-name`

### Code Quality Commands

```bash
# Run tests
uv run pytest

# Format code (if needed)
uv run ruff format src tests

# Lint code
uv run ruff check src tests

# Type checking
uv run pyright src
```

## Code Standards

### Python Style

- Line length: 100 characters
- Python version: 3.12+
- Use double quotes for strings
- Follow PEP 8 with ruff's extended ruleset (see pyproject.toml)

### Type Hints

- **Strict typing**: pyright is configured in strict mode
- Always add type hints to function signatures
- Use modern Python typing features (3.12+)
- Prefer explicit types over `Any`

### Testing

- All new features must have tests
- Tests go in the `tests/` directory
- Use pytest with async support when needed
- Aim for high code coverage

## Working with Pydantic AI

- Use Pydantic models for all data structures
- Leverage Pydantic AI's async capabilities
- Follow Pydantic AI's best practices for agent creation
- Ensure proper error handling for AI operations

## Important Notes

1. **Always run quality checks**: Before committing, run:

   - `uv run ruff check src tests`
   - `uv run pyright src`
   - `uv run pytest`

2. **Dependencies**: Check pyproject.toml before adding new dependencies

3. **Documentation**: Update docstrings for all public functions and classes

4. **Async Code**: Use async/await properly when working with Pydantic AI

5. **Error Handling**: Implement proper error handling, especially for AI operations

## Common Tasks

### Adding a New Feature

1. Create the feature in the appropriate module
2. Add Pydantic models if needed
3. Write comprehensive tests
4. Run all quality checks
5. Update documentation if needed

### Debugging

- Use `uv run ipython` for interactive debugging
- Check test coverage with `uv run pytest --cov`
- View HTML coverage report in `htmlcov/index.html`
- Use Logfire for all logging including debug logging

### Environment

- Store all configurable API keys in a `.env` file

### Working with AI Research

- Implement research logic in `core/` module
- Use Pydantic models for data validation
- Handle AI responses asynchronously
- Add appropriate error handling and retries

### Frameworks and documentation

Always consult the documentation when planning.
[Pydantic AI](https://ai.pydantic.dev/)
[FastAPI](https://fastapi.tiangolo.com/)
[Pydantic Models](https://docs.pydantic.dev/latest/)
[Logfire](https://logfire.pydantic.dev/docs/)

## Git Workflow

- Create feature branches for new work
- Run all tests before committing
- Keep commits focused and well-described
- Update tests when modifying existing features
