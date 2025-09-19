"""Core module initialization - loads environment variables."""

from pathlib import Path

from dotenv import load_dotenv

# Load .env file on module import to ensure API keys are available
# This runs when any core module is imported (e.g., core.workflow)
_env_loaded = False
if not _env_loaded:
    for path in [Path.cwd() / ".env"] + [p / ".env" for p in Path.cwd().parents]:
        if path.exists():
            load_dotenv(path, override=True)
            _env_loaded = True
            break
