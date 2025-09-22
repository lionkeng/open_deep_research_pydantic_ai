"""CLI utility helpers (pure functions)."""

from urllib.parse import urlparse


def validate_server_url(url: str) -> str:
    """Validate and normalize server URL (http/https).

    Args:
        url: Input URL (with or without scheme)

    Returns:
        Normalized URL string
    """
    # If URL doesn't start with http:// or https://, add http://
    if not url.startswith(("http://", "https://")):
        url = f"http://{url}"

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Invalid URL scheme: {parsed.scheme}. Only http/https are supported.")
    if not parsed.netloc:
        raise ValueError("Invalid URL: missing host")
    return url
