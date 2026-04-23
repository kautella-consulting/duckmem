"""Utility functions for DuckMem.

Provides UID generation, timestamp handling, and checksum computation.
"""

import hashlib
import time
import uuid


def generate_uid() -> str:
    """Generate a unique identifier.

    Creates a URL-safe, collision-resistant unique identifier using UUID4.

    Returns:
        A 22-character base64-encoded UUID string.

    Example:
        >>> uid = generate_uid()
        >>> len(uid)
        22
    """
    return uuid.uuid4().hex[:22]


def timestamp_ms() -> int:
    """Get current timestamp in milliseconds.

    Returns:
        Unix timestamp in milliseconds (integer).

    Example:
        >>> ts = timestamp_ms()
        >>> isinstance(ts, int)
        True
    """
    return int(time.time() * 1000)


def compute_checksum(text: str) -> str:
    """Compute SHA-256 checksum of text.

    Args:
        text: The text to hash.

    Returns:
        Hex-encoded SHA-256 hash string (64 characters).

    Example:
        >>> compute_checksum("hello")
        '2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824'
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to a maximum length with suffix.

    Args:
        text: The text to truncate.
        max_length: Maximum length including suffix.
        suffix: String to append when truncated.

    Returns:
        Truncated text with suffix if it exceeded max_length,
        otherwise the original text.

    Example:
        >>> truncate_text("Hello, world!", max_length=10)
        'Hello, ...'
        >>> truncate_text("Short", max_length=10)
        'Short'
    """
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix
