"""Tests for utility functions."""

import time

from duckmem.utils import compute_checksum, generate_uid, timestamp_ms, truncate_text


class TestGenerateUid:
    """Tests for UID generation."""

    def test_uid_length(self):
        """Test that UIDs have correct length."""
        uid = generate_uid()
        assert len(uid) == 22

    def test_uid_uniqueness(self):
        """Test that UIDs are unique."""
        uids = {generate_uid() for _ in range(1000)}
        assert len(uids) == 1000

    def test_uid_is_string(self):
        """Test that UID is a string."""
        uid = generate_uid()
        assert isinstance(uid, str)


class TestTimestampMs:
    """Tests for timestamp generation."""

    def test_timestamp_is_int(self):
        """Test that timestamp is an integer."""
        ts = timestamp_ms()
        assert isinstance(ts, int)

    def test_timestamp_is_milliseconds(self):
        """Test that timestamp is in milliseconds."""
        ts = timestamp_ms()
        # Should be around current time in ms (13+ digits)
        assert ts > 1_000_000_000_000

    def test_timestamp_increases(self):
        """Test that timestamps increase."""
        ts1 = timestamp_ms()
        time.sleep(0.01)  # 10ms
        ts2 = timestamp_ms()
        assert ts2 >= ts1


class TestComputeChecksum:
    """Tests for checksum computation."""

    def test_checksum_format(self):
        """Test that checksum is a hex string."""
        checksum = compute_checksum("test")
        assert len(checksum) == 64  # SHA-256 hex
        assert all(c in "0123456789abcdef" for c in checksum)

    def test_checksum_deterministic(self):
        """Test that same input gives same checksum."""
        text = "Hello, World!"
        cs1 = compute_checksum(text)
        cs2 = compute_checksum(text)
        assert cs1 == cs2

    def test_checksum_different_for_different_input(self):
        """Test that different inputs give different checksums."""
        cs1 = compute_checksum("text1")
        cs2 = compute_checksum("text2")
        assert cs1 != cs2

    def test_known_checksum(self):
        """Test against known SHA-256 value."""
        checksum = compute_checksum("hello")
        expected = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        assert checksum == expected


class TestTruncateText:
    """Tests for text truncation."""

    def test_short_text_unchanged(self):
        """Test that short text is not modified."""
        text = "Short"
        result = truncate_text(text, max_length=10)
        assert result == "Short"

    def test_long_text_truncated(self):
        """Test that long text is truncated."""
        text = "Hello, World!"
        result = truncate_text(text, max_length=10)
        assert len(result) == 10
        assert result.endswith("...")

    def test_custom_suffix(self):
        """Test custom truncation suffix."""
        text = "Hello, World!"
        result = truncate_text(text, max_length=10, suffix="…")
        assert result.endswith("…")

    def test_exact_length(self):
        """Test text at exact max length."""
        text = "Exact"
        result = truncate_text(text, max_length=5)
        assert result == "Exact"
