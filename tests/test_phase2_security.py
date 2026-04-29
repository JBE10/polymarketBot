"""
Phase 2 — Security module tests.

Tests the private key retrieval logic by mocking subprocess.run to avoid
hitting the real macOS Keychain in CI. Validates both success and failure paths.
"""
from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from src.core.security import get_private_key, get_private_key_optional

# A valid-looking test private key (64 hex chars with 0x prefix)
_FAKE_KEY = "0x" + "ab" * 32   # 0xabababab...  (64 hex chars)
_BARE_KEY = "ab" * 32          # without 0x prefix


class TestGetPrivateKey:
    """get_private_key() — retrieves from macOS Keychain via subprocess."""

    @patch("src.core.security.subprocess.run")
    def test_success_with_0x_prefix(self, mock_run: MagicMock):
        """Valid key with 0x prefix should be returned as-is."""
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=_FAKE_KEY + "\n", stderr=""
        )
        key = get_private_key(service="test_svc", account="test_acct")
        assert key == _FAKE_KEY
        mock_run.assert_called_once()

    @patch("src.core.security.subprocess.run")
    def test_success_without_0x_prefix(self, mock_run: MagicMock):
        """Valid key without 0x prefix should also be accepted."""
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=_BARE_KEY + "\n", stderr=""
        )
        key = get_private_key()
        assert key == _BARE_KEY

    @patch("src.core.security.subprocess.run")
    def test_empty_key_raises(self, mock_run: MagicMock):
        """Empty result from Keychain should raise PermissionError."""
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="\n", stderr=""
        )
        with pytest.raises(PermissionError, match="clave vacía"):
            get_private_key()

    @patch("src.core.security.subprocess.run")
    def test_invalid_hex_raises(self, mock_run: MagicMock):
        """Non-hex key should raise ValueError."""
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="0xGGGG" + "0" * 60 + "\n", stderr=""
        )
        with pytest.raises(ValueError, match="clave privada Ethereum"):
            get_private_key()

    @patch("src.core.security.subprocess.run")
    def test_wrong_length_raises(self, mock_run: MagicMock):
        """Key with wrong length should raise ValueError."""
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="0x" + "aa" * 16 + "\n", stderr=""
        )
        with pytest.raises(ValueError, match="64 caracteres"):
            get_private_key()

    @patch("src.core.security.subprocess.run")
    def test_keychain_error_raises(self, mock_run: MagicMock):
        """Non-zero returncode from security CLI should raise PermissionError."""
        mock_run.side_effect = subprocess.CalledProcessError(
            returncode=44, cmd=["security"]
        )
        with pytest.raises(PermissionError, match="Secure Enclave"):
            get_private_key()

    @patch("src.core.security.subprocess.run")
    def test_timeout_raises(self, mock_run: MagicMock):
        """Keychain timeout should raise PermissionError."""
        mock_run.side_effect = subprocess.TimeoutExpired(
            cmd=["security"], timeout=10
        )
        with pytest.raises(PermissionError, match="Timeout"):
            get_private_key()


class TestGetPrivateKeyOptional:
    """get_private_key_optional() — returns None instead of raising."""

    @patch("src.core.security.subprocess.run")
    def test_returns_none_on_error(self, mock_run: MagicMock):
        mock_run.side_effect = subprocess.CalledProcessError(
            returncode=44, cmd=["security"]
        )
        result = get_private_key_optional()
        assert result is None

    @patch("src.core.security.subprocess.run")
    def test_returns_key_on_success(self, mock_run: MagicMock):
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=_FAKE_KEY + "\n", stderr=""
        )
        result = get_private_key_optional()
        assert result == _FAKE_KEY
