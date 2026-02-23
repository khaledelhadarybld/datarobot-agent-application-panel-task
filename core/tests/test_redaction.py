# Copyright 2026 DataRobot, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import io
import logging
from typing import Any

import pytest

from core.telemetry.logging import (
    JsonFormatter,
    LogLevel,
    RedactingFormatter,
    TextFormatter,
    init_logging,
)


class MockIdentity:
    """Mock identity object with sensitive tokens."""

    def __init__(self) -> None:
        self.access_token = "ya29.a0AUMWg_KDI6TdrjGGy-7ZzIssji3AdseQbJ6hsy2DgFsfontL"
        self.refresh_token = "1//05I1se4f7NjcKCgYIARAAGAUSNwF-L9IrkEFC77laoDEtmb"
        self.user_id = 123

    def __repr__(self) -> str:
        return (
            f"MockIdentity(user_id={self.user_id} "
            f"access_token='{self.access_token}' "
            f"refresh_token='{self.refresh_token}')"
        )


@pytest.fixture
def mock_identity() -> MockIdentity:
    """Fixture providing a mock identity with sensitive tokens."""
    return MockIdentity()


@pytest.fixture
def logger_with_formatter(
    request: pytest.FixtureRequest,
) -> tuple[logging.Logger, io.StringIO]:
    """Fixture providing a configured logger with specified formatter."""
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)

    formatter = request.param
    redacting_formatter = RedactingFormatter(formatter)
    handler.setFormatter(redacting_formatter)

    logger = logging.getLogger(f"test_{id(stream)}")
    logger.handlers = []
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    return logger, stream


@pytest.mark.parametrize(
    "logger_with_formatter",
    [
        TextFormatter("%(message)s"),
        JsonFormatter(),
    ],
    indirect=True,
)
def test_redacts_sensitive_tokens(
    logger_with_formatter: tuple[logging.Logger, io.StringIO],
    mock_identity: MockIdentity,
) -> None:
    """Test that RedactingFormatter redacts tokens with both text and JSON formatters."""
    logger, stream = logger_with_formatter

    logger.info("found actual access token", extra={"identity": mock_identity})
    output = stream.getvalue()

    # Tokens should be redacted
    assert "ya29.a0AUMWg_KDI6TdrjGGy" not in output
    assert "1//05I1se4f7NjcKCgYIARAAGAUSNwF" not in output
    assert "[REDACTED]" in output


@pytest.mark.parametrize(
    "sensitive_data",
    [
        {
            "access_token": "access-token-value",
            "refresh_token": "refresh-token-value",
            "user_id": 123,
        },
        {
            "user": {"id": 123, "access_token": "access-token-value"},
            "list_data": [{"access_token": "refresh-token-value"}],
        },
    ],
)
def test_redacts_nested_structures(sensitive_data: dict[str, Any]) -> None:
    """Test that RedactingFormatter handles dictionaries and nested structures."""
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(RedactingFormatter(TextFormatter("%(message)s")))

    logger = logging.getLogger(f"test_{id(stream)}")
    logger.handlers = [handler]
    logger.setLevel(logging.INFO)

    logger.info("logging sensitive data", extra={"data": sensitive_data})
    output = stream.getvalue()

    assert "access-token-value" not in output
    assert "refresh-token-value" not in output
    assert "[REDACTED]" in output


def test_init_logging_applies_redaction(mock_identity: MockIdentity) -> None:
    """Test that init_logging properly applies RedactingFormatter."""
    stream = io.StringIO()
    init_logging(level=LogLevel.INFO, format_type="text", stream=stream)

    logger = logging.getLogger("test_init")
    logger.info("testing init_logging", extra={"identity": mock_identity})

    output = stream.getvalue()
    assert "ya29.a0AUMWg_KDI6TdrjGGy" not in output
    assert "[REDACTED]" in output


def test_redaction_does_not_mutate_original_object(mock_identity: MockIdentity) -> None:
    """Test that RedactingFormatter does not mutate the original object."""
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(RedactingFormatter(TextFormatter("%(message)s")))

    logger = logging.getLogger(f"test_{id(stream)}")
    logger.handlers = [handler]
    logger.setLevel(logging.INFO)

    # Store original values
    original_access_token = mock_identity.access_token
    original_refresh_token = mock_identity.refresh_token

    # Log the object
    logger.info("testing non-mutation", extra={"identity": mock_identity})

    # Verify the original object was not mutated
    assert mock_identity.access_token == original_access_token
    assert mock_identity.refresh_token == original_refresh_token

    # Verify the log output was still redacted
    output = stream.getvalue()
    assert original_access_token not in output
    assert original_refresh_token not in output
    assert "[REDACTED]" in output
