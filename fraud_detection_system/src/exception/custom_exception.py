"""Custom exception classes for fraud detection system."""

from __future__ import annotations

import traceback


class FraudDetectionException(Exception):
    """Wraps exceptions with contextual stack information."""

    def __init__(self, message: str, original_exception: Exception | None = None):
        self.message = message
        self.original_exception = original_exception
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        if self.original_exception is None:
            return self.message

        return (
            f"{self.message} | Cause: {type(self.original_exception).__name__}: "
            f"{self.original_exception}\n{traceback.format_exc()}"
        )
