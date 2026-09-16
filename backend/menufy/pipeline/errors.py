"""Provider-agnostic errors raised by pipeline stages."""

# Base error
class LLMError(Exception):
    """A model call failed. Pipeline code catches this instead of SDK errors."""

# Network and availability
class LLMTimeoutError(LLMError):
    """The model did not respond within the configured timeout."""

class LLMUnavailableError(LLMError):
    """The model could not be reached, or the requested model is missing."""

class LLMRateLimitError(LLMError):
    """The provider rejected the call because a rate limit or quota was hit."""

# Unusable replies
class LLMEmptyResponseError(LLMError):
    """The model returned no usable text, e.g. a safety filter blocked it."""

    def __init__(self, message: str, *, reason: str | None = None) -> None:
        super().__init__(message)
        self.reason = reason
