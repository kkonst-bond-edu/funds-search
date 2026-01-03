"""Custom exceptions for Firecrawl service."""


class FirecrawlError(Exception):
    """Base exception for Firecrawl-related errors."""
    pass


class FirecrawlAuthError(FirecrawlError):
    """Raised when Firecrawl API authentication fails."""
    pass


class FirecrawlAPIError(FirecrawlError):
    """Raised when Firecrawl API returns an error."""
    pass


class FirecrawlRateLimitError(FirecrawlError):
    """Raised when Firecrawl API rate limit is exceeded."""
    pass


class FirecrawlConnectionError(FirecrawlError):
    """Raised when connection to Firecrawl API fails."""
    pass

