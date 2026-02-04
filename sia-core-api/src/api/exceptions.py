"""
Custom exceptions and error handling for the API.

Author: Lorena Calvo-Bartolomé
Date: 04/02/2026 (Migrated to FastAPI)
"""

from fastapi import HTTPException, Request  # type: ignore
from fastapi.responses import JSONResponse  # type: ignore
from typing import Any, Dict, Optional


# ======================================================
# Error Codes
# ======================================================
class ErrorCodes:
    """Standardized error codes for the API."""
    # Client errors (4xx)
    BAD_REQUEST = "BAD_REQUEST"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    NOT_FOUND = "NOT_FOUND"
    CONFLICT = "CONFLICT"
    UNPROCESSABLE = "UNPROCESSABLE_ENTITY"

    # Server errors (5xx)
    INTERNAL_ERROR = "INTERNAL_ERROR"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    SOLR_ERROR = "SOLR_ERROR"
    PROCESSING_ERROR = "PROCESSING_ERROR"


# ======================================================
# Custom Exceptions
# ======================================================
class APIException(HTTPException):
    """Base API exception with standardized error response."""

    def __init__(
        self,
        status_code: int,
        error: str,
        error_code: str,
        details: Optional[Dict[str, Any]] = None
    ):
        self.error = error
        self.error_code = error_code
        self.details = details
        super().__init__(status_code=status_code, detail=error)


class NotFoundException(APIException):
    """Resource not found exception."""

    def __init__(self, resource: str, identifier: str):
        super().__init__(
            status_code=404,
            error=f"{resource} not found: {identifier}",
            error_code=ErrorCodes.NOT_FOUND,
            details={"resource": resource, "identifier": identifier}
        )


class ConflictException(APIException):
    """Resource already exists exception."""

    def __init__(self, resource: str, identifier: str):
        super().__init__(
            status_code=409,
            error=f"{resource} already exists: {identifier}",
            error_code=ErrorCodes.CONFLICT,
            details={"resource": resource, "identifier": identifier}
        )


class ValidationException(APIException):
    """Validation error exception."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            status_code=400,
            error=message,
            error_code=ErrorCodes.VALIDATION_ERROR,
            details=details
        )


class SolrException(APIException):
    """Solr-specific exception."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            status_code=500,
            error=f"Solr error: {message}",
            error_code=ErrorCodes.SOLR_ERROR,
            details=details
        )


class ProcessingException(APIException):
    """Processing/inference exception."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            status_code=500,
            error=f"Processing error: {message}",
            error_code=ErrorCodes.PROCESSING_ERROR,
            details=details
        )


class InternalException(APIException):
    """Internal server error exception."""

    def __init__(self, message: str = "An unexpected error occurred"):
        super().__init__(
            status_code=500,
            error=message,
            error_code=ErrorCodes.INTERNAL_ERROR
        )


# ======================================================
# Exception Handlers
# ======================================================
async def api_exception_handler(request: Request, exc: APIException) -> JSONResponse:
    """Handler for custom API exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.error,
            "error_code": exc.error_code,
            "details": exc.details
        }
    )


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handler for generic HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": str(exc.detail),
            "error_code": ErrorCodes.BAD_REQUEST if exc.status_code < 500 else ErrorCodes.INTERNAL_ERROR,
            "details": None
        }
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handler for unhandled exceptions."""
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "An unexpected error occurred",
            "error_code": ErrorCodes.INTERNAL_ERROR,
            "details": {"exception_type": type(exc).__name__}
        }
    )
