"""Error Taxonomy for TamaBench V1.

Strictly separates Schema Errors (Stage 1 syntax/structure parsing)
from Environment Errors (Stage 2 domain preconditions & resources).
"""

from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field


class ErrorCategory(str, Enum):
    SCHEMA = "SCHEMA"
    ENVIRONMENT = "ENVIRONMENT"


class ErrorType(str, Enum):
    # Schema Errors (Stage 1)
    INVALID_JSON = "INVALID_JSON"
    INVALID_SCHEMA = "INVALID_SCHEMA"
    UNKNOWN_ACTION = "UNKNOWN_ACTION"
    MISSING_ARGUMENT = "MISSING_ARGUMENT"
    EXTRA_ARGUMENT = "EXTRA_ARGUMENT"
    WRONG_TYPE = "WRONG_TYPE"
    OUT_OF_RANGE = "OUT_OF_RANGE"

    # Environment / Precondition Errors (Stage 2)
    PRECONDITION_FAILED = "PRECONDITION_FAILED"
    INSUFFICIENT_RESOURCE = "INSUFFICIENT_RESOURCE"
    ACTION_UNAVAILABLE = "ACTION_UNAVAILABLE"
    INVALID_STATE = "INVALID_STATE"


class BenchmarkError(BaseModel):
    category: ErrorCategory
    error_type: ErrorType
    message: str
    details: Optional[dict[str, Any]] = Field(default=None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category.value,
            "error_type": self.error_type.value,
            "message": self.message,
            "details": self.details or {},
        }
