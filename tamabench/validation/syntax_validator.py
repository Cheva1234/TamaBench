"""One strict validation boundary for JSON, dictionaries, and typed actions.

Only complete JSON objects are accepted. No prose extraction, case folding,
markdown stripping, truncation repair, non-finite numbers, or duplicate keys.
"""

import json
from typing import Any, Optional, Tuple
from pydantic import ValidationError
from tamabench.schemas.actions import ActionProposal, ActionType
from tamabench.schemas.errors import BenchmarkError, ErrorCategory, ErrorType


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON number: {value}")


class SyntaxValidator:
    VALID_ACTIONS = {action.value for action in ActionType}

    @staticmethod
    def _error(kind: ErrorType, message: str) -> BenchmarkError:
        return BenchmarkError(category=ErrorCategory.SCHEMA, error_type=kind, message=message)

    @classmethod
    def validate_raw(cls, raw_output: str) -> Tuple[Optional[ActionProposal], Optional[BenchmarkError]]:
        if not isinstance(raw_output, str):
            return None, cls._error(ErrorType.WRONG_TYPE, "Raw output must be a JSON string")
        try:
            data = json.loads(raw_output, object_pairs_hook=_unique_object,
                              parse_constant=_reject_constant)
        except (ValueError, TypeError, RecursionError) as error:
            return None, cls._error(ErrorType.INVALID_JSON, f"Invalid JSON: {error}")
        return cls.validate_data(data)

    @classmethod
    def validate_data(cls, data: Any) -> Tuple[Optional[ActionProposal], Optional[BenchmarkError]]:
        # Revalidate even model_construct/model_copy results and mutable models.
        # A typed proposal must never be a shortcut around schema enforcement.
        if isinstance(data, ActionProposal):
            data = {**data.__dict__, **(data.__pydantic_extra__ or {})}
        if not isinstance(data, dict):
            return None, cls._error(ErrorType.INVALID_SCHEMA, "Action must be a JSON object")
        if "action" not in data:
            return None, cls._error(ErrorType.MISSING_ARGUMENT, "Missing required field 'action'")
        action = data["action"]
        if not isinstance(action, str):
            return None, cls._error(ErrorType.WRONG_TYPE, "Argument 'action' must be a string")
        if action not in cls.VALID_ACTIONS:
            return None, cls._error(ErrorType.UNKNOWN_ACTION, f"Unknown action '{action}'")
        required = {"work": "job_id", "buy": "item"}.get(action)
        if required and (required not in data or data[required] in (None, "")):
            return None, cls._error(ErrorType.MISSING_ARGUMENT,
                                   f"Action '{action}' requires string argument '{required}'")
        try:
            return ActionProposal.model_validate(data), None
        except ValidationError as error:
            first = error.errors()[0]
            kind = first["type"]
            if kind.endswith("_type"):
                code = ErrorType.WRONG_TYPE
            elif kind in ("greater_than", "greater_than_equal", "less_than", "less_than_equal", "finite_number"):
                code = ErrorType.OUT_OF_RANGE
            elif kind == "extra_forbidden":
                code = ErrorType.EXTRA_ARGUMENT
            elif kind == "missing":
                code = ErrorType.MISSING_ARGUMENT
            elif first["loc"] == ("hours",):
                code = ErrorType.OUT_OF_RANGE
            else:
                code = ErrorType.INVALID_SCHEMA
            return None, cls._error(code, f"Invalid action schema: {error}")
