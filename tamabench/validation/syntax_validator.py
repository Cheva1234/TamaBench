"""Stage 1: Syntax and Schema Validator for TamaBench V1.

Parses raw LLM string generation into structured ActionProposal
and returns Schema Errors (INVALID_JSON, WRONG_TYPE, MISSING_ARGUMENT, etc.).
"""

import json
from typing import Any, Tuple, Optional
from pydantic import ValidationError
from tamabench.schemas.actions import ActionProposal, ActionType, DecisionTrace, ActionPrediction
from tamabench.schemas.errors import BenchmarkError, ErrorCategory, ErrorType


class SyntaxValidator:
    VALID_ACTIONS = {a.value for a in ActionType}

    @classmethod
    def validate_raw(cls, raw_output: str) -> Tuple[Optional[ActionProposal], Optional[BenchmarkError]]:
        """Parses raw text into JSON and validates strict structural schema."""
        cleaned = raw_output.strip()

        # Handle markdown codeblock wrapping if present
        if "```json" in cleaned:
            parts = cleaned.split("```json")
            if len(parts) > 1:
                cleaned = parts[1].split("```")[0]
        elif "```" in cleaned:
            parts = cleaned.split("```")
            if len(parts) > 1:
                cleaned = parts[1]

        cleaned = cleaned.strip()
        
        # Extract JSON object substring if surrounded by prose or handle truncation
        start_idx = cleaned.find("{")
        end_idx = cleaned.rfind("}")
        if start_idx != -1:
            if end_idx > start_idx:
                cleaned = cleaned[start_idx : end_idx + 1]
            else:
                cleaned = cleaned[start_idx:]
                if cleaned.count('"') % 2 != 0:
                    cleaned += '"'
                open_braces = cleaned.count("{") - cleaned.count("}")
                if open_braces > 0:
                    cleaned += "}" * open_braces

        # 1. Parse JSON
        try:
            data = json.loads(cleaned)
        except Exception as e:
            return None, BenchmarkError(
                category=ErrorCategory.SCHEMA,
                error_type=ErrorType.INVALID_JSON,
                message=f"Failed to parse output as valid JSON: {str(e)}",
                details={"raw_output": raw_output},
            )

        if not isinstance(data, dict):
            return None, BenchmarkError(
                category=ErrorCategory.SCHEMA,
                error_type=ErrorType.INVALID_SCHEMA,
                message="Output JSON must be a dictionary object",
                details={"parsed_type": type(data).__name__},
            )

        # 2. Check Action Field
        if "action" not in data:
            return None, BenchmarkError(
                category=ErrorCategory.SCHEMA,
                error_type=ErrorType.MISSING_ARGUMENT,
                message="Missing required field 'action'",
            )

        action_name = str(data.get("action", "")).lower()
        if action_name not in cls.VALID_ACTIONS:
            return None, BenchmarkError(
                category=ErrorCategory.SCHEMA,
                error_type=ErrorType.UNKNOWN_ACTION,
                message=f"Unknown action '{action_name}'. Valid actions are: {sorted(list(cls.VALID_ACTIONS))}",
                details={"provided_action": action_name},
            )

        # 3. Action-specific parameter type and presence validation
        if action_name == "work":
            if "job_id" not in data or not data["job_id"]:
                return None, BenchmarkError(
                    category=ErrorCategory.SCHEMA,
                    error_type=ErrorType.MISSING_ARGUMENT,
                    message="Action 'work' requires string argument 'job_id'",
                )
            if not isinstance(data["job_id"], str):
                return None, BenchmarkError(
                    category=ErrorCategory.SCHEMA,
                    error_type=ErrorType.WRONG_TYPE,
                    message="Argument 'job_id' must be a string",
                )

        elif action_name == "buy":
            if "item" not in data or not data["item"]:
                return None, BenchmarkError(
                    category=ErrorCategory.SCHEMA,
                    error_type=ErrorType.MISSING_ARGUMENT,
                    message="Action 'buy' requires string argument 'item'",
                )
            if not isinstance(data["item"], str):
                return None, BenchmarkError(
                    category=ErrorCategory.SCHEMA,
                    error_type=ErrorType.WRONG_TYPE,
                    message="Argument 'item' must be a string",
                )
            if "amount" in data:
                if not isinstance(data["amount"], int):
                    return None, BenchmarkError(
                        category=ErrorCategory.SCHEMA,
                        error_type=ErrorType.WRONG_TYPE,
                        message="Argument 'amount' must be an integer",
                    )
                if data["amount"] <= 0:
                    return None, BenchmarkError(
                        category=ErrorCategory.SCHEMA,
                        error_type=ErrorType.OUT_OF_RANGE,
                        message="Argument 'amount' must be greater than 0",
                    )

        elif action_name == "wait":
            if "minutes" in data:
                if not isinstance(data["minutes"], int):
                    return None, BenchmarkError(
                        category=ErrorCategory.SCHEMA,
                        error_type=ErrorType.WRONG_TYPE,
                        message="Argument 'minutes' must be an integer",
                    )
                if data["minutes"] <= 0:
                    return None, BenchmarkError(
                        category=ErrorCategory.SCHEMA,
                        error_type=ErrorType.OUT_OF_RANGE,
                        message="Argument 'minutes' must be greater than 0",
                    )

        # 4. Construct ActionProposal
        try:
            trace_obj = None
            if "trace" in data and isinstance(data["trace"], dict):
                trace_obj = DecisionTrace(**data["trace"])

            prediction_obj = None
            if "prediction" in data and isinstance(data["prediction"], dict):
                prediction_obj = ActionPrediction(**data["prediction"])

            proposal = ActionProposal(
                action=action_name,
                job_id=data.get("job_id"),
                item=data.get("item"),
                amount=data.get("amount", 1),
                minutes=data.get("minutes", 60),
                prediction=prediction_obj,
                trace=trace_obj,
            )
            return proposal, None

        except ValidationError as ve:
            return None, BenchmarkError(
                category=ErrorCategory.SCHEMA,
                error_type=ErrorType.INVALID_SCHEMA,
                message=f"Pydantic validation failed: {str(ve)}",
            )
