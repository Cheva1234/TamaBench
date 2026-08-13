"""Deterministic Failure Attribution Engine for TamaBench V1.

Classifies run failures into primary and secondary contributing categories
(Schema, Precondition, Bad Prediction, Resource Management, Bad Planning)
without subjective LLM judges.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
from tamabench.schemas.errors import ErrorCategory


class FailureCategory(str, Enum):
    SCHEMA = "SCHEMA"
    PRECONDITION = "PRECONDITION"
    BAD_PREDICTION = "BAD_PREDICTION"
    RESOURCE_MANAGEMENT = "RESOURCE_MANAGEMENT"
    BAD_PLANNING = "BAD_PLANNING"
    OTHER = "OTHER"


@dataclass
class FailureAttribution:
    failure_primary: FailureCategory
    failure_contributors: list[FailureCategory] = field(default_factory=list)
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "failure_primary": self.failure_primary.value,
            "failure_contributors": [c.value for c in self.failure_contributors],
            "description": self.description,
        }


class FailureAnalysisEngine:
    @classmethod
    def analyze_run_failures(
        cls, decisions: list[dict[str, Any]], outcome: Optional[dict[str, Any]] = None
    ) -> FailureAttribution:
        """Determines primary and contributing failure categories for an episode run."""
        contributors: list[FailureCategory] = []

        schema_errors = 0
        precondition_errors = 0

        for d in decisions:
            cat = d.get("error_category")
            if cat == ErrorCategory.SCHEMA.value:
                schema_errors += 1
            elif cat == ErrorCategory.ENVIRONMENT.value:
                precondition_errors += 1

        if schema_errors > 0:
            contributors.append(FailureCategory.SCHEMA)
        if precondition_errors > 0:
            contributors.append(FailureCategory.PRECONDITION)

        # Determine Primary Category
        if schema_errors > 5:
            primary = FailureCategory.SCHEMA
            desc = f"Episode failed due to high schema error rate ({schema_errors} schema errors)."
        elif precondition_errors > 5:
            primary = FailureCategory.PRECONDITION
            desc = f"Episode failed due to persistent precondition violations ({precondition_errors} precondition errors)."
        elif outcome and not bool(outcome.get("survived")):
            if outcome.get("total_spending", 0) == 0 and outcome.get("final_money", 0) >= 50:
                primary = FailureCategory.RESOURCE_MANAGEMENT
                contributors.append(FailureCategory.RESOURCE_MANAGEMENT)
                desc = "Pet died due to resource management failure (agent had funds but failed to purchase supplies)."
            else:
                primary = FailureCategory.BAD_PLANNING
                contributors.append(FailureCategory.BAD_PLANNING)
                desc = "Pet died due to bad long-horizon planning and timing choice."
        else:
            primary = FailureCategory.OTHER
            desc = "Episode completed cleanly or failed due to unclassified factors."

        # Unique contributors excluding primary
        unique_contributors = [c for c in list(set(contributors)) if c != primary]

        return FailureAttribution(
            failure_primary=primary,
            failure_contributors=unique_contributors,
            description=desc,
        )
