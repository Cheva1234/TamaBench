"""Tests for two-stage validation engine (Schema vs Environment errors)."""

import pytest
from tamabench.env.core import TamaEnv
from tamabench.validation.syntax_validator import SyntaxValidator
from tamabench.validation.env_validator import EnvironmentValidator
from tamabench.schemas.actions import ActionProposal
from tamabench.schemas.errors import ErrorCategory, ErrorType


@pytest.mark.unit
def test_stage_1_invalid_json():
    proposal, err = SyntaxValidator.validate_raw("not a valid json string")
    assert proposal is None
    assert err is not None
    assert err.category == ErrorCategory.SCHEMA
    assert err.error_type == ErrorType.INVALID_JSON


@pytest.mark.unit
def test_stage_1_wrong_type():
    proposal, err = SyntaxValidator.validate_raw('{"action": "buy", "item": "food", "amount": "two"}')
    assert proposal is None
    assert err is not None
    assert err.category == ErrorCategory.SCHEMA
    assert err.error_type == ErrorType.WRONG_TYPE


@pytest.mark.unit
def test_stage_1_unknown_action():
    proposal, err = SyntaxValidator.validate_raw('{"action": "dance"}')
    assert proposal is None
    assert err is not None
    assert err.category == ErrorCategory.SCHEMA
    assert err.error_type == ErrorType.UNKNOWN_ACTION


@pytest.mark.unit
def test_stage_2_sleeping_precondition():
    env = TamaEnv()
    env.reset(seed=42)
    env.state.pet.is_sleeping = True

    # Valid schema, but pet is sleeping -> Stage 2 Environment Error
    result = env.step(ActionProposal(action="feed"))
    assert result.success is False
    assert result.error is not None
    assert result.error.category == ErrorCategory.ENVIRONMENT
    assert result.error.error_type == ErrorType.PRECONDITION_FAILED
