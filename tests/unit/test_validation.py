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


@pytest.mark.parametrize("data", [
    {"action": "buy", "item": "food", "amount": -2},
    {"action": "buy", "item": "food", "amount": True},
    {"action": "wait", "minutes": False},
    {"action": "wait", "minutes": 0},
    {"action": "wait", "minutes": 1.0},
    {"action": "sleep", "hours": True},
    {"action": "sleep", "hours": 4},
    {"action": "feed", "amount": -5},
    {"action": "dance"},
    {"action": "FEED"},
    {"action": "observe", "unknown": 4},
    {"action": "work", "job_id": 13},
    {"action": "buy"},
    {"action": "observe", "trace": "ignored before 2.0"},
    {"action": "observe", "prediction": {"pet_safe_until_completion": 1}},
])
def test_all_entry_points_reject_invalid_actions_without_mutation(data):
    import json
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        ActionProposal.model_validate(data)
    # Even deliberately bypassing Pydantic construction cannot bypass step validation.
    unchecked = ActionProposal.model_construct(**data).model_copy(update=data)
    for payload in (data, json.dumps(data), unchecked):
        env = TamaEnv()
        env.reset()
        before = env.state.compute_hash()
        result = env.step(payload)
        assert not result.success
        assert not result.completed
        assert result.error.category == ErrorCategory.SCHEMA
        assert result.execution_minutes == 0
        assert env.state.compute_hash() == before


@pytest.mark.parametrize("raw", [
    'prefix {"action":"feed"}',
    '```json\n{"action":"feed"}\n```',
    '{"action":"feed"',
    '{"action":"feed"} trailing',
    '{"action":"feed","action":"wait"}',
    '{"action":"wait","minutes":NaN}',
    '{"action":"wait","minutes":Infinity}',
])
def test_strict_json_rejects_repairs_and_ambiguous_inputs(raw):
    proposal, error = SyntaxValidator.validate_raw(raw)
    assert proposal is None
    assert error.error_type == ErrorType.INVALID_JSON


@pytest.mark.parametrize("hours", [3, 5, 8])
def test_sleep_hours_survive_canonical_json_validation(hours):
    import json
    proposal, error = SyntaxValidator.validate_raw(json.dumps({"action": "sleep", "hours": hours}))
    assert error is None
    assert proposal.hours == hours
    env = TamaEnv()
    env.reset()
    result = env.step(json.dumps({"action": "sleep", "hours": hours}))
    assert result.execution_minutes == hours * 60


def test_json_dict_and_typed_proposals_have_identical_defaults_and_effects():
    snapshots = []
    for proposal in ('{"action":"wait"}', {"action": "wait"}, ActionProposal(action="wait")):
        env = TamaEnv()
        env.reset()
        result = env.step(proposal)
        assert result.execution_minutes == 30
        snapshots.append(result.model_dump())
    assert snapshots[0] == snapshots[1] == snapshots[2]
