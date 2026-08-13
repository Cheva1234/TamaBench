"""Deterministic Mock Torture Test for TamaBench Error Taxonomy.

Assers 100.0% classification accuracy of Stage 1 Schema Errors and Stage 2 Environment Errors
using deterministic fixture payloads.
"""

import pytest
from tamabench.env.core import TamaEnv
from tamabench.schemas.actions import ActionProposal
from tamabench.schemas.errors import ErrorCategory, ErrorType
from tamabench.validation.syntax_validator import SyntaxValidator


MOCK_FIXTURES = [
    # 1. Invalid JSON string
    ("{invalid json syntax", ErrorCategory.SCHEMA, ErrorType.INVALID_JSON),
    # 2. Unknown action name
    ('{"action": "fly_to_mars"}', ErrorCategory.SCHEMA, ErrorType.UNKNOWN_ACTION),
    # 3. Missing required job_id argument for work action
    ('{"action": "work"}', ErrorCategory.SCHEMA, ErrorType.MISSING_ARGUMENT),
    # 4. Wrong argument type (string instead of integer for amount)
    ('{"action": "buy", "item": "food", "amount": "ten"}', ErrorCategory.SCHEMA, ErrorType.WRONG_TYPE),
    # 5. Out of range argument (amount <= 0)
    ('{"action": "buy", "item": "food", "amount": -2}', ErrorCategory.SCHEMA, ErrorType.OUT_OF_RANGE),
]


@pytest.mark.unit
@pytest.mark.parametrize("payload, expected_category, expected_error_type", MOCK_FIXTURES)
def test_taxonomy_mock_fixtures(payload, expected_category, expected_error_type):
    proposal, err = SyntaxValidator.validate_raw(payload)
    assert proposal is None
    assert err is not None
    assert err.category == expected_category
    assert err.error_type == expected_error_type


@pytest.mark.unit
def test_environment_precondition_mock_fixtures():
    env = TamaEnv()
    env.reset(seed=42)

    # 1. Feed when pet is sleeping
    env.state.pet.is_sleeping = True
    res_sleep_feed = env.step(ActionProposal(action="feed"))
    assert res_sleep_feed.success is False
    assert res_sleep_feed.error.category == ErrorCategory.ENVIRONMENT
    assert res_sleep_feed.error.error_type == ErrorType.PRECONDITION_FAILED

    # 2. Feed when inventory food is 0
    env.state.pet.is_sleeping = False
    env.state.inventory.food = 0
    res_no_food = env.step(ActionProposal(action="feed"))
    assert res_no_food.success is False
    assert res_no_food.error.category == ErrorCategory.ENVIRONMENT
    assert res_no_food.error.error_type == ErrorType.INSUFFICIENT_RESOURCE

    # 3. Buy item not sold in shop
    res_bad_shop = env.step(ActionProposal(action="buy", item="spaceship", amount=1))
    assert res_bad_shop.success is False
    assert res_bad_shop.error.category == ErrorCategory.ENVIRONMENT
    assert res_bad_shop.error.error_type == ErrorType.ACTION_UNAVAILABLE
