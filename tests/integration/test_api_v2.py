"""Local in-process API checks; no external leaderboard requests."""
import importlib
import pytest

pytest.importorskip('fastapi')
from fastapi.testclient import TestClient


def test_api_submissions_are_bounded_and_explicitly_unverified(tmp_path, monkeypatch):
    module = importlib.import_module('tamabench.api.main')
    monkeypatch.setattr(module, 'DB_PATH', str(tmp_path/'leaderboard.sqlite'))
    with TestClient(module.app) as client:
        payload = {'run_id':'fixture', 'agent_name':'test', 'survived':True,
                   'simulated_days':7.0, 'avg_health':90.0, 'score':7000.0}
        result = client.post('/submit', json=payload)
        assert result.status_code == 200
        assert result.json()['verification_status'] == 'unverified'
        assert client.post('/submit', json=payload).status_code == 409
        board = client.get('/leaderboard').json()
        assert board['verification_status'] == 'unverified'
        assert board['leaderboard'][0]['agent_name'] == 'test'
        assert client.get('/leaderboard?limit=-1').status_code == 422
        assert client.get('/leaderboard?limit=10000').status_code == 422
        payload['run_id'] = 'bad'; payload['avg_health'] = 101
        assert client.post('/submit', json=payload).status_code == 422
