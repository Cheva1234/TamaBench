"""Offline syntax and execution checks; no Colab service or pip networking."""
import io
import json
from pathlib import Path
import subprocess
import sys
import types
import zipfile

import pytest


NOTEBOOK = Path(__file__).resolve().parents[2] / "notebooks" / "TamaBench_Colab.ipynb"


def _cells():
    notebook = json.loads(NOTEBOOK.read_text())
    assert notebook["nbformat"] == 4
    assert len(notebook["cells"]) == 4
    return ["".join(cell["source"]) for cell in notebook["cells"]]


def test_notebook_has_four_thin_valid_cells():
    for index, source in enumerate(_cells()):
        compile(source, f"cell_{index}", "exec")
    assert "run_experiment(config)" in _cells()[2]
    assert "TamaEnv(" not in "\n".join(_cells())


def test_notebook_cpu_flow_offline(tmp_path, monkeypatch):
    uploaded = io.BytesIO()
    with zipfile.ZipFile(uploaded, "w") as archive:
        archive.writestr("TamaBench/pyproject.toml", "[project]\nname='tamabench'\nversion='2.0.0'\n")
    downloads = []
    files = types.SimpleNamespace(upload=lambda: {"source.zip": uploaded.getvalue()}, download=downloads.append)
    colab = types.ModuleType("google.colab")
    colab.files = files
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    monkeypatch.setattr(subprocess, "check_call", lambda args: 0)
    namespace = {}
    cells = _cells()
    cells[0] = cells[0].replace('REF = "baf81456053e736031a0c46da9ae1e782d020500"', 'REF = ""')
    cells[1] = cells[1].replace('"/content/tamabench-output"', repr(str(tmp_path / "results")))
    cells[3] = cells[3].replace('"/content/tamabench-results"', repr(str(tmp_path / "archive")))
    for source in cells:
        exec(compile(source, "notebook", "exec"), namespace)
    assert namespace["result"].run_ids
    assert Path(downloads[0]).is_file()
    with zipfile.ZipFile(downloads[0]) as archive:
        assert "manifest.json" in archive.namelist()


def test_notebook_rejects_source_zip_traversal(monkeypatch):
    uploaded = io.BytesIO()
    with zipfile.ZipFile(uploaded, "w") as archive:
        archive.writestr("../../outside.py", "unsafe")
    colab = types.ModuleType("google.colab")
    colab.files = types.SimpleNamespace(upload=lambda: {"source.zip": uploaded.getvalue()})
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    monkeypatch.setattr(subprocess, "check_call", lambda args: pytest.fail("unsafe ZIP reached installation"))
    with pytest.raises(ValueError, match="Unsafe path"):
        source = _cells()[0].replace('REF = "baf81456053e736031a0c46da9ae1e782d020500"', 'REF = ""')
        exec(compile(source, "install", "exec"), {})


def test_notebook_default_installs_published_commit_without_upload(monkeypatch):
    calls = []
    monkeypatch.setattr(subprocess, "check_call", calls.append)
    exec(compile(_cells()[0], "install", "exec"), {})
    assert len(calls) == 1
    assert calls[0][-1] == "git+https://github.com/Cheva1234/TamaBench.git@baf81456053e736031a0c46da9ae1e782d020500"


def test_notebook_local_checkout_takes_priority_over_default_ref(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='tamabench'\n")
    calls = []
    monkeypatch.setattr(subprocess, "check_call", calls.append)
    source = _cells()[0].replace('LOCAL_CHECKOUT = ""', f'LOCAL_CHECKOUT = {str(tmp_path)!r}')
    exec(compile(source, "install", "exec"), {})
    assert len(calls) == 1 and calls[0][-1] == str(tmp_path)


@pytest.mark.parametrize('provider,key_name', [('openai','OPENAI_API_KEY'), ('openrouter','OPENROUTER_API_KEY'), ('groq','GROQ_API_KEY')])
def test_notebook_api_flow_with_secrets_and_mocked_response(tmp_path, monkeypatch, capsys, provider, key_name):
    import requests
    monkeypatch.delenv(key_name, raising=False)
    secret = 'notebook-secret-never-print'
    names = []
    def get_secret(name):
        names.append(name)
        return secret
    colab = types.ModuleType('google.colab')
    colab.userdata = types.SimpleNamespace(get=get_secret)
    monkeypatch.setitem(sys.modules, 'google.colab', colab)
    calls = []
    def post(self, url, **kwargs):
        calls.append(kwargs)
        response = requests.Response(); response.status_code = 200
        response._content = b'{"choices":[{"message":{"content":"{\\"action\\":\\"wait\\",\\"minutes\\":300}"},"finish_reason":"stop"}],"usage":{"prompt_tokens":50,"completion_tokens":10}}'
        return response
    monkeypatch.setattr(requests.Session, 'post', post)
    source = _cells()[1].replace('PROVIDER = "cpu"', f'PROVIDER = "{provider}"').replace('MODEL = ""', 'MODEL = "fixture"')
    source = source.replace('"/content/tamabench-output"', repr(str(tmp_path)))
    namespace = {}
    exec(compile(source, 'configure', 'exec'), namespace)
    assert names == [key_name] and not calls
    exec(compile(_cells()[2], 'run', 'exec'), namespace)
    assert namespace['result'].metrics[0].status == 'completed'
    assert len(calls) == 1
    assert calls[0]['headers']['Authorization'] == 'Bearer ' + secret
    assert secret not in capsys.readouterr().out
    assert secret not in (tmp_path / 'manifest.json').read_text()


def test_notebook_missing_secret_actionable_without_leaking_exception(tmp_path, monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    colab = types.ModuleType('google.colab')
    colab.userdata = types.SimpleNamespace(get=lambda name: (_ for _ in ()).throw(RuntimeError('sensitive-provider-text')))
    monkeypatch.setitem(sys.modules, 'google.colab', colab)
    source = _cells()[1].replace('PROVIDER = "cpu"', 'PROVIDER = "openai"').replace('MODEL = ""', 'MODEL = "fixture"')
    source = source.replace('"/content/tamabench-output"', repr(str(tmp_path)))
    with pytest.raises(RuntimeError, match='Add OPENAI_API_KEY') as error:
        exec(compile(source, 'configure', 'exec'), {})
    assert 'sensitive-provider-text' not in str(error.value)
