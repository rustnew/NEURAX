"""A model id the provider no longer serves must not end the run.

The studio saved `accounts/fireworks/models/llama-v3p1-70b-instruct` as its
Fireworks default; Fireworks later stopped serving it. Every run under a
configuration saved before that went out with a valid key, a valid endpoint
and correct routing, and died on

    404 - Model not found, inaccessible, and/or not deployed

with no way for the loop to know that only one field was wrong. The studio's
model picker fixes new configurations; it cannot reach one already sitting in
a user's localStorage. This is the half that can.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio

import pytest

import langchain_runner


FIREWORKS_404 = (
    "Error code: 404 - {'error': {'message': 'Model not found, inaccessible, "
    "and/or not deployed', 'param': 'model', 'code': 'NOT_FOUND', "
    "'type': 'error'}}"
)


class TestRecognisingTheFailure:
    def test_the_real_fireworks_refusal_is_recognised(self):
        assert langchain_runner._is_model_not_found(Exception(FIREWORKS_404))

    @pytest.mark.parametrize("message", [
        "Error code: 404 - The model `gpt-9` does not exist",
        "invalid model name",
        "Unknown model: claude-2",
    ])
    def test_other_providers_wordings_too(self, message):
        assert langchain_runner._is_model_not_found(Exception(message))

    @pytest.mark.parametrize("message", [
        "Error code: 401 - Incorrect API key provided",
        "Error code: 429 - Rate limit reached",
        "Connection error",
    ])
    def test_but_nothing_else(self, message):
        # Substituting a model would not help any of these, and doing it
        # would hide the real reason behind a second, unrelated failure.
        assert not langchain_runner._is_model_not_found(Exception(message))


class TestAskingTheKeyWhatItCanRun:
    def test_each_provider_is_asked_at_its_own_endpoint(self, monkeypatch):
        seen = {}

        class FakeResponse:
            def __init__(self, payload):
                self._payload = payload

            def raise_for_status(self):
                pass

            def json(self):
                return self._payload

        class FakeClient:
            def __init__(self, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def get(self, url, params=None, headers=None):
                seen["url"] = url
                seen["params"] = params or {}
                seen["headers"] = headers or {}
                if "generativelanguage" in url:
                    return FakeResponse({"models": [
                        {"name": "models/gemini-2.5-pro",
                         "supportedGenerationMethods": ["generateContent"]},
                        {"name": "models/text-embedding-004",
                         "supportedGenerationMethods": ["embedContent"]},
                    ]})
                return FakeResponse({"data": [{"id": "a-chat-model"}]})

        import httpx
        monkeypatch.setattr(httpx, "AsyncClient", FakeClient)

        models = asyncio.run(langchain_runner.list_available_models(
            {"provider": "fireworks", "api_key": "fw-key"}
        ))
        assert seen["url"] == "https://api.fireworks.ai/inference/v1/models"
        assert seen["headers"]["Authorization"] == "Bearer fw-key"
        assert models == ["a-chat-model"]

        asyncio.run(langchain_runner.list_available_models(
            {"provider": "anthropic", "api_key": "sk-ant"}
        ))
        assert seen["url"] == "https://api.anthropic.com/v1/models"
        assert seen["headers"]["x-api-key"] == "sk-ant"
        assert seen["headers"]["anthropic-version"] == "2023-06-01"

        google = asyncio.run(langchain_runner.list_available_models(
            {"provider": "google", "api_key": "AIza"}
        ))
        assert seen["params"]["key"] == "AIza"
        # The `models/` prefix is the API's namespacing, not part of the id,
        # and the embedding model is filtered out by the API's own answer.
        assert google == ["gemini-2.5-pro"]

    def test_a_failed_lookup_is_silent_not_fatal(self, monkeypatch):
        async def _boom(*a, **kw):
            raise RuntimeError("provider unreachable")

        monkeypatch.setattr(langchain_runner, "list_available_models", _boom)
        # The caller must be able to fall back to reporting the *original*
        # provider error, not one about this lookup.
        with pytest.raises(RuntimeError):
            asyncio.run(langchain_runner.list_available_models({}))

    def test_an_instruction_tuned_model_is_preferred(self, monkeypatch):
        async def _models(_creds):
            return ["llama-4-base", "llama-4-instruct", "text-embedding-3"]

        monkeypatch.setattr(langchain_runner, "list_available_models", _models)
        # A base completion model rarely survives a structured-output loop.
        assert asyncio.run(langchain_runner.pick_available_model({})) == "llama-4-instruct"

    def test_no_models_means_no_substitution(self, monkeypatch):
        async def _none(_creds):
            return []

        monkeypatch.setattr(langchain_runner, "list_available_models", _none)
        assert asyncio.run(langchain_runner.pick_available_model({})) is None


class TestTheLoopRecovers:
    def test_a_dead_model_is_replaced_and_the_step_succeeds(self, monkeypatch):
        built_with: list[dict] = []

        class FakeStructured:
            def __init__(self, model):
                self._model = model

            async def ainvoke(self, _messages):
                if self._model == "accounts/fireworks/models/llama-v3p1-70b-instruct":
                    raise RuntimeError(FIREWORKS_404)
                return langchain_runner._ControllerStep(
                    assistant="Adding the input block.",
                    tool=langchain_runner._ToolCall(name="add_node", args={"layer_type": "input"}),
                )

        class FakeLlm:
            def __init__(self, model):
                self._model = model

            def with_structured_output(self, _schema):
                return FakeStructured(self._model)

        def _fake_make_chat_model(credentials=None, **kwargs):
            built_with.append(dict(credentials or {}))
            return FakeLlm(str((credentials or {}).get("model") or ""))

        async def _pick(_creds):
            return "accounts/fireworks/models/llama4-maverick-instruct-basic"

        monkeypatch.setattr(langchain_runner, "make_chat_model", _fake_make_chat_model)
        monkeypatch.setattr(langchain_runner, "pick_available_model", _pick)

        result = asyncio.run(langchain_runner.run_controller_step(
            user_message="build a small cnn",
            snapshot={"family": "cnn", "nodes": [], "connections": []},
            history=[],
            credentials={
                "provider": "fireworks",
                "api_key": "fw-key",
                "model": "accounts/fireworks/models/llama-v3p1-70b-instruct",
            },
        ))

        # The step actually produced its tool call rather than failing.
        assert result["tool"]["name"] == "add_node"
        # The client was rebuilt with the model the key can really run.
        assert built_with[-1]["model"] == "accounts/fireworks/models/llama4-maverick-instruct-basic"
        # And the user is told, once, with what to do about it.
        assert "llama4-maverick-instruct-basic" in result["assistant"]
        assert "Account → API & Agent" in result["assistant"]
        assert "Adding the input block." in result["assistant"]

    def test_the_original_error_stands_when_no_model_can_replace_it(self, monkeypatch):
        class FakeStructured:
            async def ainvoke(self, _messages):
                raise RuntimeError(FIREWORKS_404)

        class FakeLlm:
            def with_structured_output(self, _schema):
                return FakeStructured()

        monkeypatch.setattr(langchain_runner, "make_chat_model", lambda **kw: FakeLlm())

        async def _none(_creds):
            return None

        monkeypatch.setattr(langchain_runner, "pick_available_model", _none)

        with pytest.raises(ValueError, match="Model not found"):
            asyncio.run(langchain_runner.run_controller_step(
                user_message="build it",
                snapshot={"family": "cnn", "nodes": [], "connections": []},
                history=[],
                credentials={"provider": "fireworks", "api_key": "k", "model": "dead"},
            ))
