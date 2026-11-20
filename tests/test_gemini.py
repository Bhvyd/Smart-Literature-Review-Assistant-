
import json

import httpx
import pytest

from app.config import Settings
from services.llm import DEFAULT_MODELS, LLMClient, LLMUnavailable, resolve_model
from store.db import connect, init_db


class Gemini:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.requests = []

    def __call__(self, request):
        self.requests.append({"url": str(request.url), "headers": dict(request.headers),
                              "body": json.loads(request.content)})
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        return reply(request) if callable(reply) else reply


def ok(text="An answer.", finish="STOP", parts=None):
    parts = parts if parts is not None else [{"text": text}]
    return httpx.Response(200, json={
        "candidates": [{"content": {"parts": parts, "role": "model"}, "finishReason": finish}],
        "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 7, "totalTokenCount": 12},
    })


def err(status, message, code=None, status_name="", details=None):
    return httpx.Response(status, json={"error": {"code": code or status, "message": message,
                                                   "status": status_name, "details": details or []}})


@pytest.fixture
def db_path(tmp_path):
    path = tmp_path / "g.sqlite3"
    init_db(path)
    return path


def client(db_path, provider, model="gemini-3.8-flash", sleeps=None):
    return LLMClient(api_key="AQ.fake-key", model=model, db_path=db_path, provider="gemini",
                     transport=httpx.MockTransport(provider),
                     sleep=(sleeps.append if sleeps is not None else (lambda s: None)))


def test_the_request_matches_the_documented_gemini_shape(db_path):
    provider = Gemini(ok())
    client(db_path, provider).complete("be brief", "hello", max_output_tokens=321)
    request = provider.requests[0]
    assert request["url"].endswith("/v1beta/models/gemini-3.8-flash:generateContent")
    assert request["headers"]["x-goog-api-key"] == "AQ.fake-key"
    assert "authorization" not in request["headers"]
    body = request["body"]
    assert body["systemInstruction"] == {"parts": [{"text": "be brief"}]}
    assert body["contents"] == [{"role": "user", "parts": [{"text": "hello"}]}]
    assert body["generationConfig"] == {"maxOutputTokens": 321, "thinkingConfig": {"thinkingBudget": 0}}


def test_text_parts_are_joined_and_hidden_thought_parts_are_dropped(db_path):
    parts = [{"text": "private reasoning", "thought": True}, {"text": "Hello "}, {"text": "world."}]
    result = client(db_path, Gemini(ok(parts=parts))).complete("s", "u")
    assert result.text == "Hello world." and result.usage["totalTokenCount"] == 12


def test_identical_calls_are_cached_and_the_provider_is_part_of_the_key(db_path):
    provider = Gemini(ok("first"))
    llm = client(db_path, provider)
    assert llm.complete("s", "u").cached is False
    assert llm.complete("s", "u").cached is True
    assert len(provider.requests) == 1
    other = LLMClient(api_key="sk-x", model="gemini-3.8-flash", db_path=db_path, provider="openai",
                      transport=httpx.MockTransport(Gemini(httpx.Response(
                          200, json={"choices": [{"message": {"content": "from openai"}}]}))))
    assert other.complete("s", "u").text == "from openai"


def test_a_truncated_reply_is_an_error_and_is_never_cached(db_path):
    llm = client(db_path, Gemini(ok("The answer is cut o", finish="MAX_TOKENS")))
    with pytest.raises(LLMUnavailable) as caught:
        llm.complete("s", "u")
    assert caught.value.code == "truncated"
    assert connect(db_path).execute("SELECT count(*) FROM llm_cache").fetchone()[0] == 0


def test_a_blocked_prompt_has_its_own_code(db_path):
    blocked = httpx.Response(200, json={"promptFeedback": {"blockReason": "SAFETY"}})
    with pytest.raises(LLMUnavailable) as caught:
        client(db_path, Gemini(blocked)).complete("s", "u")
    assert caught.value.code == "blocked" and "SAFETY" in caught.value.message


def test_an_empty_reply_is_an_error_not_an_empty_answer(db_path):
    with pytest.raises(LLMUnavailable) as caught:
        client(db_path, Gemini(ok(parts=[], finish="STOP"))).complete("s", "u")
    assert caught.value.code == "empty_response"


def test_a_rate_limit_is_retried_after_the_providers_own_delay(db_path):
    limited = err(429, "Resource exhausted", status_name="RESOURCE_EXHAUSTED",
                  details=[{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "12s"}])
    sleeps = []
    provider = Gemini(limited, ok("after the wait"))
    result = client(db_path, provider, sleeps=sleeps).complete("s", "u")
    assert result.text == "after the wait" and len(provider.requests) == 2
    assert sleeps == [12.5]


def test_a_persistent_rate_limit_gives_up_after_three_attempts_with_a_clear_code(db_path):
    provider = Gemini(err(429, "slow down", status_name="RESOURCE_EXHAUSTED"))
    llm = client(db_path, provider)
    with pytest.raises(LLMUnavailable) as caught:
        llm.complete("s", "u")
    assert caught.value.code == "rate_limited" and caught.value.status == 429
    assert len(provider.requests) == 3
    assert llm.last_error["code"] == "rate_limited"


def test_a_daily_quota_is_insufficient_quota_and_is_not_retried(db_path):
    provider = Gemini(err(429, "You exceeded your current quota (GenerateRequestsPerDayPerProject)",
                          status_name="RESOURCE_EXHAUSTED"))
    with pytest.raises(LLMUnavailable) as caught:
        client(db_path, provider).complete("s", "u")
    assert caught.value.code == "insufficient_quota" and len(provider.requests) == 1


def test_a_retired_model_is_reported_by_name(db_path):
    message = "models/gemini-2.5-flash is no longer available to new users"
    with pytest.raises(LLMUnavailable) as caught:
        client(db_path, Gemini(err(404, message, status_name="NOT_FOUND")), model="gemini-2.5-flash").complete("s", "u")
    assert caught.value.code == "model_not_found" and "gemini-2.5-flash" in caught.value.message


@pytest.mark.parametrize("response", [
    err(400, "API key not valid. Please pass a valid API key.", status_name="INVALID_ARGUMENT"),
    err(403, "denied", status_name="PERMISSION_DENIED"),
    err(401, "bad", status_name="UNAUTHENTICATED"),
])
def test_bad_keys_never_echo_the_key_or_the_providers_text(db_path, response):
    with pytest.raises(LLMUnavailable) as caught:
        client(db_path, Gemini(response)).complete("s", "u")
    assert caught.value.code == "invalid_api_key" and "AQ.fake-key" not in caught.value.message


def test_a_model_that_rejects_the_thinking_setting_is_retried_without_it(db_path):
    rejected = err(400, "Request contains an invalid argument.", status_name="INVALID_ARGUMENT")
    provider = Gemini(rejected, ok("worked without it"))
    result = client(db_path, provider).complete("s", "u")
    assert result.text == "worked without it"
    assert "thinkingConfig" in provider.requests[0]["body"]["generationConfig"]
    assert "thinkingConfig" not in provider.requests[1]["body"]["generationConfig"]


def test_a_stale_openai_model_name_next_to_a_gemini_key_is_replaced_and_reported():
    model, note = resolve_model("gemini", "gpt-5-nano")
    assert model == DEFAULT_MODELS["gemini"] and "gpt-5-nano" in note
    assert resolve_model("gemini", "gemini-3.5-flash-lite") == ("gemini-3.5-flash-lite", None)
    assert resolve_model("gemini", "models/gemini-3.8-flash") == ("gemini-3.8-flash", None)
    assert resolve_model("gemini", None) == (DEFAULT_MODELS["gemini"], None)
    assert resolve_model("openai", "gemini-3.8-flash")[0] == DEFAULT_MODELS["openai"]


def test_the_owners_mixed_case_variable_name_selects_gemini(tmp_path, monkeypatch):
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY", "LLM_MODEL", "LLM_PROVIDER"):
        monkeypatch.delenv(name, raising=False)
    env = tmp_path / ".env"
    env.write_text("Gemini_API_KEY=AQ.fake-key\nLLM_MODEL=gpt-5-nano\n", encoding="utf-8")
    settings = Settings.from_env(env)
    assert settings.llm_provider == "gemini" and settings.llm_api_key == "AQ.fake-key"
    assert settings.llm_model == DEFAULT_MODELS["gemini"] and "gpt-5-nano" in settings.llm_config_note


def test_an_openai_only_environment_still_selects_openai(tmp_path, monkeypatch):
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY", "LLM_MODEL", "LLM_PROVIDER"):
        monkeypatch.delenv(name, raising=False)
    env = tmp_path / ".env"
    env.write_text("OPENAI_API_KEY=sk-fake\n", encoding="utf-8")
    settings = Settings.from_env(env)
    assert (settings.llm_provider, settings.llm_model) == ("openai", DEFAULT_MODELS["openai"])


def test_health_reports_the_provider_model_and_any_config_slip(tmp_path):
    from fastapi.testclient import TestClient

    from app.main import create_app

    settings = Settings(db_path=tmp_path / "h.sqlite3", data_dir=tmp_path, llm_api_key="AQ.fake-key",
                        llm_provider="gemini", llm_model="gemini-3.8-flash",
                        llm_config_note="LLM_MODEL='gpt-5-nano' is not a gemini model")
    with TestClient(create_app(settings, warm_query_embedder=False)) as api:
        body = api.get("/api/health").json()
    assert body["generation_provider"] == "gemini" and body["generation_model"] == "gemini-3.8-flash"
    assert "gpt-5-nano" in body["generation_config_note"] and "AQ.fake-key" not in json.dumps(body)


def test_googles_standard_billing_wording_on_a_per_minute_limit_is_a_retryable_rate_limit(db_path):
    message = "You exceeded your current quota, please check your plan and billing details."
    per_minute = err(429, message, status_name="RESOURCE_EXHAUSTED", details=[
        {"@type": "type.googleapis.com/google.rpc.QuotaFailure",
         "violations": [{"quotaId": "GenerateRequestsPerMinutePerProjectPerModel"}]},
        {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "7s"}])
    provider = Gemini(per_minute, ok("recovered"))
    result = client(db_path, provider).complete("s", "u")
    assert result.text == "recovered" and len(provider.requests) == 2


def test_a_per_day_limit_is_still_an_exhausted_quota(db_path):
    daily = err(429, "You exceeded your current quota, please check your plan and billing details.",
                status_name="RESOURCE_EXHAUSTED", details=[
                    {"@type": "type.googleapis.com/google.rpc.QuotaFailure",
                     "violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]}])
    provider = Gemini(daily)
    with pytest.raises(LLMUnavailable) as caught:
        client(db_path, provider).complete("s", "u")
    assert caught.value.code == "insufficient_quota" and len(provider.requests) == 1
