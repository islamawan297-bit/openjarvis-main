"""Tests for model registry and diagnostic endpoints (/v1/models/registry, test-connection, test-prompt)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from openjarvis.server.app import create_app
from openjarvis.intelligence.model_registry import (
    STATUS_VERIFIED_WORKING,
    STATUS_AUTHENTICATION_FAILED,
    STATUS_INSUFFICIENT_CREDITS,
    STATUS_RATE_LIMITED,
    mask_key,
    check_local_hardware_compatibility,
)


def _make_mock_engine():
    engine = MagicMock()
    engine.engine_id = "mock"
    engine.health.return_value = True
    engine.list_models.return_value = ["test-model"]
    return engine


@pytest.fixture
def client():
    engine = _make_mock_engine()
    app = create_app(engine, "test-model", engine_name="mock")
    return TestClient(app)


class TestModelRegistryEndpoints:
    def test_get_registry_success(self, client):
        resp = client.get("/v1/models/registry")
        assert resp.status_code == 200
        data = resp.json()
        assert "providers" in data
        assert "hardware" in data
        assert "summary" in data
        assert isinstance(data["providers"], list)
        assert len(data["providers"]) > 0
        # Verify providers contain models
        for p in data["providers"]:
            assert "id" in p
            assert "name" in p
            assert "models" in p
            assert isinstance(p["models"], list)

    def test_test_connection_requires_provider(self, client):
        resp = client.post("/v1/models/test-connection", json={})
        assert resp.status_code == 400
        assert "provider" in resp.json()["detail"].lower()

    def test_test_connection_mocked_success(self, client):
        mock_result = {
            "status": STATUS_VERIFIED_WORKING,
            "latency_ms": 120,
            "error": None,
        }
        with patch("openjarvis.intelligence.model_registry.test_provider_connection", new_callable=AsyncMock) as mock_test:
            mock_test.return_value = mock_result
            resp = client.post("/v1/models/test-connection", json={"provider": "groq"})
            assert resp.status_code == 200
            assert resp.json()["status"] == STATUS_VERIFIED_WORKING
            assert resp.json()["latency_ms"] == 120

    def test_test_connection_mocked_auth_failure(self, client):
        mock_result = {
            "status": STATUS_AUTHENTICATION_FAILED,
            "latency_ms": None,
            "error": "Invalid API key provided",
        }
        with patch("openjarvis.intelligence.model_registry.test_provider_connection", new_callable=AsyncMock) as mock_test:
            mock_test.return_value = mock_result
            resp = client.post("/v1/models/test-connection", json={"provider": "openai"})
            assert resp.status_code == 200
            assert resp.json()["status"] == STATUS_AUTHENTICATION_FAILED
            assert "Invalid API key" in resp.json()["error"]

    def test_test_prompt_requires_fields(self, client):
        resp1 = client.post("/v1/models/test-prompt", json={"provider": "groq"})
        assert resp1.status_code == 400
        assert "model" in resp1.json()["detail"].lower()

        resp2 = client.post("/v1/models/test-prompt", json={"model": "llama-3.3-70b-versatile"})
        assert resp2.status_code == 400
        assert "provider" in resp2.json()["detail"].lower()

    def test_test_prompt_mocked_success(self, client):
        mock_result = {
            "status": STATUS_VERIFIED_WORKING,
            "latency_ms": 250,
            "sample_output": "Hello! System operational.",
            "error": None,
            "timestamp": 123456789.0,
        }
        with patch("openjarvis.intelligence.model_registry.test_model_inference", new_callable=AsyncMock) as mock_test:
            mock_test.return_value = mock_result
            resp = client.post(
                "/v1/models/test-prompt",
                json={"provider": "groq", "model": "llama-3.3-70b-versatile"},
            )
            assert resp.status_code == 200
            assert resp.json()["status"] == STATUS_VERIFIED_WORKING
            assert resp.json()["sample_output"] == "Hello! System operational."

    def test_test_prompt_mocked_credit_exhausted_scenario(self, client):
        mock_result = {
            "status": STATUS_INSUFFICIENT_CREDITS,
            "latency_ms": None,
            "error": "You exceeded your current quota, please check your plan and billing details.",
            "timestamp": 123456789.0,
        }
        with patch("openjarvis.intelligence.model_registry.test_model_inference", new_callable=AsyncMock) as mock_test:
            mock_test.return_value = mock_result
            resp = client.post(
                "/v1/models/test-prompt",
                json={"provider": "openai", "model": "gpt-4o"},
            )
            assert resp.status_code == 200
            assert resp.json()["status"] == STATUS_INSUFFICIENT_CREDITS
            assert "quota" in resp.json()["error"]

    def test_mask_key_security(self):
        assert mask_key(None) is None
        assert mask_key("") is None
        assert mask_key("short") == "****"
        masked = mask_key("sk-abcdef1234567890abcdef12")
        assert masked == "sk-a...ef12"
        assert "1234567890" not in masked

    def test_hardware_compatibility_check(self):
        hw = check_local_hardware_compatibility()
        assert "ram_total_gb" in hw
        assert "ram_available_gb" in hw
        assert "can_run_local_7b" in hw
        assert isinstance(hw["can_run_local_7b"], bool)
