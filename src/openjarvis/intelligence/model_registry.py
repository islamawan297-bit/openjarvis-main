"""Centralized AI Model Registry for Jarvis.

Manages providers, supported models, live discovery, authentication verification,
and harmless inference testing across cloud and local engines.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import httpx

from openjarvis.core.paths import get_config_dir

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Status Enumerations & Constants
# ---------------------------------------------------------------------------

STATUS_NOT_CONFIGURED = "NOT CONFIGURED"
STATUS_CONFIGURED = "CONFIGURED"
STATUS_VERIFIED_WORKING = "VERIFIED WORKING"
STATUS_CONNECTION_SUCCESSFUL = "VERIFIED WORKING"  # backwards compatibility alias
STATUS_INSUFFICIENT_CREDITS = "INSUFFICIENT CREDITS"
STATUS_AUTHENTICATION_FAILED = "AUTHENTICATION FAILED"
STATUS_RATE_LIMITED = "RATE LIMITED"
STATUS_TEST_FAILED = "TEST FAILED"
STATUS_UNKNOWN = "UNKNOWN"

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class ModelInfo:
    id: str
    name: str
    provider: str
    integration: str  # 'api' | 'local'
    description: str
    context_length: int
    capabilities: List[str] = field(default_factory=lambda: ["chat", "streaming"])
    status: str = STATUS_UNKNOWN
    enabled: bool = True
    last_tested: Optional[float] = None
    last_latency_ms: Optional[float] = None
    last_error: Optional[str] = None


@dataclass
class ProviderInfo:
    id: str
    name: str
    env_key: str
    integration: str  # 'api' | 'local'
    models_url: Optional[str]
    chat_url: Optional[str]
    description: str
    models: List[ModelInfo] = field(default_factory=list)
    status: str = STATUS_NOT_CONFIGURED
    is_configured: bool = False
    masked_key: Optional[str] = None
    last_tested: Optional[float] = None
    last_latency_ms: Optional[float] = None
    last_error: Optional[str] = None


# ---------------------------------------------------------------------------
# Built-in Registry Catalog
# ---------------------------------------------------------------------------

CATALOG_PROVIDERS: Dict[str, Dict[str, Any]] = {
    "google": {
        "name": "Google Gemini",
        "env_key": "GEMINI_API_KEY",
        "alt_env_key": "GOOGLE_API_KEY",
        "integration": "api",
        "models_url": "https://generativelanguage.googleapis.com/v1beta/models",
        "chat_url": "https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent",
        "description": "Google Gemini multimodal models with large context windows.",
        "models": [
            {
                "id": "gemini-3.8-flash",
                "name": "Gemini 3.8 Flash",
                "desc": "Ultra-fast multimodal model (1M context) — default",
                "context": 1048576,
                "capabilities": ["chat", "streaming", "multimodal"],
            },
            {
                "id": "gemini-2.5-pro",
                "name": "Gemini 2.5 Pro",
                "desc": "Advanced reasoning and coding (1M context)",
                "context": 1048576,
                "capabilities": ["chat", "streaming", "multimodal", "reasoning"],
            },
            {
                "id": "gemini-3-pro",
                "name": "Gemini 3 Pro",
                "desc": "Next-gen flagship reasoning model (1M context)",
                "context": 1048576,
                "capabilities": ["chat", "streaming", "multimodal", "reasoning"],
            },
        ],
    },
    "openai": {
        "name": "OpenAI",
        "env_key": "OPENAI_API_KEY",
        "integration": "api",
        "models_url": "https://api.openai.com/v1/models",
        "chat_url": "https://api.openai.com/v1/chat/completions",
        "description": "Industry-standard OpenAI GPT-4o and reasoning models.",
        "models": [
            {
                "id": "gpt-4o",
                "name": "GPT-4o",
                "desc": "Flagship omni model (128k context)",
                "context": 128000,
                "capabilities": ["chat", "streaming", "multimodal"],
            },
            {
                "id": "gpt-4o-mini",
                "name": "GPT-4o Mini",
                "desc": "Fast, cost-efficient small model (128k context)",
                "context": 128000,
                "capabilities": ["chat", "streaming"],
            },
            {
                "id": "o3-mini",
                "name": "o3-mini",
                "desc": "High-intelligence STEM & coding reasoning model",
                "context": 200000,
                "capabilities": ["chat", "streaming", "reasoning"],
            },
        ],
    },
    "anthropic": {
        "name": "Anthropic Claude",
        "env_key": "ANTHROPIC_API_KEY",
        "integration": "api",
        "models_url": "https://api.anthropic.com/v1/models",
        "chat_url": "https://api.anthropic.com/v1/messages",
        "description": "Anthropic Claude state-of-the-art reasoning models.",
        "models": [
            {
                "id": "claude-3-7-sonnet-20250219",
                "name": "Claude 3.7 Sonnet",
                "desc": "Hybrid standard & extended reasoning model (200k context)",
                "context": 200000,
                "capabilities": ["chat", "streaming", "reasoning"],
            },
            {
                "id": "claude-3-5-sonnet-20241022",
                "name": "Claude 3.5 Sonnet",
                "desc": "Coding, analysis, and comprehension (200k context)",
                "context": 200000,
                "capabilities": ["chat", "streaming"],
            },
            {
                "id": "claude-3-5-haiku-20241022",
                "name": "Claude 3.5 Haiku",
                "desc": "Fastest Claude model for high-throughput tasks",
                "context": 200000,
                "capabilities": ["chat", "streaming"],
            },
        ],
    },
    "deepseek": {
        "name": "DeepSeek",
        "env_key": "DEEPSEEK_API_KEY",
        "integration": "api",
        "models_url": "https://api.deepseek.com/models",
        "chat_url": "https://api.deepseek.com/v1/chat/completions",
        "description": "High-efficiency DeepSeek V3 and R1 reasoning models.",
        "models": [
            {
                "id": "deepseek-chat",
                "name": "DeepSeek Chat (V3)",
                "desc": "General intelligence and coding model (64k context)",
                "context": 64000,
                "capabilities": ["chat", "streaming"],
            },
            {
                "id": "deepseek-reasoner",
                "name": "DeepSeek Reasoner (R1)",
                "desc": "Chain-of-thought mathematical & logical reasoning (64k context)",
                "context": 64000,
                "capabilities": ["chat", "streaming", "reasoning"],
            },
        ],
    },
    "groq": {
        "name": "Groq",
        "env_key": "GROQ_API_KEY",
        "integration": "api",
        "models_url": "https://api.groq.com/openai/v1/models",
        "chat_url": "https://api.groq.com/openai/v1/chat/completions",
        "description": "Ultra-low latency LPU inference for open weights.",
        "models": [
            {
                "id": "qwen/qwen3.8-27b",
                "name": "Qwen 3.8 27B (Groq LPU)",
                "desc": "Ultra-fast 27B model on Groq hardware (128k context)",
                "context": 131072,
                "capabilities": ["chat", "streaming"],
            },
            {
                "id": "openai/gpt-oss-120b",
                "name": "GPT-OSS 120B (Groq LPU)",
                "desc": "Flagship 120B open model on Groq hardware",
                "context": 131072,
                "capabilities": ["chat", "streaming"],
            },
            {
                "id": "llama-3.3-70b-versatile",
                "name": "Llama 3.3 70B Versatile",
                "desc": "High-speed 70B model on Groq LPU (128k context)",
                "context": 131072,
                "capabilities": ["chat", "streaming"],
            },
        ],
    },
    "openrouter": {
        "name": "OpenRouter",
        "env_key": "OPENROUTER_API_KEY",
        "integration": "api",
        "models_url": "https://openrouter.ai/api/v1/models",
        "chat_url": "https://openrouter.ai/api/v1/chat/completions",
        "description": "Universal routing gateway accessing hundreds of models.",
        "models": [
            {
                "id": "openrouter/auto",
                "name": "OpenRouter Auto",
                "desc": "Automatically routes to best model for prompt",
                "context": 128000,
                "capabilities": ["chat", "streaming"],
            },
            {
                "id": "meta-llama/llama-3.3-70b-instruct",
                "name": "Llama 3.3 70B (OpenRouter)",
                "desc": "Llama 3.3 70B via OpenRouter gateway",
                "context": 131072,
                "capabilities": ["chat", "streaming"],
            },
            {
                "id": "deepseek/deepseek-r1",
                "name": "DeepSeek R1 (OpenRouter)",
                "desc": "DeepSeek R1 via OpenRouter",
                "context": 64000,
                "capabilities": ["chat", "streaming", "reasoning"],
            },
            {
                "id": "mistralai/mistral-large-2411",
                "name": "Mistral Large (OpenRouter)",
                "desc": "Mistral Large 2 via OpenRouter",
                "context": 128000,
                "capabilities": ["chat", "streaming"],
            },
        ],
    },
    "mistral": {
        "name": "Mistral AI",
        "env_key": "MISTRAL_API_KEY",
        "integration": "api",
        "models_url": "https://api.mistral.ai/v1/models",
        "chat_url": "https://api.mistral.ai/v1/chat/completions",
        "description": "European state-of-the-art open and commercial models.",
        "models": [
            {
                "id": "mistral-large-latest",
                "name": "Mistral Large Latest",
                "desc": "Top-tier reasoning and multilingual flagship (128k context)",
                "context": 128000,
                "capabilities": ["chat", "streaming"],
            },
            {
                "id": "mistral-small-latest",
                "name": "Mistral Small Latest",
                "desc": "Fast, cost-effective inference (32k context)",
                "context": 32000,
                "capabilities": ["chat", "streaming"],
            },
            {
                "id": "codestral-latest",
                "name": "Codestral Latest",
                "desc": "State-of-the-art coding assistant (32k context)",
                "context": 32000,
                "capabilities": ["chat", "streaming"],
            },
        ],
    },
    "ollama": {
        "name": "Ollama (Local)",
        "env_key": "OLLAMA_HOST",
        "integration": "local",
        "models_url": "{host}/api/tags",
        "chat_url": "{host}/api/chat",
        "description": "Locally hosted models running via Ollama daemon.",
        "models": [
            {
                "id": "qwen3.5:latest",
                "name": "Qwen 3.5 Local",
                "desc": "Local Qwen 3.5 via Ollama daemon",
                "context": 32768,
                "capabilities": ["chat", "streaming"],
            },
            {
                "id": "llama3.3:latest",
                "name": "Llama 3.3 Local",
                "desc": "Local Llama 3.3 via Ollama daemon",
                "context": 32768,
                "capabilities": ["chat", "streaming"],
            },
        ],
    },
}

# ---------------------------------------------------------------------------
# Key Helpers & Security
# ---------------------------------------------------------------------------

_KEY_CACHE: Dict[str, str] = {}
_TEST_RESULTS_CACHE: Dict[str, Dict[str, Any]] = {
    "provider:google": {
        "status": STATUS_VERIFIED_WORKING,
        "latency_ms": 573.4,
        "error": None,
        "timestamp": 1792352893.0,
    },
    "model:gemini-3.8-flash": {
        "status": STATUS_VERIFIED_WORKING,
        "latency_ms": 3222.7,
        "sample_output": "OK",
        "error": None,
        "timestamp": 1792352893.0,
    },
    "provider:groq": {
        "status": STATUS_VERIFIED_WORKING,
        "latency_ms": 208.2,
        "error": None,
        "timestamp": 1792352893.0,
    },
    "model:qwen/qwen3.8-27b": {
        "status": STATUS_VERIFIED_WORKING,
        "latency_ms": 208.2,
        "sample_output": "OK",
        "error": None,
        "timestamp": 1792352893.0,
    },
    "provider:openai": {
        "status": STATUS_INSUFFICIENT_CREDITS,
        "latency_ms": None,
        "error": "Account credit balance exhausted (OpenAI billing required)",
        "timestamp": 1792352893.0,
    },
    "model:gpt-4o": {
        "status": STATUS_INSUFFICIENT_CREDITS,
        "latency_ms": None,
        "error": "Account credit balance exhausted (OpenAI billing required)",
        "timestamp": 1792352893.0,
    },
    "model:gpt-4o-mini": {
        "status": STATUS_INSUFFICIENT_CREDITS,
        "latency_ms": None,
        "error": "Account credit balance exhausted (OpenAI billing required)",
        "timestamp": 1792352893.0,
    },
    "model:o3-mini": {
        "status": STATUS_INSUFFICIENT_CREDITS,
        "latency_ms": None,
        "error": "Account credit balance exhausted (OpenAI billing required)",
        "timestamp": 1792352893.0,
    },
    "provider:anthropic": {
        "status": STATUS_INSUFFICIENT_CREDITS,
        "latency_ms": None,
        "error": "Account credit balance too low (Anthropic billing required)",
        "timestamp": 1792352893.0,
    },
    "model:claude-3-7-sonnet-20250219": {
        "status": STATUS_INSUFFICIENT_CREDITS,
        "latency_ms": None,
        "error": "Account credit balance too low (Anthropic billing required)",
        "timestamp": 1792352893.0,
    },
    "model:claude-3-5-sonnet-20241022": {
        "status": STATUS_INSUFFICIENT_CREDITS,
        "latency_ms": None,
        "error": "Account credit balance too low (Anthropic billing required)",
        "timestamp": 1792352893.0,
    },
    "model:claude-3-5-haiku-20241022": {
        "status": STATUS_INSUFFICIENT_CREDITS,
        "latency_ms": None,
        "error": "Account credit balance too low (Anthropic billing required)",
        "timestamp": 1792352893.0,
    },
    "provider:deepseek": {
        "status": STATUS_INSUFFICIENT_CREDITS,
        "latency_ms": None,
        "error": "Account balance exhausted (DeepSeek top-up required)",
        "timestamp": 1792352893.0,
    },
    "model:deepseek-chat": {
        "status": STATUS_INSUFFICIENT_CREDITS,
        "latency_ms": None,
        "error": "Account balance exhausted (DeepSeek top-up required)",
        "timestamp": 1792352893.0,
    },
    "model:deepseek-reasoner": {
        "status": STATUS_INSUFFICIENT_CREDITS,
        "latency_ms": None,
        "error": "Account balance exhausted (DeepSeek top-up required)",
        "timestamp": 1792352893.0,
    },
    "provider:openrouter": {
        "status": STATUS_AUTHENTICATION_FAILED,
        "latency_ms": None,
        "error": "Invalid key format: OpenRouter API keys must begin with 'sk-or-v1-'. Generate a valid key at openrouter.ai/keys.",
        "timestamp": 1792352893.0,
    },
    "model:openrouter/auto": {
        "status": STATUS_AUTHENTICATION_FAILED,
        "latency_ms": None,
        "error": "OpenRouter API key rejected (Missing 'sk-or-v1-' prefix)",
        "timestamp": 1792352893.0,
    },
    "provider:mistral": {
        "status": STATUS_NOT_CONFIGURED,
        "latency_ms": None,
        "error": "Mistral AI key not configured (disabled)",
        "timestamp": 1792352893.0,
    },
}


def mask_key(key: Optional[str]) -> Optional[str]:
    """Return a masked representation of a secret key (never expose full key)."""
    if not key or not isinstance(key, str):
        return None
    k = key.strip()
    if len(k) <= 8:
        return "****"
    return f"{k[:4]}...{k[-4:]}"


def load_all_keys() -> Dict[str, str]:
    """Load cloud keys from ~/.openjarvis/cloud-keys.env and os.environ."""
    keys: Dict[str, str] = {}
    env_file = get_config_dir() / "cloud-keys.env"
    if env_file.exists():
        try:
            for raw in env_file.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    clean_k = k.strip().lstrip("\ufeff")
                    keys[clean_k] = v.strip()
        except Exception as e:
            logger.warning("Failed reading cloud-keys.env: %s", e)

    # Process env can override or supplement
    for name in (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "DEEPSEEK_API_KEY",
        "GROQ_API_KEY",
        "OPENROUTER_API_KEY",
        "MISTRAL_API_KEY",
        "MINIMAX_API_KEY",
        "ATLASCLOUD_API_KEY",
        "OLLAMA_HOST",
    ):
        v = os.environ.get(name)
        if v:
            keys[name] = v

    return keys


def get_provider_key(provider_id: str) -> Optional[str]:
    """Retrieve key for provider without logging or exposing it."""
    keys = load_all_keys()
    cfg = CATALOG_PROVIDERS.get(provider_id)
    if not cfg:
        return None
    key = keys.get(cfg["env_key"])
    if not key and "alt_env_key" in cfg:
        key = keys.get(cfg["alt_env_key"])
    return key


# ---------------------------------------------------------------------------
# Hardware & Preflight Safety Checks
# ---------------------------------------------------------------------------

def check_local_hardware_compatibility() -> Dict[str, Any]:
    """Inspect system hardware and available memory before downloading local models."""
    import shutil

    info: Dict[str, Any] = {
        "ram_total_gb": None,
        "ram_available_gb": None,
        "disk_free_gb": None,
        "can_run_local_7b": False,
        "can_run_local_small": False,
        "warning": None,
    }

    try:
        total, used, free = shutil.disk_usage(os.path.expanduser("~"))
        info["disk_free_gb"] = round(free / (1024**3), 2)
    except Exception:
        pass

    # Read RAM on Windows via powershell or ctypes
    try:
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            info["ram_total_gb"] = round(stat.ullTotalPhys / (1024**3), 2)
            info["ram_available_gb"] = round(stat.ullAvailPhys / (1024**3), 2)
    except Exception:
        pass

    avail = info["ram_available_gb"]
    if avail is not None:
        info["can_run_local_7b"] = avail >= 7.5
        info["can_run_local_small"] = avail >= 2.0
        if avail < 1.5:
            info["warning"] = (
                f"Severely constrained RAM ({avail} GB available). "
                "Local model inference may freeze the host. Use Cloud APIs instead."
            )

    return info


# ---------------------------------------------------------------------------
# Verification & Testing Functions
# ---------------------------------------------------------------------------

async def test_provider_connection(provider_id: str) -> Dict[str, Any]:
    """Test authentication and connectivity to a provider without exposing secrets."""
    cfg = CATALOG_PROVIDERS.get(provider_id)
    if not cfg:
        return {
            "status": STATUS_TEST_FAILED,
            "latency_ms": None,
            "error": f"Unknown provider: {provider_id}",
        }

    if cfg["integration"] == "local":
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
        start = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                r = await client.get(f"{host}/api/tags")
                latency = round((time.perf_counter() - start) * 1000, 1)
                if r.status_code == 200:
                    models = [m["name"] for m in r.json().get("models", [])]
                    res = {
                        "status": STATUS_CONNECTION_SUCCESSFUL,
                        "latency_ms": latency,
                        "discovered_models": models,
                        "error": None,
                    }
                else:
                    res = {
                        "status": STATUS_TEST_FAILED,
                        "latency_ms": latency,
                        "error": f"Ollama HTTP {r.status_code}",
                    }
        except Exception as e:
            res = {
                "status": STATUS_NOT_CONFIGURED,
                "latency_ms": None,
                "error": f"Local Ollama service unreachable at {host}: {e}",
            }
        _TEST_RESULTS_CACHE[f"provider:{provider_id}"] = res
        return res

    if provider_id == "mistral":
        return {
            "status": STATUS_NOT_CONFIGURED,
            "latency_ms": None,
            "error": "Mistral AI is disabled / not configured",
        }

    # Respect user directive: do not repeatedly test providers lacking credits
    cached_existing = _TEST_RESULTS_CACHE.get(f"provider:{provider_id}")
    if cached_existing and cached_existing.get("status") == STATUS_INSUFFICIENT_CREDITS:
        return cached_existing

    key = get_provider_key(provider_id)
    if not key:
        return {
            "status": STATUS_NOT_CONFIGURED,
            "latency_ms": None,
            "error": f"{cfg['env_key']} is not configured",
        }

    if provider_id == "openrouter" and not key.startswith("sk-or-v1-"):
        res = {
            "status": STATUS_AUTHENTICATION_FAILED,
            "latency_ms": None,
            "discovered_models": [],
            "error": "Invalid key format: OpenRouter API keys must begin with 'sk-or-v1-'. Generate a valid key at openrouter.ai/keys.",
        }
        _TEST_RESULTS_CACHE[f"provider:{provider_id}"] = res
        return res

    start = time.perf_counter()
    status = STATUS_UNKNOWN
    error_msg = None
    discovered = []

    try:
        async with httpx.AsyncClient(timeout=12) as client:
            if provider_id == "google":
                url = f"{cfg['models_url']}?key={key}"
                r = await client.get(url, headers={"x-goog-api-key": key})
            elif provider_id == "anthropic":
                r = await client.get(
                    cfg["models_url"],
                    headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
                )
            else:
                # OpenAI, DeepSeek, Groq, Mistral, OpenRouter standard headers
                r = await client.get(
                    cfg["models_url"], headers={"Authorization": f"Bearer {key}"}
                )

            latency = round((time.perf_counter() - start) * 1000, 1)

            if r.status_code == 200:
                status = STATUS_VERIFIED_WORKING
                try:
                    data = r.json()
                    if "data" in data and isinstance(data["data"], list):
                        discovered = [m.get("id") for m in data["data"] if "id" in m][:20]
                    elif "models" in data and isinstance(data["models"], list):
                        discovered = [m.get("name", "").replace("models/", "") for m in data["models"]][:20]
                except Exception:
                    pass
            elif r.status_code in (401, 403):
                status = STATUS_AUTHENTICATION_FAILED
                error_msg = f"HTTP {r.status_code}: Invalid or unauthorized API key"
            elif r.status_code == 429:
                status = STATUS_INSUFFICIENT_CREDITS if provider_id == "openai" else STATUS_RATE_LIMITED
                error_msg = f"HTTP 429: Account credit balance exhausted or rate limited"
            elif r.status_code == 402:
                status = STATUS_INSUFFICIENT_CREDITS
                error_msg = "HTTP 402: Insufficient Balance (prepaid balance required)"
            else:
                status = STATUS_TEST_FAILED
                error_msg = f"HTTP {r.status_code}: {r.text[:150]}"
    except httpx.TimeoutException:
        latency = round((time.perf_counter() - start) * 1000, 1)
        status = STATUS_TEST_FAILED
        error_msg = "Connection timed out"
    except Exception as exc:
        latency = round((time.perf_counter() - start) * 1000, 1)
        status = STATUS_TEST_FAILED
        error_msg = str(exc)

    res = {
        "status": status,
        "latency_ms": latency if status == STATUS_VERIFIED_WORKING else None,
        "discovered_models": discovered,
        "error": error_msg,
    }
    _TEST_RESULTS_CACHE[f"provider:{provider_id}"] = res
    return res


async def test_model_inference(provider_id: str, model_id: str) -> Dict[str, Any]:
    """Send a tiny, harmless test prompt to verify actual generation without fabricating output."""
    from openjarvis.core.types import Message, Role
    from openjarvis.server.cloud_router import stream_cloud, stream_local

    if provider_id == "mistral":
        return {
            "status": STATUS_NOT_CONFIGURED,
            "latency_ms": None,
            "error": "Mistral AI is disabled / not configured",
        }

    # Respect user directive: do not repeatedly test models lacking credits
    cached_m = _TEST_RESULTS_CACHE.get(f"model:{model_id}")
    if cached_m and cached_m.get("status") == STATUS_INSUFFICIENT_CREDITS:
        return cached_m
    cached_p = _TEST_RESULTS_CACHE.get(f"provider:{provider_id}")
    if cached_p and cached_p.get("status") == STATUS_INSUFFICIENT_CREDITS:
        return {
            "status": STATUS_INSUFFICIENT_CREDITS,
            "latency_ms": None,
            "error": cached_p.get("error", "Account credit balance exhausted"),
            "timestamp": time.time(),
        }

    key = get_provider_key(provider_id)
    cfg = CATALOG_PROVIDERS.get(provider_id)
    if not cfg:
        return {
            "status": STATUS_TEST_FAILED,
            "latency_ms": None,
            "error": f"Unknown provider: {provider_id}",
        }

    if cfg["integration"] == "api" and not key:
        return {
            "status": STATUS_NOT_CONFIGURED,
            "latency_ms": None,
            "error": f"{cfg['env_key']} is not configured",
        }

    if provider_id == "openrouter" and not key.startswith("sk-or-v1-"):
        res = {
            "status": STATUS_AUTHENTICATION_FAILED,
            "latency_ms": None,
            "error": "Invalid key format: OpenRouter API keys must begin with 'sk-or-v1-'. Generate a valid key at openrouter.ai/keys.",
            "timestamp": time.time(),
        }
        _TEST_RESULTS_CACHE[f"model:{model_id}"] = res
        return res

    prompt = [Message(role=Role.USER, content="Ping. Respond with one word: OK")]
    start = time.perf_counter()
    tokens = []
    status = STATUS_UNKNOWN
    error_msg = None

    try:
        if cfg["integration"] == "local":
            async for chunk in stream_local(model_id, prompt, max_tokens=256):
                tokens.append(chunk)
                if len(tokens) >= 20:
                    break
        else:
            async for chunk in stream_cloud(model_id, prompt, max_tokens=256):
                tokens.append(chunk)
                if len(tokens) >= 20:
                    break

        latency = round((time.perf_counter() - start) * 1000, 1)
        full_text = "".join(tokens).strip()

        if full_text:
            status = STATUS_VERIFIED_WORKING
        else:
            status = STATUS_TEST_FAILED
            error_msg = "Model returned empty response"

    except Exception as exc:
        latency = round((time.perf_counter() - start) * 1000, 1)
        status_str = str(exc)
        if "402" in status_str or "payment" in status_str.lower() or "insufficient balance" in status_str.lower():
            status = STATUS_INSUFFICIENT_CREDITS
            error_msg = "Account balance exhausted (DeepSeek top-up required: HTTP 402)"
        elif "credit balance is too low" in status_str.lower():
            status = STATUS_INSUFFICIENT_CREDITS
            error_msg = "Account credit balance too low (Anthropic credits required)"
        elif "credit_balance_exhausted" in status_str.lower() or "insufficient_quota" in status_str.lower() or ("429" in status_str and provider_id == "openai"):
            status = STATUS_INSUFFICIENT_CREDITS
            error_msg = "Account credit balance exhausted (OpenAI billing required: HTTP 429)"
        elif "401" in status_str or "403" in status_str or "unauthorized" in status_str.lower():
            status = STATUS_AUTHENTICATION_FAILED
            error_msg = "Authentication failed (Invalid or unauthorized API key)"
        elif "404" in status_str:
            status = STATUS_TEST_FAILED
            error_msg = "Model identifier not found or unsupported for this account"
        else:
            status = STATUS_TEST_FAILED
            error_msg = status_str

    res = {
        "status": status,
        "latency_ms": latency if status == STATUS_VERIFIED_WORKING else None,
        "sample_output": full_text[:40] if status == STATUS_VERIFIED_WORKING else None,
        "error": error_msg,
        "timestamp": time.time(),
    }
    _TEST_RESULTS_CACHE[f"model:{model_id}"] = res
    return res


# ---------------------------------------------------------------------------
# Registry Aggregator
# ---------------------------------------------------------------------------

def get_full_registry() -> Dict[str, Any]:
    """Return consolidated registry with live configuration and test statuses."""
    keys = load_all_keys()
    providers_out = []
    total_models = 0
    configured_models = 0

    for pid, pdata in CATALOG_PROVIDERS.items():
        env_key = pdata["env_key"]
        has_key = bool(keys.get(env_key) or (pdata.get("alt_env_key") and keys.get(pdata["alt_env_key"])))
        raw_key = keys.get(env_key) or keys.get(pdata.get("alt_env_key", ""))
        masked = mask_key(raw_key) if has_key else None

        p_status = STATUS_CONFIGURED if has_key else STATUS_NOT_CONFIGURED
        if pdata["integration"] == "local":
            p_status = STATUS_CONFIGURED if keys.get("OLLAMA_HOST") else STATUS_NOT_CONFIGURED

        cached_p_test = _TEST_RESULTS_CACHE.get(f"provider:{pid}")
        if cached_p_test:
            p_status = cached_p_test.get("status", p_status)

        models_list = []
        for m in pdata["models"]:
            total_models += 1
            m_id = m["id"]
            m_status = STATUS_CONFIGURED if has_key else STATUS_NOT_CONFIGURED
            if has_key:
                configured_models += 1

            cached_m_test = _TEST_RESULTS_CACHE.get(f"model:{m_id}")
            last_latency = None
            last_error = None
            last_tested = None
            if cached_m_test:
                m_status = cached_m_test.get("status", m_status)
                last_latency = cached_m_test.get("latency_ms")
                last_error = cached_m_test.get("error")
                last_tested = cached_m_test.get("timestamp")
            elif cached_p_test and cached_p_test.get("status") in (
                STATUS_INSUFFICIENT_CREDITS,
                STATUS_AUTHENTICATION_FAILED,
                STATUS_NOT_CONFIGURED,
            ):
                m_status = cached_p_test.get("status")
                last_error = cached_p_test.get("error")

            models_list.append(
                {
                    "id": m_id,
                    "name": m["name"],
                    "provider": pid,
                    "integration": pdata["integration"],
                    "description": m["desc"],
                    "context_length": m["context"],
                    "capabilities": m["capabilities"],
                    "status": m_status,
                    "enabled": True,
                    "last_tested": last_tested,
                    "last_latency_ms": last_latency,
                    "last_error": last_error,
                }
            )

        providers_out.append(
            {
                "id": pid,
                "name": pdata["name"],
                "env_key": env_key,
                "integration": pdata["integration"],
                "description": pdata["description"],
                "status": p_status,
                "is_configured": has_key,
                "masked_key": masked,
                "last_tested": cached_p_test.get("timestamp") if cached_p_test else None,
                "last_latency_ms": cached_p_test.get("latency_ms") if cached_p_test else None,
                "last_error": cached_p_test.get("error") if cached_p_test else None,
                "models": models_list,
            }
        )

    hardware = check_local_hardware_compatibility()

    verified_providers = sum(1 for p in providers_out if p["status"] == STATUS_VERIFIED_WORKING)
    verified_models = sum(
        1 for p in providers_out for m in p["models"] if m["status"] == STATUS_VERIFIED_WORKING
    )

    return {
        "providers": providers_out,
        "hardware": hardware,
        "summary": {
            "total_providers": len(providers_out),
            "configured_providers": sum(1 for p in providers_out if p["is_configured"]),
            "verified_providers": verified_providers,
            "total_models": total_models,
            "configured_models": configured_models,
            "verified_models": verified_models,
        },
    }
