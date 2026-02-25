import os
from dotenv import load_dotenv

load_dotenv()

# App config (127.0.0.1 for localhost, 0.0.0.0 for whole network)
APP_NAME = os.getenv("APP_NAME", "A2A orchestrator")
APP_VERSION = os.getenv("APP_VERSION", "0.1.0")
APP_HOST = os.getenv("APP_HOST", "127.0.0.1")
APP_PORT = int(os.getenv("APP_PORT", 9301))

# Logging
DEBUG = bool(os.getenv("DEBUG", "False").lower() == "true")

# A2A registry
REGISTRY_URL = os.getenv("REGISTRY_URL", "http://localhost:9300")
REGISTRY_TIMEOUT_S = float(os.getenv("REGISTRY_TIMEOUT_S", 4.0))
REGISTRY_RETRIES = int(os.getenv("REGISTRY_RETRIES", 2))
SKILLS_CACHE_TTL_S = float(os.getenv("SKILLS_CACHE_TTL_S", 60.0))
SKILLS_CACHE_REFRESH_INTERVAL_S = float(os.getenv("SKILLS_CACHE_REFRESH_INTERVAL_S", 30.0))

# Ollama
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_TIMEOUT_S = float(os.getenv("OLLAMA_TIMEOUT_S", 8.0))
ROUTE_CONFIDENCE_THRESHOLD = float(os.getenv("ROUTE_CONFIDENCE_THRESHOLD", 0.6))

# A2A
AGENT_TIMEOUT_S = float(os.getenv("AGENT_TIMEOUT_S", 12.0))
AGENT_RETRIES = int(os.getenv("AGENT_RETRIES", 1))
CIRCUIT_FAILURE_THRESHOLD = int(os.getenv("CIRCUIT_FAILURE_THRESHOLD", 3))
CIRCUIT_RECOVERY_SECONDS = int(os.getenv("CIRCUIT_RECOVERY_SECONDS", 20))


def _ensure_positive(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")


def _ensure_non_negative(name: str, value: float) -> None:
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")


def _validate_runtime_config() -> None:
    _ensure_positive("APP_PORT", float(APP_PORT))
    _ensure_positive("REGISTRY_TIMEOUT_S", REGISTRY_TIMEOUT_S)
    _ensure_non_negative("REGISTRY_RETRIES", float(REGISTRY_RETRIES))
    _ensure_positive("SKILLS_CACHE_TTL_S", SKILLS_CACHE_TTL_S)
    _ensure_non_negative("SKILLS_CACHE_REFRESH_INTERVAL_S", SKILLS_CACHE_REFRESH_INTERVAL_S)
    _ensure_positive("OLLAMA_TIMEOUT_S", OLLAMA_TIMEOUT_S)
    _ensure_positive("AGENT_TIMEOUT_S", AGENT_TIMEOUT_S)
    _ensure_non_negative("AGENT_RETRIES", float(AGENT_RETRIES))
    _ensure_positive("CIRCUIT_FAILURE_THRESHOLD", float(CIRCUIT_FAILURE_THRESHOLD))
    _ensure_positive("CIRCUIT_RECOVERY_SECONDS", float(CIRCUIT_RECOVERY_SECONDS))
    if not 0.0 <= ROUTE_CONFIDENCE_THRESHOLD <= 1.0:
        raise ValueError(
            f"ROUTE_CONFIDENCE_THRESHOLD must be between 0 and 1, got {ROUTE_CONFIDENCE_THRESHOLD}"
        )


_validate_runtime_config()
