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

# Ollama
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_TIMEOUT_S = float(os.getenv("OLLAMA_TIMEOUT_S", 8.0))

# A2A clients health check
AGENT_TIMEOUT_S = float(os.getenv("AGENT_TIMEOUT_S", 12.0))
AGENT_RETRIES = int(os.getenv("AGENT_RETRIES", 1))
CIRCUIT_FAILURE_THRESHOLD = int(os.getenv("CIRCUIT_FAILURE_THRESHOLD", 3))
CIRCUIT_RECOVERY_SECONDS = int(os.getenv("CIRCUIT_RECOVERY_SECONDS", 20))
