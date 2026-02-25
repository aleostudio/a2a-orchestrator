# A2A orchestrator — Agent-to-Agent routing gateway

A2A orchestrator is the entrypoint service for **multi-agent** systems,
providing a single HTTP API for clients, classifying user intent, discovering
the correct specialist via A2A registry, and delegating execution through
A2A protocol.

## Index

- [Prerequisites](#prerequisites)
- [Configuration](#configuration)
- [Run A2A orchestrator](#run-a2a-orchestrator)
- [Endpoints](#endpoints)
- [Testing](#testing)
- [Debug in VSCode](#debug-in-vscode)
- [License](#license)

---

## Prerequisites

- Python >= 3.12
- [uv](https://docs.astral.sh/uv/getting-started/installation) and [pip](https://pip.pypa.io/en/stable/installation) installed
- [A2A registry](https://github.com/aleostudio/a2a-registry) running (default: `http://localhost:9300`)
- Specialist agents running and registered in the registry
- Ollama running (default: `http://localhost:11434`)

[↑ index](#index)

---

## Configuration

Init **virtualenv** and install dependencies with:

```bash
uv venv
source .venv/bin/activate
uv sync
```

If you prefer, there is a `Makefile` that do the job for you (follow the instructions):

```bash
make setup
```

Now create your `.env` file by copying:

```bash
cp env.dist .env
```

[↑ index](#index)

---

## Run A2A orchestrator

To run **A2A orchestrator**:

```bash
uv run python -m app.main
```

or through the shortcut:

```bash
make dev
```

Default host/port: `0.0.0.0:9301`.

[↑ index](#index)

---

## Endpoints

### `GET /health` - Health check

Returns the orchestrator status and circuit breaker snapshots.

```bash
curl http://localhost:9301/health
```

**Response**: `200 OK`

```json
{
  "status": "ok",
  "registry_url": "http://localhost:9300",
  "registry_breaker": {
    "failures": 0,
    "open": false
  },
  "agent_breakers": {}
}
```

---

### `POST /inference` - Route user request

Classifies the user message, discovers an agent by skill in the registry, and delegates via A2A.

```bash
curl -XPOST "http://localhost:9301/inference" \
  -H "Content-Type: application/json" \
  -d '{"message":"What is 125 * 37 + 89?"}'
```

```bash
curl -XPOST "http://localhost:9301/inference" \
  -H "Content-Type: application/json" \
  -d '{"message":"Tell me about Python"}'
```

**Request body**:

| Field | Type | Required | Description |
| !--- | !--- | !--- | !--- |
| `message` | string | yes | User message to route |

**Response**: `200 OK`

```json
{
  "response": "125 * 37 + 89 = 4714",
  "agent": "Math Specialist",
  "request_id": "8eb2f5e8-9d36-4e65-bf16-ecb7dafe6ef7",
  "status": "ok",
  "route_source": "rule",
  "error_code": null
}
```

If no matching agent is available, response still returns `200` with fallback text and:

- `status = "fallback"`
- `error_code = "no_agent_found"` (or other execution fallback)

[↑ index](#index)

---

## Testing

Ensure you have `pytest` installed, otherwise:

```bash
uv pip install pytest
```

Then, launch tests with:

```bash
pytest tests/
```

or through shortcut:

```bash
make test
```

[↑ index](#index)

---

## Debug in VSCode

To debug your Python microservice you need to:

- Install **VSCode**
- Ensure you have **Python extension** installed
- Ensure you have selected the **right interpreter with virtualenv** on VSCode
- Click on **Run and Debug** menu and **create a launch.json file**
- From dropdown, select **Python debugger**
- Change the `.vscode/launch.json` created in the project root with this (customizing host and port if changed):

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "A2A orchestrator debug",
            "type": "debugpy",
            "request": "launch",
            "module": "uvicorn",
            "args": [
                "app.main:app",
                "--host", "0.0.0.0",
                "--port", "9301"
            ],
            "envFile": "${workspaceFolder}/.env",
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}",
            "justMyCode": true
        }
    ]
}
```

- Put some breakpoint in the code, then press the **green play button**
- Call the API to debug

[↑ index](#index)

---

## License

This project is licensed under the MIT License.

[↑ index](#index)

---

Made with love by Alessandro Orru
