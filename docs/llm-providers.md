# LLM Provider Configuration

MicroAgent has one in-process LLM call: `extract_from_text`, which parses a free-form document (e.g., a methods section or README) to pre-fill fields when creating a `project.yaml`. You can point this at any LLM provider, or disable it entirely.

If no provider is configured the function falls back to keyword heuristics — no API key is required to use MicroAgent.

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MICROAGENT_LLM_PROVIDER` | auto-detect | `anthropic`, `openai`, `mistral`, `ollama`, or `none` |
| `MICROAGENT_LLM_MODEL` | provider default | Override the model name |
| `MICROAGENT_LLM_BASE_URL` | provider default | Override the API base URL |

**Auto-detection order** (when `MICROAGENT_LLM_PROVIDER` is not set):
1. `ANTHROPIC_API_KEY` → `anthropic`
2. `OPENAI_API_KEY` → `openai`
3. `MISTRAL_API_KEY` → `mistral`
4. No key found → `none` (keyword fallback)

---

## Supported Providers

### Anthropic (Claude)

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
# MICROAGENT_LLM_PROVIDER=anthropic is auto-detected
```

Default model: `claude-sonnet-4-6`

### OpenAI

```bash
export OPENAI_API_KEY="sk-..."
# MICROAGENT_LLM_PROVIDER=openai is auto-detected
```

Default model: `gpt-4o-mini`

### Mistral (EU-hosted)

```bash
export MISTRAL_API_KEY="..."
# MICROAGENT_LLM_PROVIDER=mistral is auto-detected
```

Default model: `mistral-small-latest`. API endpoint: `https://api.mistral.ai/v1`.

### Ollama (local, no API key)

```bash
export MICROAGENT_LLM_PROVIDER=ollama
export MICROAGENT_LLM_MODEL=llama3   # or any model you have pulled
```

Ollama must be running locally (`ollama serve`). Default endpoint: `http://localhost:11434/v1`.

### Any OpenAI-compatible endpoint

`openai`, `mistral`, and `ollama` all use the OpenAI-compatible API. You can point MicroAgent at vLLM, LM Studio, OpenRouter, or any other compatible server:

```bash
export MICROAGENT_LLM_PROVIDER=openai
export MICROAGENT_LLM_BASE_URL=http://localhost:8000/v1
export MICROAGENT_LLM_MODEL=meta-llama/Llama-3-8B-Instruct
export OPENAI_API_KEY=no-key   # required by the SDK but unused by local servers
```

### Disabled (keyword fallback only)

```bash
export MICROAGENT_LLM_PROVIDER=none
```

Field extraction will use regex/keyword heuristics. All other MicroAgent functionality is unaffected.

---

## SDK Dependencies

No LLM SDK is a hard dependency. MicroAgent installs neither `anthropic` nor `openai` by default. The relevant SDK must be available at runtime:

```bash
pip install anthropic   # for Anthropic
pip install openai      # for OpenAI, Mistral, Ollama, or any OpenAI-compatible endpoint
```

If the required SDK is not installed, MicroAgent silently falls back to keyword extraction.

---

## Note on the MCP Server

The MCP server itself makes **no LLM calls**. All AI reasoning happens in the MCP *client* (Claude, LibreChat, Cursor, etc.). The `MICROAGENT_LLM_PROVIDER` setting only affects the `microagent init --from-doc` workflow. See [mcp-integration.md](mcp-integration.md) for client-side provider options.
