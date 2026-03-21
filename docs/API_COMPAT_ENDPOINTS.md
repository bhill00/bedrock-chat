# Adding OpenAI/Anthropic-Compatible API Endpoints to Bedrock Chat

**Author:** Brad Hill (Bren School, UCSB)
**Date:** March 2026
**For:** Yaheya / LLM Sandbox Architecture Team
**Reference implementation:** [bhill00/bedrock-chat](https://github.com/bhill00/bedrock-chat/tree/v3) (fork of aws-samples/bedrock-chat with working endpoints)

---

## Summary

This document describes how to add OpenAI and Anthropic compatible API endpoints to the existing Bedrock Chat stack. The endpoints extend the **Published Bot API** infrastructure so that every published bot automatically gains standard API endpoints alongside the existing `/conversation` endpoint. Same API key, same bot, same compliance controls — different payload format.

The implementation was built and tested on a personal AWS deployment of the upstream [aws-samples/bedrock-chat](https://github.com/aws-samples/bedrock-chat) v3. The UCSB LLM Sandbox is a modified fork with NIST 800-171 compliance enhancements. The code examples and file references below are based on the upstream project — function names and file paths may differ in the Sandbox fork, but the architectural approach applies directly.

**Total new code:** ~500 lines across 3 files, plus 5 lines modified in `main.py`.

---

## Why

The current Bot API uses a custom async request/response format (POST → SQS → poll GET) that isn't compatible with the OpenAI API spec that most AI tools expect. This creates friction:

| Problem | Impact |
|---|---|
| Every tool needs a translation proxy | Extra latency, maintenance burden, point of failure |
| No real streaming | Proxy fakes SSE by polling; clients see nothing until full response |
| No per-request inference parameters | temperature/max_tokens locked to bot settings; different use cases need separate bots |
| Inaccurate token tracking | Proxy can't report real usage since Bot API doesn't expose it |

Native OpenAI and Anthropic compatible endpoints solve all of these while preserving the existing compliance controls.

---

## Architecture: Bot as Compliance Boundary

The key design decision: the OpenAI endpoint is **an extension of the Published Bot API**, not a separate system. This means:

```
Published Bot (existing)
├── POST /conversation          ← existing async Bot API (unchanged)
├── GET  /conversation/{id}     ← existing poll endpoint (unchanged)
├── POST /v1/chat/completions   ← NEW: OpenAI-compatible (sync + streaming)
├── POST /v1/messages           ← NEW: Anthropic-compatible (sync + streaming)
└── GET  /v1/models             ← NEW: list available models
```

Same API key authenticates all endpoints. The bot's configuration provides:

| Bot Setting | Endpoint Behavior |
|---|---|
| **Instruction (system prompt)** | Prepended to every request — always present, not overridable |
| **Guardrails** | Always applied — not overridable by the client |
| **Generation params** (temp, max_tokens, top_p) | Used as defaults — client can override per-request |
| **Model** | Client specifies per-request (with alias support) |

**This preserves the compliance boundary.** Whatever NIST controls are enforced through the bot configuration (guardrails, system prompts, model restrictions, audit logging) carry over to both endpoints automatically. The bot owner sets the safety rails; the API consumer gets the convenience of standard API formats with per-request tuning, but cannot bypass the guardrails.

---

## What Changed (3 new files, 1 modified)

### 1. `backend/app/routes/schemas/openai_compat.py` (new, ~50 lines)

Pydantic models for both API formats:

```python
class OpenAIMessage(BaseModel):
    role: str
    content: Union[str, list, None] = None

class OpenAIChatCompletionRequest(BaseModel):
    model: str
    messages: list[OpenAIMessage]
    temperature: Optional[float] = None    # Overrides bot default
    max_tokens: Optional[int] = None       # Overrides bot default
    top_p: Optional[float] = None          # Overrides bot default
    top_k: Optional[int] = None            # Anthropic extension (not in OpenAI spec)
    stream: Optional[bool] = False
    stop: Optional[Union[str, list[str]]] = None
    # Accept but ignore these OpenAI-specific fields for compatibility
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    n: Optional[int] = None
    user: Optional[str] = None
```

Note: `top_k` is an Anthropic/Bedrock extension not present in the OpenAI spec. OpenAI only exposes `top_p`. Including it here means Anthropic SDK clients can use it, while OpenAI SDK clients simply ignore it.

```python
# Anthropic /v1/messages schema
class AnthropicMessagesRequest(BaseModel):
    model: str
    messages: list[AnthropicMessage]
    system: Optional[str] = None           # Separate from messages (Anthropic convention)
    max_tokens: int = 4096                 # Required in Anthropic API
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None            # Native Anthropic parameter
    stream: Optional[bool] = False
    stop_sequences: Optional[list[str]] = None
    metadata: Optional[dict] = None
```

### 2. `backend/app/routes/openai_compat.py` (new, ~500 lines)

The main endpoint. Key components:

**Model alias map** — Lets clients use familiar names:
```python
MODEL_ALIASES = {
    "gpt-4": "claude-v4.5-sonnet",          # OpenAI drop-in
    "gpt-4o": "claude-v4.5-sonnet",
    "gpt-3.5-turbo": "claude-v4.5-haiku",
    "claude-sonnet-4-5": "claude-v4.5-sonnet",  # Anthropic API names
    "claude-opus-4-5": "claude-v4.5-opus",
    # ... also accepts native bedrock-chat names directly
}
```

**Bot context loader** — In Published API mode, loads the bot's instruction, generation_params, and guardrails:
```python
def _get_bot_context():
    if not PUBLISHED_API_ID:
        return None, None, None
    user = User.from_published_api_id(PUBLISHED_API_ID)
    bot_id = user.id.split("#")[1]
    _, bot = fetch_bot(user, bot_id)
    return bot.instruction, bot.generation_params, bot.bedrock_guardrails
```

**Parameter merging** — Bot defaults + per-request overrides:
```python
def build_generation_params(req, bot_params=None):
    if bot_params:
        return GenerationParamsModel(
            max_tokens=req.max_tokens if req.max_tokens is not None else bot_params.max_tokens,
            temperature=req.temperature if req.temperature is not None else bot_params.temperature,
            # ... etc
        )
```

**Message translation** — OpenAI `messages[]` → Bedrock `SimpleMessageModel`:
- System messages → extracted as `instructions` (prepended after bot instruction)
- User/assistant messages → converted to `TextContentModel` list
- Multimodal content blocks → text blocks extracted (image support can be added)

**Non-streaming path** — Calls `compose_args_for_converse_api()` → `client.converse()` → formats response as OpenAI JSON.

**Streaming path** — Calls `compose_args_for_converse_api()` → `client.converse_stream()` → yields OpenAI SSE chunks from Bedrock Converse API events:
- `contentBlockDelta` with `text` → `{"delta": {"content": "token"}}`
- `metadata` → captures `inputTokens`/`outputTokens`
- `messageStop` → `{"finish_reason": "stop", "usage": {...}}`

**Anthropic endpoint** (`/v1/messages`) — Same file, same pattern. Key differences from OpenAI:

| | OpenAI `/v1/chat/completions` | Anthropic `/v1/messages` |
|---|---|---|
| System prompt | In `messages[]` with `role: "system"` | Separate `system` field |
| `max_tokens` | Optional (defaults to 4096) | Required |
| `top_k` | Extension (not in OpenAI spec) | Native parameter |
| `stop` param | `stop` (string or array) | `stop_sequences` (array) |
| Response body | `choices[0].message.content` (string) | `content[]` (array of `{type, text}` blocks) |
| Streaming format | `data: {"choices":[{"delta":{"content":"..."}}]}` | `event: content_block_delta\ndata: {"delta":{"text":"..."}}` |
| Usage field | `usage.prompt_tokens` / `completion_tokens` | `usage.input_tokens` / `output_tokens` |

The Anthropic endpoint enables tools that use the Anthropic SDK (Claude Code, some LangChain configs) to connect directly.

### 3. `backend/app/main.py` (modified, 5 lines)

```python
# Add import
from app.routes.openai_compat import router as openai_compat_router

# Register router (after the conditional block, so it's available in both modes)
app.include_router(openai_compat_router)
```

The router is registered **unconditionally** — available in both the main app (Cognito JWT auth) and Published API mode (API key auth). The auth middleware handles both cases transparently.

---

## What the Endpoint Reuses (no changes needed)

| Component | File | What it does |
|---|---|---|
| `compose_args_for_converse_api()` | `bedrock.py:948` | Builds Bedrock Converse API payload |
| `get_model_id()` | `bedrock.py:1162` | Maps model names → Bedrock IDs with cross-region/global inference |
| `generation_params_to_converse_configuration()` | `bedrock.py:803` | Handles per-model-family parameter constraints |
| Auth middleware | `main.py:109` | JWT (main app) or API key (published API) — already in place |
| `User.from_published_api_id()` | `user.py:47` | Creates user context from published bot ID |
| `fetch_bot()` | `usecases/bot.py:349` | Loads bot config (instruction, params, guardrails) from DynamoDB |
| `get_bedrock_runtime_client()` | `utils.py:45` | Boto3 Bedrock client |
| Request logging middleware | `main.py:138` | Audit trail — already captures all requests |

Both endpoints introduce **zero new infrastructure** — no new DynamoDB tables, no new IAM roles, no new Lambda functions. They're additional routes on the existing FastAPI app.

---

## Deployment Notes

### Published API Lambda Environment Variables

The Published Bot API Lambda needs these env vars for cross-region inference to work (the main stack has them, but the published API CDK construct doesn't set them by default):

```
ENABLE_BEDROCK_CROSS_REGION_INFERENCE=true
ENABLE_BEDROCK_GLOBAL_INFERENCE=true
```

Without these, models resolve to bare IDs (e.g., `anthropic.claude-sonnet-4-5-20250929-v1:0`) which fail on accounts that require inference profiles. The fix is to add these to the published API CDK construct's Lambda environment.

**File to modify:** Look for the CDK construct that creates the Published API Lambda (likely in `cdk/lib/constructs/api-publish-codebuild.ts` or similar) and add these env vars.

### Frontend Build Fix (upstream bug)

The upstream project has a broken frontend build due to an `xstate` v4/v5 conflict. `@aws-amplify/ui-react` needs xstate v4, but the app uses xstate v5. Rollup resolves all bare `xstate` imports to v5, breaking Amplify.

**Fix:** Install xstate v4 as an npm alias (`"xstate-v4": "npm:xstate@4.38.3"`) and add a Vite plugin that redirects `xstate` imports from `@aws-amplify` packages to `xstate-v4`. See commits `12138bd` through `29794f7` in the fork. This may or may not affect the Sandbox fork depending on its Amplify version.

---

## Testing

Once deployed, test with the Published Bot API key:

```bash
# Health check
curl https://<api-url>/api/health -H "x-api-key: <key>"

# List models
curl https://<api-url>/api/v1/models -H "x-api-key: <key>"

# Non-streaming chat
curl https://<api-url>/api/v1/chat/completions \
  -H "x-api-key: <key>" \
  -H "Content-Type: application/json" \
  -d '{"model":"claude-v4-sonnet","messages":[{"role":"user","content":"Hello!"}]}'

# Streaming chat with per-request temperature
curl https://<api-url>/api/v1/chat/completions \
  -H "x-api-key: <key>" \
  -H "Content-Type: application/json" \
  -d '{"model":"claude-v4-sonnet","messages":[{"role":"user","content":"Hello!"}],"stream":true,"temperature":0.2}'
```

### Anthropic format (`/v1/messages`)

```bash
# Non-streaming
curl https://<api-url>/api/v1/messages \
  -H "x-api-key: <key>" \
  -H "Content-Type: application/json" \
  -d '{"model":"claude-v4-sonnet","system":"You are helpful.","messages":[{"role":"user","content":"Hello!"}],"max_tokens":1024}'

# Streaming
curl https://<api-url>/api/v1/messages \
  -H "x-api-key: <key>" \
  -H "Content-Type: application/json" \
  -d '{"model":"claude-v4-sonnet","messages":[{"role":"user","content":"Hello!"}],"max_tokens":1024,"stream":true}'
```

### Client SDK examples

```python
# OpenAI SDK
from openai import OpenAI
client = OpenAI(base_url="https://<api-url>/api/v1", api_key="<key>")
response = client.chat.completions.create(
    model="claude-v4-sonnet",
    messages=[{"role": "user", "content": "Hello!"}],
    temperature=0.2,
)

# Anthropic SDK
from anthropic import Anthropic
client = Anthropic(
    base_url="https://<api-url>/api/v1",
    api_key="<key>",
)
message = client.messages.create(
    model="claude-v4-sonnet",
    max_tokens=1024,
    system="You are a helpful assistant.",
    messages=[{"role": "user", "content": "Hello!"}],
)

# LangChain (OpenAI)
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(
    base_url="https://<api-url>/api/v1",
    api_key="<key>",
    model="claude-v4-sonnet",
)

# LangChain (Anthropic)
from langchain_anthropic import ChatAnthropic
llm = ChatAnthropic(
    base_url="https://<api-url>/api/v1",
    api_key="<key>",
    model="claude-v4-sonnet",
)
```

---

## What This Does NOT Include

- **Structured function calling** (`tools` array) — Would require mapping OpenAI's tool_calls protocol to Bedrock's tool use format. Prompt-engineered tool use works fine.
- **Embeddings** (`/v1/embeddings`) — Bedrock has embedding models but they use a different API. Could be added separately.
- **Conversation persistence** — Both endpoints are stateless by design. Each request is independent. The existing `/conversation` endpoint still handles persistent conversations.

---

## Compliance Preservation Summary

| Control | Existing Bot API | OpenAI + Anthropic Endpoints | Notes |
|---|---|---|---|
| API key auth | Yes | Yes | Same key, same mechanism |
| Bot-level guardrails | Yes | Yes | Applied unconditionally |
| Bot instruction/system prompt | Yes | Yes | Prepended to every request |
| Request audit logging | Yes | Yes | Same middleware |
| Token usage tracking | Via bot dashboard | Via response `usage` field + CloudWatch | From Bedrock Converse API metrics |
| Data isolation (VPC) | Yes | Yes | Same Lambda, same network |
| Encryption in transit | TLS | TLS | Same API Gateway |
| Per-bot access control | Yes | Yes | Same published API isolation |
| Conversation persistence | Yes (DynamoDB) | No (stateless) | Privacy advantage for some use cases |

The stateless nature of both endpoints is actually a compliance feature — no conversation history is stored server-side, reducing the data retention surface.

---

## Files in This Fork

All changes are on the `v3` branch of [bhill00/bedrock-chat](https://github.com/bhill00/bedrock-chat/tree/v3):

| File | Change | Purpose |
|---|---|---|
| `backend/app/routes/openai_compat.py` | New | OpenAI + Anthropic endpoints, streaming, bot integration |
| `backend/app/routes/schemas/openai_compat.py` | New | Pydantic models for both API formats |
| `backend/app/main.py` | +5 lines | Router registration |
| `frontend/package.json` | Modified | xstate-v4 npm alias (build fix) |
| `frontend/vite.config.ts` | Modified | Vite plugin for xstate resolution (build fix) |
| `frontend/package-lock.json` | Modified | Lockfile update |

To see the exact diff: `git diff origin/v3..v3`

---

## Verified Test Results

Tested on a live deployment (personal AWS account, March 2026):

| Test | Endpoint | Result |
|---|---|---|
| Non-streaming, OpenAI format | `/v1/chat/completions` | "Hello there, friend!" — 22 tokens |
| Streaming SSE, OpenAI format | `/v1/chat/completions` | Real token-by-token, pirate system prompt works |
| Non-streaming, Anthropic format | `/v1/messages` | "Ahoy there, matey!" — 31 tokens, `content[]` blocks |
| Streaming SSE, Anthropic format | `/v1/messages` | Typed events (message_start, content_block_delta, etc.) |
| Per-request temperature | Both | 0.2 and 0.7 tested, override bot defaults |
| System prompt (OpenAI) | `/v1/chat/completions` | Via `messages[]` with `role: "system"` |
| System prompt (Anthropic) | `/v1/messages` | Via separate `system` field |
| Model aliases | Both | `gpt-4` → `claude-v4.5-sonnet`, Anthropic names work |
| Token usage (non-streaming) | Both | Accurate from Bedrock Converse API metrics |
| Bot guardrails | Both | Applied from bot config, not overridable |
| Bot instruction | Both | Prepended to system messages |
| API key auth | Both | Same key as existing `/conversation` endpoint |

### Standalone proxy (bedrock-api-proxy)

The lightweight Lambda proxy ([bhill00/bedrock-api-proxy](https://github.com/bhill00/bedrock-api-proxy)) was the original proof of concept. It provides the same OpenAI + Anthropic endpoints without the Bedrock Chat infrastructure (no bots, no Cognito, no DynamoDB). Useful for direct Bedrock access with just an API key.
