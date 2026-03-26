# Adding OpenAI/Anthropic-Compatible API Endpoints to Bedrock Chat

**Author:** Brad Hill (Bren School, UCSB)
**Date:** March 2026
**For:** Yaheya / LLM Sandbox Architecture Team
**Reference implementation:** [bhill00/bedrock-chat](https://github.com/bhill00/bedrock-chat/tree/v3) (fork of aws-samples/bedrock-chat with working endpoints)

---

## Summary

This document describes how to add OpenAI and Anthropic compatible API endpoints to the existing Bedrock Chat stack. The implementation is split into two independent parts:

| Part | What it does | Required for |
|---|---|---|
| **Part 1: Payload Compatibility** | Adds `/v1/chat/completions`, `/v1/messages`, `/v1/models`, `/v1/health` routes | Anthropic SDK, LangChain Anthropic, any tool using `x-api-key` |
| **Part 2: Auth Compatibility** | Adds `Authorization: Bearer` header support alongside `x-api-key` | OpenAI SDK, LangChain OpenAI, and most generic OpenAI-compatible tools |

**Part 1 alone is sufficient if you only need Anthropic SDK compatibility.** The Anthropic SDK sends `x-api-key` by default, which API Gateway already accepts. Part 2 is only needed if you want OpenAI SDK clients to work without custom header configuration.

The endpoints extend the **Published Bot API** infrastructure so that every published bot automatically gains standard API endpoints alongside the existing `/conversation` endpoint. Same API key, same bot, same compliance controls — different payload format.

The implementation was built and tested on a personal AWS deployment of the upstream [aws-samples/bedrock-chat](https://github.com/aws-samples/bedrock-chat) v3. The UCSB LLM Sandbox is a modified fork with NIST 800-171 compliance enhancements. The code examples and file references below are based on the upstream project — function names and file paths may differ in the Sandbox fork, but the architectural approach applies directly.

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
├── GET  /v1/health             ← NEW: health check
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

**This preserves the compliance boundary.** Whatever NIST controls are enforced through the bot configuration (guardrails, system prompts, model restrictions, audit logging) carry over to both endpoints automatically.

---

## Which clients work with which parts

| Client / Tool | Auth header sent | Part 1 only | Part 1 + Part 2 |
|---|---|---|---|
| Anthropic SDK | `x-api-key` | ✅ Works | ✅ Works |
| LangChain `ChatAnthropic` | `x-api-key` | ✅ Works | ✅ Works |
| Claude Code | `x-api-key` | ✅ Works | ✅ Works |
| OpenAI SDK | `Authorization: Bearer` | ❌ Blocked by API GW | ✅ Works |
| LangChain `ChatOpenAI` | `Authorization: Bearer` | ❌ Blocked by API GW | ✅ Works |
| Most generic OpenAI-compat tools | `Authorization: Bearer` | ❌ Blocked by API GW | ✅ Works |
| OpenAI SDK with custom headers | `x-api-key` (overridden) | ✅ Works | ✅ Works |

---

## Part 1: Payload Compatibility

**Total new code:** ~500 lines across 3 files, plus 5 lines modified in `main.py`.

### 1a. `backend/app/routes/schemas/openai_compat.py` (new, ~50 lines)

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

### 1b. `backend/app/routes/openai_compat.py` (new, ~500 lines)

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

**Health endpoint** — Simple route under the `/v1` prefix so tools that probe `/v1/health` get a response:
```python
@router.get("/health")
def health():
    return {"status": "ok"}
```

The Anthropic endpoint enables tools that use the Anthropic SDK (Claude Code, some LangChain configs) to connect directly.

### 1c. `backend/app/main.py` (modified, 5 lines)

```python
# Add import
from app.routes.openai_compat import router as openai_compat_router

# Register router (after the conditional block, so it's available in both modes)
app.include_router(openai_compat_router)
```

The router is registered **unconditionally** — available in both the main app (Cognito JWT auth) and Published API mode (API key auth). The auth middleware handles both cases transparently.

### What Part 1 reuses (no changes needed)

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

Part 1 introduces **zero new infrastructure** — no new DynamoDB tables, no new IAM roles, no new Lambda functions.

---

## Part 2: Auth Compatibility (Optional — Bearer Token Support)

> **Only needed if you want OpenAI SDK clients and generic OpenAI-compatible tools to work without custom header configuration.**
>
> If your users only use the Anthropic SDK, Claude Code, or LangChain's Anthropic integration, skip this section.

### The problem

API Gateway's built-in key validation only accepts `x-api-key`. The OpenAI SDK and most OpenAI-compatible tools send `Authorization: Bearer <key>` by default. This causes API Gateway to reject requests before they reach the FastAPI app.

### The solution

Replace API Gateway's built-in key validation with a Lambda authorizer that accepts the API key from either header, validates it, and returns an IAM allow/deny policy. Per-key throttling and quotas are preserved via `usageIdentifierKey`.

### 2a. `cdk/lambda/api-key-authorizer/index.py` (new, ~70 lines)

```python
def handler(event, context):
    headers = {k.lower(): v for k, v in (event.get("headers") or {}).items()}

    # Accept key from either header
    api_key = None
    if headers.get("x-api-key"):
        api_key = headers["x-api-key"].strip()
    elif headers.get("authorization", "").lower().startswith("bearer "):
        api_key = headers["authorization"][7:].strip()

    if not api_key:
        raise Exception("Unauthorized")

    # Validate against API Gateway usage plan keys
    valid_keys = _load_keys()  # cached in-memory per Lambda container
    if api_key not in valid_keys:
        raise Exception("Unauthorized")

    return {
        "principalId": valid_keys[api_key],
        "policyDocument": {
            "Version": "2012-10-17",
            "Statement": [{"Action": "execute-api:Invoke", "Effect": "Allow", "Resource": wildcard_arn}],
        },
        "usageIdentifierKey": api_key,  # keeps per-key throttling/quotas working
    }
```

The authorizer **fails closed** — any error or missing key raises `Unauthorized`. The key cache is per Lambda container lifetime, avoiding an API Gateway list call on every request.

### 2b. `cdk/lib/api-publishment-stack.ts` (modified)

Add the authorizer Lambda and wire it into the API:

```typescript
import * as lambda from "aws-cdk-lib/aws-lambda";

// Lambda authorizer
const authorizerFn = new lambda.Function(this, "ApiKeyAuthorizer", {
  runtime: lambda.Runtime.PYTHON_3_12,
  handler: "index.handler",
  code: lambda.Code.fromAsset(
    path.join(__dirname, "../lambda/api-key-authorizer")
  ),
  timeout: cdk.Duration.seconds(10),
  role: new iam.Role(this, "AuthorizerRole", {
    assumedBy: new iam.ServicePrincipal("lambda.amazonaws.com"),
    managedPolicies: [
      iam.ManagedPolicy.fromAwsManagedPolicyName("service-role/AWSLambdaBasicExecutionRole"),
    ],
    inlinePolicies: {
      ReadApiKeys: new iam.PolicyDocument({
        statements: [
          new iam.PolicyStatement({
            actions: ["apigateway:GET"],
            resources: ["*"],
          }),
        ],
      }),
    },
  }),
});

const authorizer = new apigateway.RequestAuthorizer(this, "Authorizer", {
  handler: authorizerFn,
  // Use context as identity source so API GW always invokes the Lambda
  // regardless of which auth header is present
  identitySources: [apigateway.IdentitySource.context("httpMethod")],
  resultsCacheTtl: cdk.Duration.seconds(0), // no caching — key revocation is immediate
});

// Change the API to use authorizer instead of built-in key validation
const api = new apigateway.LambdaRestApi(this, "Api", {
  // ...existing props...
  defaultMethodOptions: {
    apiKeyRequired: false,        // turn off built-in key check
    authorizer: authorizer,
    authorizationType: apigateway.AuthorizationType.CUSTOM,
  },
});
```

### NIST considerations for Part 2

| Control | Impact | Mitigation |
|---|---|---|
| 3.1 Access Control | Low — boundary moves to Lambda | Fail-closed implementation, minimal IAM, code review |
| 3.3 Audit/Accountability | Low — TTL set to 0, no revocation lag | No caching means every request hits the authorizer |
| 3.5 Identification & Auth | Low-Medium — two accepted header formats | Document both in SSP; same underlying credential either way |
| 3.13 Comms Protection | None | TLS, VPC, network controls unchanged |

With `resultsCacheTtl: cdk.Duration.seconds(0)`, key revocation is immediate — no lag between revoking a key and it being rejected. The tradeoff is a small additional Lambda invocation per request, but the authorizer is lightweight.

The SSP should be updated to reflect that the API accepts the key via two header formats.

---

## Deployment Notes

### Published API Lambda Environment Variables

> **Note for UCSB:** Cross-region inference should remain `false` for NIST data residency compliance. These notes describe what's needed for non-NIST deployments.

For non-NIST deployments, the Published Bot API Lambda needs these env vars for cross-region inference to work (the main stack has them, but the published API CDK construct doesn't set them by default):

```
ENABLE_BEDROCK_CROSS_REGION_INFERENCE=true
ENABLE_BEDROCK_GLOBAL_INFERENCE=true
```

Without these, models resolve to bare IDs (e.g., `anthropic.claude-sonnet-4-5-20250929-v1:0`) which fail on accounts that require inference profiles. For UCSB, leave both `false` to ensure requests stay within the deployment region.

### Frontend Build Fix (upstream bug)

The upstream project has a broken frontend build due to an `xstate` v4/v5 conflict. `@aws-amplify/ui-react` needs xstate v4, but the app uses xstate v5. Rollup resolves all bare `xstate` imports to v5, breaking Amplify.

**Fix:** Install xstate v4 as an npm alias (`"xstate-v4": "npm:xstate@4.38.3"`) and add a Vite plugin that redirects `xstate` imports from `@aws-amplify` packages to `xstate-v4`. See commits `12138bd` through `29794f7` in the fork. This may or may not affect the Sandbox fork depending on its Amplify version.

---

## Testing

### Part 1 only (x-api-key)

```bash
# Health checks
curl https://<api-url>/api/health -H "x-api-key: <key>"
curl https://<api-url>/api/v1/health -H "x-api-key: <key>"

# List models
curl https://<api-url>/api/v1/models -H "x-api-key: <key>"

# Non-streaming chat (OpenAI format)
curl https://<api-url>/api/v1/chat/completions \
  -H "x-api-key: <key>" \
  -H "Content-Type: application/json" \
  -d '{"model":"claude-v4.5-sonnet","messages":[{"role":"user","content":"Hello!"}]}'

# Streaming chat with per-request temperature
curl https://<api-url>/api/v1/chat/completions \
  -H "x-api-key: <key>" \
  -H "Content-Type: application/json" \
  -d '{"model":"claude-v4.5-sonnet","messages":[{"role":"user","content":"Hello!"}],"stream":true,"temperature":0.2}'

# Non-streaming (Anthropic format)
curl https://<api-url>/api/v1/messages \
  -H "x-api-key: <key>" \
  -H "Content-Type: application/json" \
  -d '{"model":"claude-v4.5-sonnet","system":"You are helpful.","messages":[{"role":"user","content":"Hello!"}],"max_tokens":1024}'
```

### Part 1 + Part 2 (Bearer token)

All the above work with `Authorization: Bearer <key>` in place of `x-api-key: <key>`. Also verify no-auth is rejected:

```bash
# Should work
curl https://<api-url>/api/v1/models -H "Authorization: Bearer <key>"

# Should return 401
curl https://<api-url>/api/v1/models
```

### Client SDK examples

```python
# Anthropic SDK — works with Part 1 only
from anthropic import Anthropic
client = Anthropic(
    base_url="https://<api-url>/api/v1",
    api_key="<key>",
)
message = client.messages.create(
    model="claude-v4.5-sonnet",
    max_tokens=1024,
    system="You are a helpful assistant.",
    messages=[{"role": "user", "content": "Hello!"}],
)

# OpenAI SDK — requires Part 2, OR use custom headers as workaround
from openai import OpenAI

# With Part 2 (Bearer works natively):
client = OpenAI(base_url="https://<api-url>/api/v1", api_key="<key>")

# Without Part 2 (custom header workaround):
client = OpenAI(
    base_url="https://<api-url>/api/v1",
    api_key="placeholder",
    default_headers={"x-api-key": "<key>"},
)

response = client.chat.completions.create(
    model="claude-v4.5-sonnet",
    messages=[{"role": "user", "content": "Hello!"}],
    temperature=0.2,
)

# LangChain (Anthropic) — works with Part 1 only
from langchain_anthropic import ChatAnthropic
llm = ChatAnthropic(
    base_url="https://<api-url>/api/v1",
    api_key="<key>",
    model="claude-v4.5-sonnet",
)

# LangChain (OpenAI) — requires Part 2
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(
    base_url="https://<api-url>/api/v1",
    api_key="<key>",
    model="claude-v4.5-sonnet",
)
```

---

## What This Does NOT Include

- **Structured function calling** (`tools` array) — Would require mapping OpenAI's tool_calls protocol to Bedrock's tool use format. Prompt-engineered tool use works fine.
- **Embeddings** (`/v1/embeddings`) — Bedrock has embedding models but they use a different API. Could be added separately.
- **Conversation persistence** — Both endpoints are stateless by design. Each request is independent. The existing `/conversation` endpoint still handles persistent conversations.

---

## Compliance Preservation Summary

| Control | Existing Bot API | Part 1 (Payload) | Part 1 + Part 2 (Bearer) | Notes |
|---|---|---|---|---|
| API key auth | Yes | Yes | Yes | Same key value, different header |
| Bot-level guardrails | Yes | Yes | Yes | Applied unconditionally |
| Bot instruction/system prompt | Yes | Yes | Yes | Prepended to every request |
| Request audit logging | Yes | Yes | Yes | Same middleware |
| Token usage tracking | Via bot dashboard | Via `usage` field + CloudWatch | Via `usage` field + CloudWatch | From Bedrock Converse API |
| Data isolation (VPC) | Yes | Yes | Yes | Same Lambda, same network |
| Encryption in transit | TLS | TLS | TLS | Same API Gateway |
| Per-bot access control | Yes | Yes | Yes | Same published API isolation |
| Key revocation | Immediate | Immediate | Immediate | TTL=0 on authorizer cache |
| Conversation persistence | Yes (DynamoDB) | No (stateless) | No (stateless) | Privacy advantage for some use cases |

---

## Files in This Fork

All changes are on the `v3` branch of [bhill00/bedrock-chat](https://github.com/bhill00/bedrock-chat/tree/v3):

**Part 1 — Payload Compatibility:**

| File | Change | Purpose |
|---|---|---|
| `backend/app/routes/openai_compat.py` | New | OpenAI + Anthropic endpoints, streaming, bot integration, `/v1/health` |
| `backend/app/routes/schemas/openai_compat.py` | New | Pydantic models for both API formats |
| `backend/app/main.py` | +5 lines | Router registration |
| `frontend/package.json` | Modified | xstate-v4 npm alias (build fix) |
| `frontend/vite.config.ts` | Modified | Vite plugin for xstate resolution (build fix) |
| `frontend/package-lock.json` | Modified | Lockfile update |

**Part 2 — Auth Compatibility (Optional):**

| File | Change | Purpose |
|---|---|---|
| `cdk/lambda/api-key-authorizer/index.py` | New | Lambda authorizer — validates key from `x-api-key` or `Authorization: Bearer` |
| `cdk/lib/api-publishment-stack.ts` | Modified | Wires in Lambda authorizer, replaces built-in API GW key validation |

To see the exact diff: `git diff origin/v3..v3`

---

## Verified Test Results

Tested on a live deployment (personal AWS account, March 2026):

**Part 1:**

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
| API key auth (`x-api-key`) | Both | Same key as existing `/conversation` endpoint |
| Health check | `/v1/health` | `{"status":"ok"}` |

**Part 2:**

| Test | Result |
|---|---|
| `x-api-key` header | ✅ Still works |
| `Authorization: Bearer` header | ✅ Works |
| No auth | ✅ Returns 401 Unauthorized |
| OpenAI SDK (default config, no custom headers) | ✅ Works end-to-end |
| Key revocation | ✅ Immediate (TTL=0) |

### Standalone proxy (bedrock-api-proxy)

The lightweight Lambda proxy ([bhill00/bedrock-api-proxy](https://github.com/bhill00/bedrock-api-proxy)) was the original proof of concept. It provides the same OpenAI + Anthropic endpoints without the Bedrock Chat infrastructure (no bots, no Cognito, no DynamoDB). Useful for direct Bedrock access with just an API key. It handles Bearer tokens natively since it owns its own auth entirely.
