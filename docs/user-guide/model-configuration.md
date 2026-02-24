# Model Configuration

DELM supports multiple LLM providers through the [Instructor](https://python.useinstructor.com/) library. This page covers how to configure different providers and use local LLMs.

## Supported Providers

DELM supports any provider that Instructor supports:

| Provider | `provider` value | Example `model` |
|----------|------------------|-----------------|
| OpenAI | `"openai"` | `"gpt-4o-mini"`, `"gpt-4o"` |
| Anthropic | `"anthropic"` | `"claude-3-5-sonnet-20241022"` |
| Google | `"google"` | `"gemini-1.5-flash"` |
| Ollama | `"ollama"` | `"llama3.2"`, `"mistral"` |
| DeepSeek | `"deepseek"` | `"deepseek-chat"` |

### Basic Usage

```python
from delm import DELM

# OpenAI (default)
delm = DELM(
    schema=my_schema,
    provider="openai",
    model="gpt-4o-mini",
)

# Anthropic
delm = DELM(
    schema=my_schema,
    provider="anthropic",
    model="claude-3-5-sonnet-20241022",
)

# Google Gemini
delm = DELM(
    schema=my_schema,
    provider="google",
    model="gemini-1.5-flash",
)
```

## Custom API Endpoints with `base_url`

The `base_url` parameter allows you to point any provider to a custom API endpoint. This is passed directly to Instructor's `from_provider` function.


### Examples

#### OpenAI Compatible Server

```python
from delm import DELM

delm = DELM(
    schema=my_schema,
    provider="openai", 
    model="gpt-4o-mini",
    base_url="http://127.0.0.1:1234/v1",
)
```

#### Ollama

```python
from delm import DELM

delm = DELM(
    schema=my_schema,
    provider="ollama",
    model="llama3.2",
    base_url="http://localhost:11434/v1",
    track_cost=False,
)
```

## Instructor Mode

The `mode` parameter controls how Instructor formats requests to the LLM. Different servers support different modes:

| Mode | Description | Use When |
|------|-------------|----------|
| `"tools"` | Uses function calling | OpenAI, Anthropic, capable local models |
| `"json"` | Uses `response_format: json_object` | Standard OpenAI-compatible servers |
| `"json_schema"` | Uses `response_format: json_schema` | LM Studio, some local servers |
| `"md_json"` | Prompts model to output JSON in markdown | Maximum compatibility |

### Example: LM Studio

LM Studio only supports `json_schema` mode:

```python
from delm import DELM

delm = DELM(
    schema=my_schema,
    provider="openai",
    model="your-model",
    base_url="http://localhost:1234/v1",
    mode="json_schema",
    track_cost=False,
)
```

### Example: Maximum Compatibility

For unknown or limited servers, use `md_json`:

```python
from delm import DELM

delm = DELM(
    schema=my_schema,
    provider="openai",
    model="your-model",
    base_url="http://localhost:8000/v1",
    mode="md_json",  # Works with almost any server
    track_cost=False,
)
```

## Max Completion Tokens

DELM passes `max_completion_tokens` directly to Instructor for each request. The default is `4096`.

```python
from delm import DELM

delm = DELM(
    schema=my_schema,
    provider="openai",
    model="gpt-4o-mini",
    max_completion_tokens=2048,
)
```

## API Pass-through (`api_kwargs`)

The `api_kwargs` parameter lets you pass arbitrary keyword arguments through to the underlying LLM API call. This is useful for provider-specific options that DELM doesn't expose directly.

```python
from delm import DELM

# Disable request storage on Fireworks AI
delm = DELM(
    schema=my_schema,
    provider="openai",
    model="accounts/fireworks/models/llama-v3p1-8b-instruct",
    base_url="https://api.fireworks.ai/inference/v1",
    api_kwargs={"store": False},
)
```

Any keys in `api_kwargs` are unpacked into the `instructor` `create_with_completion` call alongside `model`, `temperature`, `messages`, etc.

## Rate Limiting

Control how fast DELM sends API requests using token-bucket rate limiting. You can specify the number of tokens and/or requests allowed within a configurable time window.

```python
from delm import DELM

# 500k tokens and 500 requests per minute (default period)
delm = DELM(
    schema=my_schema,
    rate_limit_tokens=500_000,
    rate_limit_requests=500,
)

# 33k tokens per second (for high-throughput providers)
delm = DELM(
    schema=my_schema,
    rate_limit_tokens=33_000,
    rate_limit_period_seconds=1.0,
)
```

If a single request requires more tokens than the limit (e.g., a 99k-token prompt with a 33k token/sec limit), the limiter waits until the bucket is fully refilled before allowing that request, then naturally delays subsequent requests until the "debt" is recovered.

## API Keys

DELM reads API keys from environment variables:

| Provider | Environment Variable |
|----------|---------------------|
| OpenAI | `OPENAI_API_KEY` |
| Anthropic | `ANTHROPIC_API_KEY` |
| Google | `GOOGLE_API_KEY` |
| ... | ... |

```bash
# Set your API key
export OPENAI_API_KEY="sk-..."
```

For local servers that don't require authentication, some providers (like Ollama) use placeholder keys automatically.