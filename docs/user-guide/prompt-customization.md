# Prompt Customization

Learn how to customize prompts at multiple levels to optimize extraction quality for your specific use case.

## Overview

DELM builds prompts through multiple layers that you can customize:

1. **ExtractionVariable descriptions** - Define what each field means
2. **Prompt template** - Structure how variables and text are presented
3. **System prompt** - Set the LLM's role and behavior
4. **Instructor wrapper** - Adds structured output instructions (automatic)

## 1. ExtractionVariable Descriptions

The most common customization point. Variable descriptions directly influence what the LLM extracts.

### Basic Example

```python
from delm import DELM, Schema, ExtractionVariable

schema = Schema.simple([
    ExtractionVariable(
        name="price",
        description="The numeric price value mentioned in the text",
        data_type="number"
    ),
    ExtractionVariable(
        name="currency",
        description="The currency code (USD, EUR, GBP, etc.)",
        data_type="string"
    )
])
```

**Generated prompt variables section:**
```
- price (number): The numeric price value mentioned in the text
- currency (string): The currency code (USD, EUR, GBP, etc.)
```

### Advanced Descriptions

Use descriptions to:
- **Clarify ambiguity**
- **Provide examples**
- **Set extraction rules**

```python
ExtractionVariable(
    name="horizon",
    description="Time horizon for the forecast if mentioned (e.g., '2024', 'Q1 2025', 'next year'). Only extract if explicitly stated.",
    data_type="string"
)

ExtractionVariable(
    name="commodity_type",
    description="Type of commodity mentioned. Must be one of: oil, gas, gold, silver, copper. Do NOT extract if the commodity is not explicitly mentioned with a price.",
    data_type="string",
    allowed_values=["oil", "gas", "gold", "silver", "copper"]
)

ExtractionVariable(
    name="price_range",
    description="Extract as a list of two numbers [min, max] if a range is given (e.g., '$50-75 per barrel' → [50, 75]). If single price, return empty list.",
    data_type="[number]"
)
```

## 2. Prompt Template

The `prompt_template` controls how your variables and text are presented to the LLM.

### Default Template

```python
default_template = """Extract the following information from the text:

{variables}

Text to analyze:
{text}"""
```

### Custom Templates

```python
# Example: Financial analysis focus
delm = DELM(
    schema=schema,
    prompt_template="""You are analyzing a financial earnings report.

Extract these specific data points:
{variables}

Important: Only extract information explicitly stated in the text. Do not infer or estimate values.

Report text:
{text}

Remember: If a field is not mentioned, leave it empty rather than guessing."""
)
```

```python
# Example: Academic research extraction
delm = DELM(
    schema=schema,
    prompt_template="""Task: Extract structured data from an academic paper excerpt.

Data fields to extract:
{variables}

Source text:
{text}

Extraction guidelines:
- Be precise and cite exact phrases when possible
- Distinguish between stated facts and author interpretations
- Mark uncertainty when information is ambiguous"""
)
```

```python
# Example: Multi-language support
delm = DELM(
    schema=schema,
    prompt_template="""Extract the following information from the text (text may be in English, Spanish, or French):

{variables}

Text:
{text}

Note: Normalize all extracted values to English."""
)
```

### Available Placeholders

- **`{text}`** - The text chunk to extract from (required)
- **`{variables}`** - Auto-generated list of variables from your schema (required)

## Few-Shot Examples

Include hand-labeled examples in the prompt to improve extraction quality.
Each example is a dict with a `"text"` key (source text) and an `"output"`
key (expected extraction result as a dict or JSON string).

```python
delm = DELM(
    schema=schema,
    few_shot_examples=[
        {
            "text": "Brent crude is forecast to hit $85 per barrel in Q4.",
            "output": {"commodity": "Brent crude", "price_value": 85},
        },
        {
            "text": "Gold remained flat this week.",
            "output": {"commodity": "gold", "price_value": None},
        },
    ],
    few_shot_num_examples=2,        # examples per prompt
    few_shot_truncate_length=200,   # max tokens per example text (None = no truncation)
    few_shot_random_sample=True,    # sample randomly from the pool per request
)
```

Notes:

- The rendered examples block is prepended once to the final prompt (this also
  holds for `Schema.multiple`, which repeats the template per sub-schema).
- Random sampling is seeded (`SYSTEM_RANDOM_SEED = 42`), so runs are reproducible.
- If fewer examples are available than requested, all available examples are
  used and a warning is logged.
- Few-shot examples count toward input tokens; cost estimation utilities
  include them automatically.

## 3. System Prompt

The `system_prompt` sets the LLM's role and behavior. This is sent as the "system" message in the API call.

### Default System Prompt

```python
default_system_prompt = "You are a precise data-extraction assistant."
```

### Custom System Prompts

```python
# Example: Strict extraction
delm = DELM(
    schema=schema,
    system_prompt="""You are a meticulous data extraction specialist.
Your core principle: ONLY extract information that is explicitly stated in the text.
NEVER infer, guess, or fill in missing information.
When uncertain, leave the field empty."""
)
```

```python
# Example: Domain expert
delm = DELM(
    schema=schema,
    system_prompt="""You are a finance professor with expertise in commodity markets.
You extract structured data from earnings reports and market analyses with high precision.
You understand financial terminology and can distinguish between forecasts, guidance, and reported figures."""
)
```

```python
# Example: Quality focus
delm = DELM(
    schema=schema,
    system_prompt="""You are a data-extraction assistant optimized for accuracy over coverage.
Better to extract nothing than extract something incorrectly.
Only extract data when you are highly confident it matches the field description."""
)
```

## 4. Complete Example

Combining all customization layers:

```python
from delm import DELM, Schema, ExtractionVariable

# 1. Define schema with detailed descriptions
schema = Schema.nested(
    container_name="price_forecasts",
    variables_list=[
        ExtractionVariable(
            name="commodity",
            description="Specific commodity mentioned (e.g., 'Brent crude oil', 'natural gas', 'gold'). Include the full commodity name as stated.",
            data_type="string"
        ),
        ExtractionVariable(
            name="price_value",
            description="The forecasted price value as a number. If a range is given, extract the midpoint.",
            data_type="number"
        ),
        ExtractionVariable(
            name="unit",
            description="The unit of measure (e.g., 'per barrel', 'per MMBtu', 'per ounce')",
            data_type="string"
        ),
        ExtractionVariable(
            name="time_horizon",
            description="When this forecast applies (e.g., 'Q4 2024', '2025', 'end of year'). Only extract if explicitly mentioned.",
            data_type="string"
        ),
        ExtractionVariable(
            name="source",
            description="Who made this forecast (e.g., company name, analyst firm, 'management'). Only extract if clearly attributed.",
            data_type="string"
        )
    ]
)

# 2. Create DELM with custom prompts
delm = DELM(
    schema=schema,
    provider="openai",
    model="gpt-4o-mini",
    temperature=0.0,
    
    # Custom system prompt
    system_prompt="""You are a financial data extraction specialist with expertise in commodity markets and earnings reports.
Your goal is to extract price forecasts with high accuracy.
Only extract information that is explicitly stated - never infer or estimate.""",
    
    # Custom prompt template
    prompt_template="""Extract price forecast information from this earnings report excerpt.

For each distinct price forecast mentioned, extract:
{variables}

CRITICAL RULES:
1. Only extract forecasts that include an actual price number
2. If multiple forecasts are mentioned, create separate entries
3. If information for a field is not stated, leave it empty
4. Distinguish between forecasts and historical/current prices

Text to analyze:
{text}"""
)

# 3. Run extraction
results = delm.extract("data/earnings_reports.csv")
```

## How Instructor Wraps Your Prompts

DELM uses the [Instructor library](https://python.useinstructor.com/) for structured output. Instructor automatically:

1. **Adds JSON schema instructions** to ensure the LLM returns valid structured data
2. **Wraps your prompts** in a messages array:
   ```python
   messages = [
       {"role": "system", "content": your_system_prompt},
       {"role": "user", "content": your_prompt_template_filled}
   ]
   ```
3. **Validates responses** against your schema and retries if needed

You don't need to worry about JSON formatting instructions - Instructor handles this automatically.

## Preview Your Prompts

Use `preview_prompt()` to see the DELM-built prompt after variable substitution:

```python
delm = DELM(schema=schema, prompt_template="...")

# Preview with sample text
prompt = delm.preview_prompt(text="Oil prices are expected to reach $80 per barrel by Q4 2024.")
print(prompt)
```

**Important**: This shows only the user prompt that DELM builds (your `prompt_template` with `{text}` and `{variables}` filled in). It does **not** include:
- The system prompt (sent separately in the API call)
- Instructor's JSON schema wrapper (added automatically during the API call)

This preview is useful for debugging your template and variable formatting, but the actual LLM receives additional instructions from Instructor for structured output.

## Best Practices

### 1. Start Simple

Begin with clear variable descriptions before customizing the template:

```python
# Good: Clear, specific description
ExtractionVariable(
    name="price",
    description="The numeric price value mentioned (without currency symbol)",
    data_type="number"
)

# Less effective: Vague description
ExtractionVariable(
    name="price",
    description="price",
    data_type="number"
)
```

### 2. Be Explicit About Edge Cases

```python
ExtractionVariable(
    name="revenue",
    description="Quarterly revenue in millions. If revenue is stated in billions, convert to millions. If 'year-to-date' or 'annual' revenue is mentioned, do NOT extract.",
    data_type="number"
)
```

### 3. Test Incrementally

1. Start with default prompts
2. Run evaluation to identify issues
3. Adjust descriptions for low-performing fields
4. If descriptions aren't enough, customize the prompt template
5. Use system prompt for overall behavior changes

### 4. Avoid Redundancy

Don't repeat the same instructions in multiple places. If all variables need the same rule, put it in the prompt template or system prompt:

```python
# Less efficient: Repeating in every variable
ExtractionVariable(
    name="price",
    description="Price value. Only extract if explicitly mentioned.",
    data_type="number"
)
ExtractionVariable(
    name="volume",
    description="Volume value. Only extract if explicitly mentioned.",
    data_type="number"
)

# Better: Put general rule in prompt template
prompt_template = """Extract the following (ONLY if explicitly mentioned):
{variables}

Text: {text}"""
```

### 5. Consider Token Cost

Longer prompts cost more. Find the balance between clarity and conciseness:

```python
# Verbose (higher cost)
description="This field should contain the price value mentioned in the text. The price should be a numeric value without any currency symbols or units. If a price range is given, extract the midpoint. If multiple prices are mentioned, extract all of them as a list."

# Concise (lower cost, equally clear)
description="Numeric price value without symbols. For ranges, use midpoint. Extract all if multiple."
```

