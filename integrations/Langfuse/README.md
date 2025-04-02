---
title: "Langfuse and Fireworks AI Integration"
description: "Overview of integrating Langfuse with Fireworks for tracing, evaluating and debugging of AI applications."
---

# Langfuse and Fireworks AI Integration

## About Langfuse

[Langfuse](https://langfuse.com) is an open-source platform for LLM engineering. It provides tracing and monitoring capabilities for AI agents, helping developers debug, analyze, and optimize their products. Langfuse integrates with various tools and frameworks via native integrations, OpenTelemetry, and SDKs.

## Why Integrate Langfuse with Fireworks?

 
- Complex, repeated, chained, or agentic calls to foundation models make debugging challenging - observability helps pinpoint root causes of issues.
- LLMs generate unpredictable and variable outputs that are difficult to test against expected results - observability enables monitoring output quality.
- Multiple points of failure in complex LLM applications can impact quality, cost, and latency - observability helps identify and address these issues effectively.

## How Langfuse Works with Fireworks

Fireworks AI's API endpoints are fully [compatible](https://docs.fireworks.ai/tools-sdks/openai-compatibility) with the OpenAI SDK, allowing you to trace and monitor your AI applications seamlessly.

### Step 1: Install Dependencies


```python
%pip install openai langfuse
```

### Step 2: Set Up Environment Variables


```python
import os

# Get keys for your project from the project settings page
# https://cloud.langfuse.com

os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-..." 
os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-..."
os.environ["LANGFUSE_HOST"] = "https://cloud.langfuse.com" # 🇪🇺 EU region
# os.environ["LANGFUSE_HOST"] = "https://us.cloud.langfuse.com" # 🇺🇸 US region

# Set your Fireworks API details
os.environ["FIREWORKS_AI_API_BASE"] = "https://api.fireworks.ai/inference/v1"
os.environ["FIREWORKS_AI_API_KEY"] = "fw_..."
```

### Step 3: Use Langfuse OpenAI Drop-in Replacement


```python
from langfuse.openai import openai

client = openai.OpenAI(
  api_key=os.environ.get("FIREWORKS_AI_API_KEY"),
  base_url=os.environ.get("FIREWORKS_AI_API_BASE")
)
```

### Step 4: Run an Example


```python
response = client.chat.completions.create(
  model="accounts/fireworks/models/llama-v3p1-8b-instruct",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Why is open source important?"},
  ],
  name = "Fireworks-AI-Trace" # name of the trace
)
print(response.choices[0].message.content)
```

### Step 5: Enhance Tracing (Optional)

You can enhance your Fireworks AI traces:

- Add [metadata](https://langfuse.com/docs/tracing-features/metadata), [tags](https://langfuse.com/docs/tracing-features/tags), [log levels](https://langfuse.com/docs/tracing-features/log-levels) and [user IDs](https://langfuse.com/docs/tracing-features/users) to traces
- Group traces by [sessions](https://langfuse.com/docs/tracing-features/sessions)
- [`@observe()` decorator](https://langfuse.com/docs/sdk/python/decorators) to trace additional application logic
- Use [Langfuse Prompt Management](https://langfuse.com/docs/prompts/get-started) and link prompts to traces
- Add [score](https://langfuse.com/docs/scores/custom) to traces

Visit the [OpenAI SDK cookbook](https://langfuse.com/docs/integrations/openai/python/examples) to see more examples on passing additional parameters.
Find out more about Langfuse Evaluations and Prompt Management in the [Langfuse documentation](https://langfuse.com/docs).

### Step 6: See Traces in Langfuse

After running the example, log in to Langfuse to view the detailed traces, including:

- Request parameters
- Response content
- Token usage and latency metrics

<img src="https://langfuse.com/images/cookbook/integration-fireworks-ai/fireworks-ai-example-trace.png" alt="Langfuse Trace Example" style="border-radius: 8px;" />

_[Public example trace link in Langfuse](https://cloud.langfuse.com/project/cloramnkj0002jz088vzn1ja4/traces/2c11b0e4-eb40-49de-aee9-2ed11bed2839?timestamp=2025-03-05T13%3A31%3A34.781Z&observation=e9668bb4-29d7-4239-87be-e3019480f71f)_

## Additional Resources
- **[Langfuse Documentation](https://langfuse.com/docs)** 
- **[Fireworks AI Documentation](https://docs.fireworks.ai/getting-started/introduction)**