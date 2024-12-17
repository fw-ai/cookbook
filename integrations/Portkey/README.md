# Portkey + Fireworks Integration

This cookbook demonstrates how to effectively use Fireworks models through Portkey's AI Gateway. Learn how to integrate Fireworks models with enhanced reliability, observability, and performance features.

<img width="400" src="https://assets-global.website-files.com/64060a74d132d0ca9fb8c033/64060a74d132d0cb6bb8c080_Logo.svg" alt="portkey">

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/your_notebook_link)

## Features

- 🚀 Fast AI Gateway access to Fireworks models
- 📊 Comprehensive observability and logging
- 💾 Semantic caching for improved performance
- 🔄 Automated retries and fallbacks
- 📝 Custom metadata and request tracing

## Getting Started

1. Install the required packages:
```bash
pip install portkey-ai
```

2. Set up your API keys in environment variables:
```python
export PORTKEY_API_KEY="your_portkey_key"
export FIREWORKS_VIRTUAL_KEY="your_fireworks_key"  # Get from Portkey dashboard
```

## Examples

The `examples/` directory contains several Python scripts demonstrating different features:

- `basic_chat.py`: Simple chat completion setup
- `routing.py`: Load balancing and fallback configurations
- `observability.py`: Request tracing and logging
- `caching.py`: Semantic caching implementation
- `production.py`: Production-ready patterns

## Usage

Basic chat completion:
```python
from portkey_ai import Portkey

portkey = Portkey(
    api_key="your_portkey_key",
    virtual_key="your_fireworks_key"
)

completion = portkey.chat.completions.create(
    messages=[{"role": "user", "content": "Hello!"}],
    model="accounts/fireworks/models/llama-v3-8b-instruct"
)
print(completion.choices[0].message.content)
```

## Features Overview

### Load Balancing
Distribute traffic across multiple models or API keys with custom weights.

### Request Tracing
Track requests with custom metadata and trace IDs for better observability.

### Semantic Caching
Improve response times and reduce costs with intelligent response caching.

### Fallbacks & Retries
Ensure high availability with automatic retries and model fallbacks.

## Directory Structure

```
portkey-fireworks/
├── README.md
├── examples/
│   ├── basic_chat.py
│   ├── routing.py
│   ├── observability.py
│   ├── caching.py
│   └── production.py
├── notebooks/
│   └── portkey_fireworks.ipynb
└── requirements.txt
```

## Contributing

Please read our [Contributing Guide](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## Support

- [Portkey Documentation](https://docs.portkey.ai)
- [Fireworks Documentation](https://docs.fireworks.ai)
- [Join Portkey Discord](https://discord.gg/sDk9JaNfK8)