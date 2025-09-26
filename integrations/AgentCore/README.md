# AgentCore Integration with Fireworks AI

Deploy AI agents on AgentCore using Fireworks AI models with Strands framework.

## Directory Structure

```
AgentCore/
├── Makefile                                      # Setup and environment management
├── requirements.txt                              # Python dependencies
├── runtime_with_strands_and_fireworksai_models.ipynb  # Main deployment notebook
├── strands_agents_fireworks_ai.py              # AgentCore deployment script
└── strands_agents_fireworks_ai_local.py        # Local testing script
```

## Setup

Initialize the environment with all dependencies:

```bash
make setup
```

This will:
- Install `uv` package manager
- Create Python 3.11 virtual environment
- Install all required dependencies

## Deploy to AgentCore

1. Set your Fireworks API key in `.env`:
```
FIREWORKS_API_KEY=your_api_key_here
```

2. Open the Jupyter notebook:
```bash
jupyter notebook runtime_with_strands_and_fireworksai_models.ipynb
```

3. Follow the notebook cells to:
   - Configure your AgentCore runtime
   - Deploy the Strands agent with Fireworks AI models
   - Test the deployed endpoint

## Local Testing

Test agents locally before deployment:

```bash
python strands_agents_fireworks_ai_local.py
```