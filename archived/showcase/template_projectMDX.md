---
title: "Example Project Template"
description: "Template for contributing example projects and notebooks to the Fireworks Cookbook"
---

# Project Title

> Brief description of your project, focusing on what it accomplishes or demonstrates.

## Overview

- **Author(s)**: [Your Name(s) - link GitHub/LinkedIn if desired]
- **Date Created**: YYYY-MM-DD
- **Category**: Production-ready / Learning-focused / Community Showcase
- **Fireworks Features Highlighted**: List any specific Fireworks features or APIs this project demonstrates (e.g., RAG, function-calling, agentic workflows).

## Project Summary

Provide a short, high-level summary of the project, explaining:
- What the project does.
- Why it’s valuable or interesting.
- Key technical elements or techniques involved.

## Getting Started

### Prerequisites

List any prerequisites (e.g., installed libraries, knowledge areas) needed to set up or understand the project:
- Docker
- Python 3.8+
- Any specific Fireworks dependencies

### Installation and Setup

Provide clear instructions for setting up the project:
```bash
# Clone the repository
git clone [project-url]

# Navigate to the project directory
cd [project-folder]

# Install dependencies
pip install -r requirements.txt

# Set up environment variables (if applicable)
cp .env.template .env
```

### Running the Project

Instructions for running the project locally, in Docker, or another environment:
```bash
# Example command for running locally
python app.py
```

OR for Docker-based setup:
```bash
# Build Docker image
docker build -t my-fireworks-example .

# Run Docker container
docker run -p 8080:8080 my-fireworks-example
```

## How It Works

Explain the project’s functionality and major components:
1. **Input data**: Describe the input(s) used (e.g., text, images).
2. **Processing**: Explain the core methods, functions, or Fireworks features that power the project.
3. **Output**: Describe the output format or expected results.

## Results and Insights

Share any interesting findings or insights from the project. For learning-focused projects, provide any "aha" moments or lessons learned.

## Key Code Snippets

Highlight essential code sections that demonstrate core functionality:
```python
# Example of loading and using Fireworks model
from fireworks import Model

model = Model("model-name")
response = model.predict(input_data)
```

## Additional Resources

- **Related Documentation**: [Fireworks Documentation](https://docs.fireworks.ai)
- **Additional Links**: Link any related tutorials, blog posts, or resources

## Contributing and Feedback

We appreciate your feedback and contributions! If you'd like to improve this project or have questions, please [open an issue](https://github.com/fireworks-ai/examples/issues) or contact us on [Discord](https://discord.gg/9nKGzdCk).

**Thank you for contributing to the Fireworks community!**

---

> **Note**: Delete any sections that aren't applicable to your project to keep this template streamlined.
