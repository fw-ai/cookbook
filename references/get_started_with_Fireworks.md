
# Getting Started with Fireworks

This section focuses on examples built entirely using Fireworks AI. These projects demonstrate the core capabilities of Fireworks without external dependencies, giving you a straightforward path to mastering Fireworks.

---

## Inference with Fireworks

- **Project**: [Inference with Fireworks](./1_inference/README.md)
- **Objective**: Learn how to perform model inference using Fireworks AI.
- **Topics Covered**: Loading Fireworks models, performing inference, and optimizing performance.
- **Getting Started**: Detailed instructions can be found in the `1_inference` directory.
- **Video Link**: [Watch Video](#)

---

## Fine-Tuning with Fireworks

- **Project**: [Fine-Tuning with Fireworks](./2_fine-tuning/README.md)
- **Objective**: Learn how to fine-tune models using Fireworks AI.
- **Topics Covered**: Dataset preparation, model fine-tuning techniques, and evaluation.
- **Getting Started**: Instructions are available in the `2_fine-tuning` directory.
- **Video Link**: [Watch Video](#)

---

## Function-Calling with Fireworks

- **Project**: [Function-Calling with Fireworks](./3_function-calling/README.md)
- **Objective**: Explore Fireworks' function-calling capabilities and how to deploy function-calling LLMs.
- **Topics Covered**: Function-call models, handling function requests, and workflows.
- **Getting Started**: Instructions are in the `3_function-calling` directory.
- **Video Link**: [Watch Video](#)

---

## RAG with Fireworks

- **Project**: [RAG with Fireworks](./4_rag/README.md)
- **Objective**: Learn how to build RAG systems entirely using Fireworks AI.
- **Topics Covered**: Document retrieval, knowledge-base integration, and response generation.
- **Getting Started**: Detailed guide available in the `4_rag` directory.
- **Video Link**: [Watch Video](#)

---

## Found an Error?

If you encounter any errors or issues while working through the examples, we encourage you to [open an issue](https://github.com/fireworks-ai/examples/issues/new). Please provide details about the problem and, if possible, steps to reproduce it. Our team is constantly improving the projects based on community feedback, and we appreciate your contributions!

---


## Installation and Setup

To run these projects locally, follow the instructions below to set up your environment:

### Step 1: Clone the Repository

Start by cloning the Fireworks repository:

```bash
git clone https://github.com/fireworks-ai/examples.git
cd examples
```

### Step 2: Set Up a Python Virtual Environment

Creating a virtual environment ensures that dependencies are isolated from your system Python installation.

1. **Create the virtual environment:**

    ```bash
    python3 -m venv env
    ```

2. **Activate the virtual environment:**

    - On **MacOS/Linux**:
      ```bash
      source env/bin/activate
      ```
    - On **Windows**:
      ```bash
      .\env\Scripts\activate
      ```

### Step 3: Install Dependencies

Once the virtual environment is activated, install the dependencies for the project:

```bash
pip install -r requirements.txt
```

Each project may have additional dependencies specified in its own directory. After navigating to the respective project folder, install any extra requirements with:

```bash
pip install -r <project_folder>/requirements.txt
```

### Step 4: Run the Projects

Now that your environment is set up, navigate to the project directory you want to run. For example, for inference:

```bash
cd 1_inference
python inference_example.py
```

Follow the instructions in each project’s `README.md` for specific details and commands.

---

## Resources

For more resources and detailed documentation on Fireworks, visit the [Fireworks Documentation](https://docs.fireworks.ai) or join the discussions in the [community forum](https://community.fireworks.ai).

