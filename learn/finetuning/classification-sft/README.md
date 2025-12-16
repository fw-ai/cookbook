# Quick Start Classification and Eval SFT Demo

This project demonstrates a complete workflow for Supervised Fine-Tuning (SFT) on Fireworks AI. We will fine-tune a model to **classify customer support tickets** into three categories: `billing`, `hardware`, or `software`.

The workflow covers data generation, training, deployment, and **post-training evaluation** to ensure quality.

## Workflow Steps

1. **Setup Environment**

   **Install Tools**:
   - Install `firectl` CLI: [Installation Guide](https://docs.fireworks.ai/tools-sdks/firectl/firectl)
   - Install Python dependencies:
     ```bash
     pip install requests
     # Or with uv:
     uv pip install requests
     ```

   **Set Credentials**:
   Set your API key and Account ID.
   ```bash
   export FIREWORKS_API_KEY="your_api_key"
   export ACCOUNT_ID="your_account_id"

   # Optional: Set WANDB configuration
   export WANDB_PROJECT="your_wandb_project"
   export WANDB_ENTITY="your_wandb_entity"
   export WANDB_API_KEY="your_wandb_api_key"
   ```

2. **Generate Data**
   Run the data generator to create synthetic support tickets.
   ```bash
   python gen_toy_data.py
   ```
   *This creates a 90/10 split: `toy_support_data_train.jsonl` and `toy_support_data_val.jsonl`.*

3. **Train**
   Launch the fine-tuning job. This script automatically handles resource cleanup, dataset uploads, and launches the job.
   ```bash
   python run_sft_job.py
   ```
   *Wait for the job to complete (~5-10 mins). The script will print the next steps when finished.*

4. **Deploy**
   Deploy the fine-tuned model. We provide a script that creates the deployment and polls it until the model is fully loaded on the GPU (warm start).
   ```bash
   python deploy.py
   ```
   *This script sets `min-replica-count 1` and waits until the inference endpoint successfully returns a response.*

5. **Evaluate**
   Run the evaluation script to calculate metrics on the validation set (`toy_support_data_val.jsonl`).

   ```bash
   python eval_model.py
   ```
   *This prints detailed one-vs-rest Precision/Recall/F1 metrics.*

   You can also test a specific prompt:
   ```bash
   python eval_model.py "My login is broken"
   ```

6. **Cleanup**
   Remove all datasets, models, and deployments created by this demo.
   ```bash
   python cleanup.py
   ```

## Configuration
All constants (Model IDs, file names, deployment settings) are centralized in `config.py`.
