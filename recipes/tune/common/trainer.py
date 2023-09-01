# Copyright (c) Fireworks AI, Inc. and affiliates.
#
# All Rights Reserved.

import torch
import torch.distributed as dist
from datasets import DatasetDict
import transformers
from transformers import AutoTokenizer
from omegaconf import DictConfig


def train(
    config: DictConfig,
    tokenizer: AutoTokenizer,
    dataset: DatasetDict,
    model: torch.nn.Module,
) -> torch.nn.Module:
    """
    Fine tunes a summarization model.

    Args:
        config: the configuration describing the training program,
        tokenizer: the tokenizer to use,
        dataset: the training dataset,
        model: the model to train.

    Returns:
        trained model.
    """
    per_device_macro_batch_size = config.model.batch_size // dist.get_world_size()
    gradient_accumulation_steps = (
        per_device_macro_batch_size // config.model.micro_batch_size
    )
    print(
        f"per_device_train_batch_size: {config.model.micro_batch_size} "
        f"gradient_accumulation_steps: {gradient_accumulation_steps}"
    )
    deepspeed_config = config.model.get("deepspeed_config", None)
    if deepspeed_config:
        print(f"Deepspeed config: {deepspeed_config}")
    load_in_8bit = config.model.get("load_in_8bit", False)
    ddp_find_unused_parameters = not (
        config.model.gradient_checkpointing or load_in_8bit
    )
    kwargs = {}
    lr_scheduler_type = config.model.get("lr_scheduler_type")
    if lr_scheduler_type:
        kwargs["lr_scheduler_type"] = lr_scheduler_type
    optim = config.model.get("optim")
    if optim:
        kwargs["optim"] = optim
    warmup_steps = config.model.get("warmup_steps", 10)
    trainer = transformers.Trainer(
        model=model,
        train_dataset=dataset,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=config.model.micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            num_train_epochs=config.model.epochs,
            learning_rate=config.model.learning_rate,
            bf16=config.model.get("bf16", False),
            logging_steps=1,
            output_dir=config.working_dir,
            save_strategy="no",
            deepspeed=deepspeed_config,
            gradient_checkpointing=config.model.gradient_checkpointing,
            ddp_find_unused_parameters=ddp_find_unused_parameters,
            group_by_length=False,
            # max_steps=1,
            report_to=None,
            **kwargs,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False
    trainer.train(resume_from_checkpoint=False)

    return model
