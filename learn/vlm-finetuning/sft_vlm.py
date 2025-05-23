# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 
# Modified by Fireworks AI 2025 - Added support for inline image format

"""
pip install pillow

# Tested on 8x H100 GPUs
accelerate launch
    --config_file=examples/accelerate_configs/deepspeed_zero3.yaml \
    examples/scripts/sft_vlm.py \
    --dataset_name HuggingFaceH4/llava-instruct-mix-vsft \
    --model_name_or_path llava-hf/llava-1.5-7b-hf \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --output_dir sft-llava-1.5-7b-hf \
    --bf16 \
    --torch_dtype bfloat16 \
    --gradient_checkpointing

For LLaVA-NeXT, use: (requires transformers>=4.45)
    --model_name_or_path llava-hf/llava-v1.6-mistral-7b-hf

For meta-llama/Llama-3.2-11B-Vision-Instruct, use: (requires transformers>=4.45.1)
    --model_name_or_path meta-llama/Llama-3.2-11B-Vision-Instruct
"""

import torch
from datasets import load_dataset
from transformers import AutoModelForVision2Seq, AutoProcessor, LlavaForConditionalGeneration
import base64
from io import BytesIO
import os
from pydantic import BaseModel
from PIL import Image
from trl import (
    ModelConfig,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from dataclasses import dataclass, field
from datasets import Dataset, DatasetDict
import json
from typing import Dict, Any, List, Union, Generator
from qwen_vl_utils import process_vision_info

@dataclass
class ScriptArguments:
    dataset_path: str = field(default=None, metadata={
        "help": (
            "Path to .jsonl file containing rows in OpenAI format. Supports images in base64 and file path"
        )
    })

def create_collate_fn(processor, dataset_path):
    def fn(examples):
        return collate_fn(examples, processor, dataset_path)
    return fn

def collate_fn(examples, processor, dataset_path):
    """
    Collate function that handles inline image format with both base64 and file paths.
    
    Expected format:
    {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "path/to/image.jpg"},  # or base64
                    {"type": "text", "text": "What's in this image?"}
                ]
            },
            {"role": "assistant", "content": "I see a cat."}
        ]
    }
    """
    texts = []
    images = []
    
    for example in examples:
        
        messages = example["messages"]
        
        # Apply chat template
        text = processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=False
        )
        
        example_images = []
        for msg in messages:
            if isinstance(msg["content"], list):
                for item in msg["content"]:
                    if item["type"] == "image":
                        # Load image (handles both base64 and file paths)
                        try:
                            img = load_image(item["image"], dataset_path)
                            example_images.append(img)
                        except Exception as e:
                            print(f"WARNING: Failed to load image: {e}")
                            # Skip this image but continue processing
                            continue
        
        # Apply chat template to get formatted text
        print(text)
        print(example_images)
        print("--------------------------------")
        texts.append(text)
        images.append(example_images)
    
    # Tokenize texts and process images
    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

    print(batch)
    
    # Create labels for training (input_ids with padding masked)
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    
    # Mask image tokens in loss computation (model-specific)
    if hasattr(processor, 'image_token'):
        image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
        labels[labels == image_token_id] = -100
    
    batch["labels"] = labels
    
    return batch

def load_image(image_data, dataset_path):
    """
    Load image from either file path or base64 string.
    
    Args:
        image_data (str): Either a file path or base64 encoded image
        
    Returns:
        PIL.Image: RGB image ready for processing
    """
    try:
        # Check if it's base64 (data URL format or long string)
        if image_data.startswith('data:image') or (len(image_data) > 100 and not os.path.exists(image_data)):
            # Handle base64 image
            if image_data.startswith('data:image'):
                # Remove data URL prefix (e.g., "data:image/jpeg;base64,")
                base64_string = image_data.split(',', 1)[1]
            else:
                # Assume it's raw base64
                base64_string = image_data
            
            # Decode base64 to bytes
            image_bytes = base64.b64decode(base64_string)
            
            # Convert to PIL Image
            image = Image.open(BytesIO(image_bytes))
        else:
            # Assume file path
            if image_data.startswith('/'):
                if not os.path.exists(image_data):
                    raise FileNotFoundError(f"Image file not found: {image_data}")
                image = Image.open(image_data)
            else:
                dir_path = os.path.dirname(dataset_path)
                if not os.path.exists(os.path.join(dir_path, image_data)):
                    raise FileNotFoundError(f"Image file not found: {image_data}")
                image = Image.open(os.path.join(dir_path, image_data))
        
        # Ensure RGB format (removes alpha channel, handles different formats)
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        return image
        
    except Exception as e:
        raise ValueError(f"Failed to load image from '{image_data[:50]}...': {str(e)}")


def jsonl_generator(file_path: str) -> Generator[Dict[str, Any], None, None]:
    """
    Huggingface Dataset doesn't like when "content" is a string and can be an array.
    Memory-efficient generator that yields one parsed JSON object at a time.
    Handles mixed content formats (string vs array) in messages.
    
    Args:
        file_path (str): Path to the JSONL file
        standardize_to_array (bool): If True, convert all content to array format.
                                   If False, convert all content to string format.
        
    Yields:
        Dict[str, Any]: Parsed JSON object from each line with standardized content
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:  # Skip empty lines
                try:
                    data = json.loads(line)
                    
                    # Standardize content fields in messages if they exist
                    if 'messages' in data and isinstance(data['messages'], list):
                        for message in data['messages']:
                            if 'content' in message:
                                message['content'] = standardize_content(message['content'])
                    yield data
                except json.JSONDecodeError as e:
                    print(f"Warning: Error parsing line {line_num}: {e}")
                    continue
                except Exception as e:
                    print(f"Warning: Error processing line {line_num}: {e}")
                    continue

def standardize_content(content: Union[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Standardize content field to always be an array format.
    
    Args:
        content: Either a string or array of content objects
        
    Returns:
        List[Dict[str, Any]]: Standardized content as array of objects
    """
    if isinstance(content, str):
        # Convert string to array format
        return [{"type": "text", "text": content}]
    elif isinstance(content, list):
        # Already in array format, return as-is
        return content
    else:
        raise Error("Unexpected content type: " + type(content))


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig,))
    script_args, training_args, model_args = parser.parse_args_and_config()
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    training_args.remove_unused_columns = False
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}

    ################
    # Model, Tokenizer & Processor
    ################
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
    )

    model = AutoModelForVision2Seq.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, **model_kwargs
    )

    ################
    # Dataset
    ################
    dataset = Dataset.from_generator(
        jsonl_generator, 
        gen_kwargs={"file_path": script_args.dataset_path}
    )

    ################
    # Training
    ################
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        data_collator=create_collate_fn(processor, script_args.dataset_path),
        train_dataset=dataset,
        processing_class=processor.tokenizer,
        peft_config=get_peft_config(model_args),
    )

    trainer.train()

    trainer.save_model(training_args.output_dir)