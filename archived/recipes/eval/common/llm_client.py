# Copyright (c) Fireworks AI, Inc. and affiliates.
#
# All Rights Reserved.

import asyncio
import base64
import warnings
from typing import Optional, Union

import aiohttp
import anthropic
import backoff
import google.generativeai as genai
import openai
from anthropic import AsyncAnthropic
from fireworks.client import AsyncFireworks, error
from google.generativeai.types import GenerationConfig
from openai import AsyncOpenAI


def achat_completion_create_retrying(
    client: Union[AsyncOpenAI, AsyncFireworks, AsyncAnthropic],
    semaphore: asyncio.Semaphore,
    sleep_interval: int = 6,
):
    """
    Creates a version of chat_completion_create_retrying bound to the given client and semaphore.

    Usage:
        semaphore = asyncio.Semaphore(6)
        client = AsyncOpenAI()
        achat = achat_completion_create_retrying(client, semaphore)

    Args:
        client: AsyncOpenAI client instance.
        semaphore: asyncio.Semaphore instance.
        sleep interval: time to sleep before we start the request

    Returns:
        A function that returns ChatCompletionResponse
    """

    @backoff.on_exception(
        wait_gen=backoff.expo,
        exception=(
            openai.RateLimitError,
            openai.APIConnectionError,
            openai.APITimeoutError,
            anthropic.RateLimitError,
            anthropic.APIConnectionError,
            anthropic.InternalServerError,
            aiohttp.client_exceptions.ClientPayloadError,
            error.BadGatewayError,
            error.InternalServerError,
            error.RateLimitError,
            error.ServiceUnavailableError,
        ),
        max_value=60,
        max_tries=32,
        factor=1.5,
    )
    async def _chat_completion_create_retrying(*args, **kwargs):
        """
        Calls completion endpoint with retries and backoff in case of transient failures.
        """
        async with semaphore:
            try:
                await asyncio.sleep(sleep_interval)
                if isinstance(client, AsyncOpenAI):
                    result = await client.chat.completions.create(*args, **kwargs)
                elif isinstance(client, AsyncFireworks):
                    result = await client.chat.completions.acreate(*args, **kwargs)
                elif isinstance(client, AsyncAnthropic):
                    result = await client.messages.create(*args, **kwargs)
                else:
                    raise TypeError(f"unsupported client {type(client)}")
                if "error" in result:
                    warnings.warn(result)
                    raise openai.APIError(result["error"], request=None, body=None)
            except Exception as e:
                print("show exception", e)
                raise e
            return result

    return _chat_completion_create_retrying


def achat_completion_create_retrying_gemini(
    semaphore: asyncio.Semaphore,
    model: Optional[genai.GenerativeModel] = None,
    sleep_interval: int = 6,
):
    """
    Creates a version of chat_completion_create_retrying for Google's Gemini model.
    Supports both text and image inputs in OpenAI's format.

    Usage:
        semaphore = asyncio.Semaphore(6)
        achat = achat_completion_create_retrying_gemini(semaphore)

    Args:
        semaphore: asyncio.Semaphore instance to control concurrent requests.
        model: Gemini model instance to use.
        sleep_interval: Time in seconds to sleep before starting each request. Defaults to 6.

    Returns:
        A function that takes chat completion arguments and returns a Gemini response.
        The returned function expects 'model' and 'messages' in its kwargs, where:
        - model: Name of the Gemini model to use
        - messages: List of message dicts with 'role' and 'content' keys
          Supports 'system', 'user' and 'model' roles.
          For messages with images, 'content' should be a list of content items where each item
          can be either a string or a dict with 'type' and 'image_url' keys.
    """

    @backoff.on_exception(
        wait_gen=backoff.expo,
        exception=(Exception),
        max_value=60,
        max_tries=32,
        factor=1.5,
    )
    async def _chat_completion_create_retrying(*args, **kwargs):
        messages = kwargs["messages"]
        system_prompt = None
        if messages[0]["role"].lower() == "system":
            if model is not None:
                raise RuntimeError(
                    "System prompt is not supported with predefined model"
                )
            system_prompt = messages[0]["content"]
            messages = messages[1:]

        gemini_messages = []
        for message in messages:
            if message["role"] == "user":
                parts = []
                content = message["content"]

                # Handle content in OpenAI's format (list of text/image objects)
                if isinstance(content, list):
                    for item in content:
                        if item.get("type") == "text":
                            parts.append(item["text"])
                        elif item.get("type") == "image_url":
                            image_url = item["image_url"]
                            # For base64 images
                            if image_url["url"].startswith("data:image"):
                                # Parse out mimetype and base64 data
                                mime_type = image_url["url"].split(";")[0].split(":")[1]
                                base64_data = image_url["url"].split(",")[1]
                                parts.append(
                                    {
                                        "mime_type": mime_type,
                                        "data": base64.b64decode(base64_data),
                                    }
                                )
                            # For regular URLs
                            else:
                                raise Exception(
                                    f"Regular URLs in message is not supported yet"
                                )
                        else:
                            raise Exception(
                                f"Only text and image_url types are supported"
                            )
                else:
                    # Handle simple string content
                    parts.append(content)

                gemini_messages.append(
                    {
                        "role": "user",
                        "parts": parts,
                    }
                )
            elif message["role"] == "assistant":
                gemini_messages.append(
                    {
                        "role": "model",
                        "parts": [message["content"]],
                    }
                )
            else:
                raise Exception(f"role {message['role']} not supported")

        async with semaphore:
            if model is None:
                generation_config = GenerationConfig(
                    temperature=kwargs.get("temperature", 0),
                    max_output_tokens=kwargs.get("max_tokens", None),
                )
                local_model = genai.GenerativeModel(
                    model_name=kwargs["model"],
                    system_instruction=system_prompt,
                    generation_config=generation_config,
                )
            else:
                local_model = model

            await asyncio.sleep(sleep_interval)
            try:
                result = await local_model.generate_content_async(gemini_messages)
            except Exception as e:
                print("show exception", e)
                raise e
            return result

    return _chat_completion_create_retrying
