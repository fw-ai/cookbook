from strands import Agent, tool
from strands_tools import file_read, file_write
import argparse
import json
# NOTE: FireworksAI is compatible with OpenAI sdk
from strands.models.openai import OpenAIModel
from dotenv import load_dotenv
import os

load_dotenv()

FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")

@tool
def code_python(user_prompt: str):
    prompt = f"""
    You are a Python developer. You will be provided a user request for python code.
    Write clean, readable Python code.  Prioritize simple, working solutions.

    Only generate the code explicitly requested by the user. Do not include any additional code, if you need to add examples of how to run the code add it to the docstrings only.

    Remember the Zen of Python:
    ```
    Beautiful is better than ugly.
    Explicit is better than implicit.
    Simple is better than complex.
    Complex is better than complicated.
    Flat is better than nested.
    Sparse is better than dense.
    Readability counts.
    Special cases aren't special enough to break the rules.
    Although practicality beats purity.
    Errors should never pass silently.
    Unless explicitly silenced.
    In the face of ambiguity, refuse the temptation to guess.
    There should be one-- and preferably only one --obvious way to do it.
    Although that way may not be obvious at first unless you're Dutch.
    Now is better than never.
    Although never is often better than right now.
    If the implementation is hard to explain, it's a bad idea.
    If the implementation is easy to explain, it may be a good idea.
    Namespaces are one honking great idea – let's do more of those!
    ```

    The user request is: {user_prompt}
    """

    return prompt

model = OpenAIModel(
    client_args={
        "api_key": FIREWORKS_API_KEY,
        "base_url": "https://api.fireworks.ai/inference/v1",
    },
    # **model_config
    model_id="accounts/fireworks/models/kimi-k2-instruct-0905",
    params={
        "max_tokens": 3000,
        "temperature": 0.0,
    }
)

agent = Agent(
    model=model,
    tools=[file_read, file_write, code_python],
    system_prompt="You are a software engineer. You can read files, write files and generate python code."
)

def strands_agent_fireworks_ai(payload):
    """
    Invoke the agent with a payload
    """
    user_input = payload.get("prompt")
    response = agent(user_input)
    return response.message['content'][0]['text']

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("payload", type=str)
    args = parser.parse_args()
    response = strands_agent_fireworks_ai(json.loads(args.payload))
    print(response)