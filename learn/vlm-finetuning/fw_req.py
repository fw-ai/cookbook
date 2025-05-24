#!/usr/bin/env python3
import requests
import base64
import json
import os
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description='Query Fireworks AI vision model with an image')
    parser.add_argument('--model', '-m', 
                       default='accounts/aidan-0d49e1/models/sft-qwen2p5-vl-7b-instruct-9',
                       help='Model to use')
    parser.add_argument('--api_key', '-k',
                       help='Fireworks API key',
                       required=True)
    args = parser.parse_args()
    
    # Read and encode image
    with open("icecream.png", 'rb') as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')

    payload = {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{image_data}"
                    }},
                    {"type": "text", "text": "What's in this image?"},
                ]
            }
        ]
    }
    
    try:
        response = requests.post(
            "https://api.fireworks.ai/inference/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {args.api_key}",
                "Content-Type": "application/json"
            },
            json=payload
        )
        
        response.raise_for_status()
        result = response.json()
        print(result['choices'][0]['message']['content'])        
    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error parsing response: {e}", file=sys.stderr)
        print(f"Raw response: {response.text}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()