import os
import time
import json
from typing import List, Dict
import requests


class LLMClient:
    """Simple client for interacting with LocalAI's API with streaming support"""

    def __init__(self, api_base: str = "http://localhost:8080/v1"):
        self.api_base = api_base
        self.headers = {"Content-Type": "application/json"}

    def list_models(self) -> List[Dict]:
        """List available models"""
        response = requests.get(f"{self.api_base}/models")
        return response.json()

    def _format_phi_messages(self, messages: List[Dict]) -> str:
        """Format messages for Phi model's expected input format"""
        formatted = ""

        system_msg = next((msg for msg in messages if msg["role"] == "system"), None)
        if system_msg:
            formatted += f"<|system|>{system_msg['content']}<|end|>"

        user_msgs = [msg for msg in messages if msg["role"] == "user"]
        if user_msgs:
            formatted += f"<|user|>{user_msgs[-1]['content']}<|end|><|assistant|>"

        return formatted

    def chat_blocking(
        self,
        messages: List[Dict],
        model: str = "phi-3.5-mini-instruct",
    ) -> str:
        """Send a blocking (non-streaming) chat completion request"""
        formatted_prompt = self._format_phi_messages(messages)

        data = {
            "prompt": formatted_prompt,
            "model": model,
            "stream": False,
            "top_p": 0.1,
            "temperature": 0.3,
            "stop": ["<|endoftext|>", "<|end|>"],
            "max_tokens": 1024,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
        }

        start_time = time.time()

        response = requests.post(
            f"{self.api_base}/completions", headers=self.headers, json=data
        )

        if response.status_code != 200:
            raise Exception(f"Error: {response.text}")

        result = response.json()
        elapsed_time = time.time() - start_time

        if result.get("choices") and len(result["choices"]) > 0:
            content = result["choices"][0].get("text", "")
            print("Time taken:", elapsed_time)
            return content

        raise Exception("No valid response received from the API")


def demonstrate_capabilities():
    """Show basic capabilities of the LLM with streaming responses"""
    llm = LLMClient(os.getenv("LLM_API_BASE", "http://localhost:8080/v1"))

    print("\nAvailable Models:")
    try:
        models = llm.list_models()
        print(json.dumps(models, indent=2))
    except Exception as e:
        print(f"Error listing models: {e}")

    examples = [
        {
            "title": "DevOps Explanation",
            "messages": [
                {
                    "role": "user",
                    "content": "What is DevOps in one sentence?",
                },
            ],
        },
    ]

    for example in examples:
        print(f"\nExample: {example['title']}")
        print("Response:")

        response = llm.chat_blocking(example["messages"])
        print(response)


if __name__ == "__main__":
    print("🤖 LocalAI Streaming Client Demo")
    print("Testing connection to LocalAI and demonstrating basic capabilities...")
    demonstrate_capabilities()
