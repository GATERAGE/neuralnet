#!/usr/bin/env python3

import os
import torch
import requests
import json
import openai

from production_transformer import ProductionTransformer

class LLMRouter:
    """
    Routes inference to local or remote LLM backends:
      - local (ProductionTransformer)
      - OpenAI
      - Together.ai
      - Ollama
    """
    def __init__(self, local_vocab_size=10000, local_ckpt="transformer_checkpoint.pt"):
        self.local_ckpt = local_ckpt
        self.local_vocab_size = local_vocab_size
        self.local_model = None

        # Env variables for external APIs
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self.together_api_key = os.getenv("TOGETHER_API_KEY", "")
        self.ollama_endpoint = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11411")

        openai.api_key = self.openai_api_key

    def load_local_model(self):
        if self.local_model is None:
            model = ProductionTransformer(vocab_size=self.local_vocab_size)
            if os.path.exists(self.local_ckpt):
                model.load_state_dict(torch.load(self.local_ckpt, map_location='cpu'))
            model.eval()
            self.local_model = model
        return self.local_model

    def generate(self, prompt, backend="local"):
        if backend == "local":
            return self._generate_local(prompt)
        elif backend == "openai":
            return self._generate_openai(prompt)
        elif backend == "together":
            return self._generate_together(prompt)
        elif backend == "ollama":
            return self._generate_ollama(prompt)
        else:
            return f"[Error] Unsupported backend: {backend}"

    def _generate_local(self, prompt):
        # Placeholder local generation (dummy logic)
        model = self.load_local_model()
        dummy_input = torch.randint(0, self.local_vocab_size, (1, 20))
        with torch.no_grad():
            logits = model(dummy_input)
        return "[Local Model Placeholder Response]"

    def _generate_openai(self, prompt):
        if not self.openai_api_key:
            return "[OpenAI API key not provided]"
        try:
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=prompt,
                max_tokens=100,
                temperature=0.7
            )
            return response['choices'][0]['text'].strip()
        except Exception as e:
            return f"[OpenAI Error: {str(e)}]"

    def _generate_together(self, prompt):
        if not self.together_api_key:
            return "[Together.ai API key not provided]"
        try:
            endpoint = "https://api.together.ai/generate"
            headers = {
                "Authorization": f"Bearer {self.together_api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "together/galactica-6.7b",
                "prompt": prompt,
                "max_tokens": 100
            }
            resp = requests.post(endpoint, headers=headers, data=json.dumps(payload))
            if resp.status_code == 200:
                data = resp.json()
                return data.get("text", "[No response from Together.ai]")
            else:
                return f"[Together.ai Error: {resp.status_code} {resp.text}]"
        except Exception as e:
            return f"[Together.ai Error: {str(e)}]"

    def _generate_ollama(self, prompt):
        """
        Hypothetical Ollama usage.
        """
        try:
            endpoint = f"{self.ollama_endpoint}/generate"
            payload = {"prompt": prompt, "model": "llama2"}
            resp = requests.post(endpoint, json=payload)
            if resp.status_code == 200:
                data = resp.json()
                return data.get("generated_text", "[No Ollama response]")
            else:
                return f"[Ollama Error: {resp.status_code} {resp.text}]"
        except Exception as e:
            return f"[Ollama Error: {str(e)}]"
