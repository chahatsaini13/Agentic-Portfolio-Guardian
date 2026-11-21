"""
Single entry point for every LLM call in the project.

LLM_PROVIDER=ollama (default) -> local Ollama server.
LLM_PROVIDER=api              -> hosted API (not implemented yet).
"""

import os

import requests
from dotenv import load_dotenv

load_dotenv()

OLLAMA_URL = "http://localhost:11434/api/generate"


def call_llm(prompt: str, json_mode: bool = False,
             temperature=None, num_predict=None) -> str:
    provider = os.environ.get("LLM_PROVIDER", "ollama").lower()

    if provider == "ollama":
        return _call_ollama(prompt, json_mode, temperature, num_predict)
    if provider == "api":
        raise NotImplementedError("LLM_PROVIDER=api is not implemented yet.")
    raise ValueError(f"Unknown LLM_PROVIDER: {provider!r} (use 'ollama' or 'api')")


def _call_ollama(prompt, json_mode, temperature, num_predict) -> str:
    model = os.environ.get("OLLAMA_MODEL", "llama3.1")
    options = {}
    if temperature is not None:
        options["temperature"] = temperature
    if num_predict is not None:
        options["num_predict"] = num_predict

    body = {"model": model, "prompt": prompt, "stream": False, "options": options}
    if json_mode:
        body["format"] = "json"

    resp = requests.post(OLLAMA_URL, json=body, timeout=300)
    resp.raise_for_status()
    return resp.json()["response"]
