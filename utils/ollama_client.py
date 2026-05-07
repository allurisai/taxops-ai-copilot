import os
import requests

DEFAULT_OLLAMA_MODEL = "llama3.2:3b"
OLLAMA_URL = "http://localhost:11434/api/generate"


def _query_ollama_local(prompt, model=DEFAULT_OLLAMA_MODEL):
    response = requests.post(
        OLLAMA_URL,
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=180,
    )
    response.raise_for_status()
    response_data = response.json()
    if response_data.get("error"):
        raise RuntimeError(response_data["error"])
    model_response = response_data.get("response", "").strip()
    if not model_response:
        raise RuntimeError("Ollama returned an empty response.")
    return model_response


def _query_cloud_fallback(prompt):
    import streamlit as st

    api_key = ""
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    if not api_key:
        api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "Ollama is not running and no GROQ_API_KEY is set. "
            "Please run: ollama serve"
        )

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": "llama-3.1-8b-instant",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1024,
        },
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


def query_ollama(prompt, model=DEFAULT_OLLAMA_MODEL):
    try:
        return _query_ollama_local(prompt, model)
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        pass
    except requests.exceptions.RequestException:
        pass

    try:
        return _query_cloud_fallback(prompt)
    except Exception as cloud_error:
        raise RuntimeError(
            f"Could not reach Ollama locally or the cloud API. Error: {cloud_error}"
        ) from cloud_error
