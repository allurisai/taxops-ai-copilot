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


def _query_google_cloud(prompt):
    try:
        import streamlit as st
        api_key = st.secrets.get("GOOGLE_API_KEY", os.environ.get("GOOGLE_API_KEY", ""))
    except:
        api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "Ollama is not running and no GOOGLE_API_KEY is set. "
            "Please run: ollama serve"
        )
    response = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}",
        headers={"content-type": "application/json"},
        json={
            "contents": [{"parts": [{"text": prompt}]}],
        },
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    return data["candidates"][0]["content"]["parts"][0]["text"].strip()


def query_ollama(prompt, model=DEFAULT_OLLAMA_MODEL):
    try:
        return _query_ollama_local(prompt, model)
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        pass
    except requests.exceptions.RequestException:
        pass

    try:
        return _query_google_cloud(prompt)
    except Exception as cloud_error:
        raise RuntimeError(
            "Could not reach Ollama locally or the cloud API. "
            "Please run: ollama serve"
        ) from cloud_error
