"""Gemini chat model with a local Ollama fallback."""

import logging
import os
from typing import Any

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableLambda

from .config import GEMINI_MODEL, OLLAMA_MODEL

load_dotenv()
logger = logging.getLogger(__name__)


def extract_message_text(response: Any) -> str:
    """Return stripped text from a string or LangChain response object."""
    content = getattr(response, "content", response)
    if isinstance(content, list):
        content = "".join(
            str(part.get("text", "")) if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content).strip()


def get_llm(model: str = GEMINI_MODEL, temperature: float = 0.0) -> Any:
    """Prefer Gemini and fall back to Ollama when Gemini is unavailable."""
    ollama_model = os.getenv("OLLAMA_MODEL", OLLAMA_MODEL)
    google_api_key = os.getenv("GOOGLE_API_KEY")
    ollama = ChatOllama(
        model=ollama_model,
        temperature=temperature,
    )
    if not google_api_key:
        logger.warning(
            "GOOGLE_API_KEY is not set; selecting local Ollama model '%s' directly.",
            ollama_model,
        )
        return ollama

    gemini = ChatGoogleGenerativeAI(
        model=model,
        google_api_key=google_api_key,
        temperature=temperature,
        convert_system_message_to_human=True,
    )
    def logged_ollama_fallback(prompt: Any) -> Any:
        logger.warning(
            "Gemini '%s' failed; falling back to local Ollama model '%s'.",
            model,
            ollama_model,
        )
        return ollama.invoke(prompt)

    return gemini.with_fallbacks([RunnableLambda(logged_ollama_fallback)])
