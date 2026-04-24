"""
memory.py - LLM provider abstraction layer.

Supports Groq, OpenAI, and Anthropic.
Provides a unified call() interface with retry logic and token limiting.
"""

import logging
import time
from typing import Optional
from config import config

logger = logging.getLogger("llm")


class LLMProvider:
    """
    Unified LLM interface. Selects provider based on config.LLM_PROVIDER.
    Implements retry logic, token limiting, and error handling.
    """

    def __init__(self):
        self.provider = config.LLM_PROVIDER.lower()
        self.max_retries = config.MAX_RETRIES
        self._client = None
        self._init_client()

    def _init_client(self):
        """Initialize the appropriate LLM client."""
        try:
            if self.provider == "groq":
                from groq import Groq
                self._client = Groq(api_key=config.GROQ_API_KEY)
                self.model = config.GROQ_MODEL
                logger.info(f"Using Groq: {self.model}")

            elif self.provider == "openai":
                from openai import OpenAI
                self._client = OpenAI(api_key=config.OPENAI_API_KEY)
                self.model = config.OPENAI_MODEL
                logger.info(f"Using OpenAI: {self.model}")

            elif self.provider == "anthropic":
                import anthropic
                self._client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
                self.model = config.ANTHROPIC_MODEL
                logger.info(f"Using Anthropic: {self.model}")

            else:
                raise ValueError(f"Unknown LLM provider: {self.provider}")

        except ImportError as e:
            logger.error(f"LLM package not installed: {e}")
            self._client = None

    def call(self, system_prompt: str, user_prompt: str,
             max_tokens: Optional[int] = None) -> str:
        """
        Send a prompt to the LLM and return the text response.
        Retries on rate limits and transient errors.
        """
        max_tokens = max_tokens or config.MAX_OUTPUT_TOKENS

        if self._client is None:
            logger.warning("LLM client not initialized. Returning stub response.")
            return self._stub_response(user_prompt)

        for attempt in range(1, self.max_retries + 1):
            try:
                if self.provider == "groq":
                    return self._call_groq(system_prompt, user_prompt, max_tokens)
                elif self.provider == "openai":
                    return self._call_openai(system_prompt, user_prompt, max_tokens)
                elif self.provider == "anthropic":
                    return self._call_anthropic(system_prompt, user_prompt, max_tokens)

            except Exception as e:
                err_str = str(e).lower()
                logger.warning(f"LLM attempt {attempt} failed: {e}")

                # Handle rate limits
                if "rate_limit" in err_str or "429" in err_str:
                    wait = 10 * attempt
                    logger.info(f"Rate limited. Waiting {wait}s...")
                    time.sleep(wait)
                    continue

                if attempt == self.max_retries:
                    logger.error(f"LLM failed after {self.max_retries} attempts")
                    return f"[LLM Error] Unable to generate response: {str(e)}"

                time.sleep(2 ** attempt)

        return "[LLM Error] Max retries exceeded"

    def _call_groq(self, system: str, user: str, max_tokens: int) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
            temperature=0.3,  # lower temp = more factual, less hallucination
        )
        return response.choices[0].message.content.strip()

    def _call_openai(self, system: str, user: str, max_tokens: int) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()

    def _call_anthropic(self, system: str, user: str, max_tokens: int) -> str:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return response.content[0].text.strip()

    def _stub_response(self, prompt: str) -> str:
        """Fallback response when no LLM key is configured."""
        return (
            "No LLM provider configured. "
            "Please set GROQ_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY in your .env file. "
            f"Would have processed: {prompt[:100]}..."
        )