"""
tools.py - External tool integrations used by agents.

Includes:
  - TavilySearchTool: real-time web search with retry + timeout
  - TextCleaner: strips noise from raw search results
  - TokenSafeChunker: splits text to fit LLM context windows
"""

import logging
import time
import re
from typing import List, Dict, Optional
import httpx
from config import config

logger = logging.getLogger("tools")


# ─── Tavily Web Search ────────────────────────────────────────────────────────

class TavilySearchTool:
    """
    Wraps the Tavily Search API with retry logic, timeout handling,
    and result normalization.
    """

    BASE_URL = "https://api.tavily.com/search"

    def __init__(self):
        self.api_key = config.TAVILY_API_KEY
        self.max_results = config.SEARCH_MAX_RESULTS
        self.timeout = config.SEARCH_TIMEOUT
        self.max_retries = config.MAX_RETRIES

    def search(self, query: str, search_depth: str = "advanced") -> List[Dict]:
        """
        Perform a web search using Tavily API.
        Returns a list of normalized result dicts.
        """
        if not self.api_key:
            logger.warning("TAVILY_API_KEY not set. Returning mock results.")
            return self._mock_results(query)

        payload = {
            "api_key": self.api_key,
            "query": query,
            "search_depth": search_depth,
            "max_results": self.max_results,
            "include_answer": True,
            "include_raw_content": False,
        }

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(f"Tavily search attempt {attempt}: {query[:60]}")
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(self.BASE_URL, json=payload)
                    response.raise_for_status()
                    data = response.json()
                    return self._normalize(data)

            except httpx.TimeoutException:
                logger.warning(f"Tavily timeout on attempt {attempt}")
                if attempt == self.max_retries:
                    raise TimeoutError(f"Tavily search timed out after {self.max_retries} attempts")
                time.sleep(2 ** attempt)  # exponential backoff

            except httpx.HTTPStatusError as e:
                logger.error(f"Tavily HTTP error: {e.response.status_code}")
                if e.response.status_code == 429:
                    time.sleep(5)
                    continue
                raise

            except Exception as e:
                logger.exception(f"Tavily unexpected error: {e}")
                if attempt == self.max_retries:
                    raise
                time.sleep(2)

        return []

    def _normalize(self, data: dict) -> List[Dict]:
        """Convert raw Tavily response to clean list of results."""
        results = []

        # Include the Tavily-generated answer if present
        if data.get("answer"):
            results.append({
                "title": "Tavily Answer",
                "url": "",
                "content": data["answer"],
                "score": 1.0,
                "source": "tavily_answer",
            })

        for item in data.get("results", []):
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "content": item.get("content", ""),
                "score": item.get("score", 0.0),
                "source": "web",
            })

        return results

    def _mock_results(self, query: str) -> List[Dict]:
        """Fallback mock results when no API key is configured."""
        return [
            {
                "title": f"Mock Result for: {query}",
                "url": "https://example.com/retail-news",
                "content": (
                    f"This is a simulated search result for '{query}'. "
                    "Configure TAVILY_API_KEY in your .env file to get real results. "
                    "Key retail trends: e-commerce growth, supply chain optimization, "
                    "AI-driven personalization, sustainability initiatives, "
                    "omnichannel strategies, and competitive pricing intelligence."
                ),
                "score": 0.9,
                "source": "mock",
            }
        ]


# ─── Text Cleaner ─────────────────────────────────────────────────────────────

class TextCleaner:
    """Cleans and normalizes raw text from web search results."""

    @staticmethod
    def clean(text: str) -> str:
        """Remove noise, extra whitespace, and boilerplate from text."""
        if not text:
            return ""

        # Remove HTML tags
        text = re.sub(r"<[^>]+>", " ", text)

        # Remove URLs inline
        text = re.sub(r"https?://\S+", "", text)

        # Remove special characters but keep punctuation
        text = re.sub(r"[^\w\s.,;:!?()\-–—\"']", " ", text)

        # Collapse multiple spaces/newlines
        text = re.sub(r"\s+", " ", text).strip()

        return text

    @staticmethod
    def truncate(text: str, max_chars: int = 800) -> str:
        """Truncate text to max_chars, ending at a sentence boundary if possible."""
        if len(text) <= max_chars:
            return text
        truncated = text[:max_chars]
        last_period = truncated.rfind(". ")
        if last_period > max_chars // 2:
            return truncated[:last_period + 1]
        return truncated + "..."

    @staticmethod
    def combine_results(results: List[Dict], max_total_chars: int = 4000) -> str:
        """Combine multiple search results into a single clean text block."""
        combined = []
        total = 0

        for r in results:
            title = r.get("title", "")
            content = TextCleaner.clean(r.get("content", ""))
            url = r.get("url", "")

            if not content:
                continue

            chunk = f"[{title}]\n{content}"
            if url:
                chunk += f"\nSource: {url}"

            if total + len(chunk) > max_total_chars:
                remaining = max_total_chars - total
                if remaining > 100:
                    combined.append(chunk[:remaining])
                break

            combined.append(chunk)
            total += len(chunk)

        return "\n\n---\n\n".join(combined)


# ─── Token-Safe Chunker ───────────────────────────────────────────────────────

class TokenSafeChunker:
    """
    Splits text into chunks that fit within LLM token limits.
    Approximation: 1 token ≈ 4 characters.
    """

    CHARS_PER_TOKEN = 4

    @classmethod
    def fit(cls, text: str, max_tokens: int = None) -> str:
        """Truncate text to fit within max_tokens."""
        max_tokens = max_tokens or config.MAX_INPUT_TOKENS
        max_chars = max_tokens * cls.CHARS_PER_TOKEN
        return TextCleaner.truncate(text, max_chars)