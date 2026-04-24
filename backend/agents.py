"""
agents.py - Multi-Agent System for Autonomous Retail Research.

Agents:
  1. ResearchAgent    → queries Tavily for real-time web data
  2. AnalysisAgent    → cleans, filters, and structures raw results
  3. SummaryAgent     → generates final structured insights via LLM
  4. StorageAgent     → coordinates final output (DB + RAG handled externally)

Orchestrator:
  ResearchOrchestrator → runs agents in collaboration, passes state between them
  Each agent receives the output of the previous one.
"""

import logging
import asyncio
from typing import Dict, List, Any
from concurrent.futures import ThreadPoolExecutor

from tools import TavilySearchTool, TextCleaner, TokenSafeChunker
from memory import LLMProvider
from rag import RAGMemory
from config import config

logger = logging.getLogger("agents")

# Shared thread pool for sync → async bridging
_executor = ThreadPoolExecutor(max_workers=4)


# ─── Base Agent ───────────────────────────────────────────────────────────────

class BaseAgent:
    """Base class for all agents. Each agent has a role, goal, and run() method."""

    def __init__(self, name: str, role: str, llm: LLMProvider):
        self.name = name
        self.role = role
        self.llm = llm
        self.logger = logging.getLogger(f"agent.{name}")

    def log(self, msg: str):
        self.logger.info(f"[{self.name}] {msg}")

    async def run_async(self, *args, **kwargs) -> Any:
        """Run the synchronous run() method in a thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_executor, lambda: self.run(*args, **kwargs))

    def run(self, *args, **kwargs) -> Any:
        raise NotImplementedError


# ─── Agent 1: Research Agent ──────────────────────────────────────────────────

class ResearchAgent(BaseAgent):
    """
    Retrieves real-time retail data from the web using Tavily Search.
    Expands the query into multiple sub-queries for broader coverage.
    """

    SYSTEM_PROMPT = """You are a retail research query expansion specialist.
Given a user's retail research query, generate 2-3 precise sub-queries
that together provide comprehensive coverage of the topic.
Return ONLY a JSON array of strings. Example: ["query1", "query2", "query3"]
Do not include any explanation or markdown."""

    def __init__(self, llm: LLMProvider):
        super().__init__("ResearchAgent", "Web Data Retrieval Specialist", llm)
        self.search_tool = TavilySearchTool()

    def run(self, query: str) -> Dict:
        """
        1. Expand query into sub-queries via LLM
        2. Search each sub-query with Tavily
        3. Combine and return raw results
        """
        self.log(f"Starting research for: {query[:60]}")

        # Step 1: Expand query
        sub_queries = self._expand_query(query)
        self.log(f"Expanded to {len(sub_queries)} sub-queries: {sub_queries}")

        # Step 2: Search each sub-query
        all_results = []
        seen_urls = set()

        for sub_q in sub_queries:
            try:
                results = self.search_tool.search(sub_q)
                for r in results:
                    url = r.get("url", "")
                    if url not in seen_urls:
                        seen_urls.add(url)
                        all_results.append(r)
            except Exception as e:
                self.log(f"Sub-query search failed for '{sub_q}': {e}")

        self.log(f"Retrieved {len(all_results)} unique results")

        return {
            "original_query": query,
            "sub_queries": sub_queries,
            "raw_results": all_results,
            "result_count": len(all_results),
        }

    def _expand_query(self, query: str) -> List[str]:
        """Use LLM to generate multiple search sub-queries."""
        try:
            import json
            response = self.llm.call(
                system_prompt=self.SYSTEM_PROMPT,
                user_prompt=f"Retail research query: {query}",
                max_tokens=200,
            )
            # Parse JSON array
            cleaned = response.strip()
            if cleaned.startswith("["):
                sub_queries = json.loads(cleaned)
                if isinstance(sub_queries, list) and len(sub_queries) > 0:
                    return sub_queries[:3]  # max 3 sub-queries
        except Exception as e:
            self.log(f"Query expansion failed: {e}. Using original query.")

        return [query]


# ─── Agent 2: Analysis Agent ─────────────────────────────────────────────────

class AnalysisAgent(BaseAgent):
    """
    Takes raw search results and:
    1. Cleans text noise
    2. Filters low-quality results
    3. Scores and ranks results by relevance
    4. Structures data for the Summary Agent
    """

    RELEVANCE_SYSTEM = """You are a retail data analyst. Given a search result snippet and query,
rate the relevance on a scale 1-10. Return ONLY a JSON object like:
{"score": 8, "key_fact": "one key insight in under 15 words"}
No explanation, no markdown."""

    MIN_CONTENT_LENGTH = 50  # chars
    MIN_RELEVANCE_SCORE = 0.2  # Tavily score threshold

    def __init__(self, llm: LLMProvider):
        super().__init__("AnalysisAgent", "Data Analysis and Filtering Specialist", llm)
        self.cleaner = TextCleaner()
        self.chunker = TokenSafeChunker()

    def run(self, research_output: Dict) -> Dict:
        """
        Filter, clean, and score raw search results.
        Returns structured data ready for summarization.
        """
        query = research_output["original_query"]
        raw_results = research_output["raw_results"]
        self.log(f"Analyzing {len(raw_results)} raw results")

        # Step 1: Clean + filter by quality thresholds
        cleaned = []
        for r in raw_results:
            content = self.cleaner.clean(r.get("content", ""))
            if len(content) < self.MIN_CONTENT_LENGTH:
                continue
            if r.get("score", 0) < self.MIN_RELEVANCE_SCORE and r.get("source") == "web":
                continue

            cleaned.append({
                "title": r.get("title", "Untitled"),
                "url": r.get("url", ""),
                "content": self.cleaner.truncate(content, 600),
                "tavily_score": r.get("score", 0),
                "source": r.get("source", "web"),
            })

        self.log(f"After filtering: {len(cleaned)} quality results")

        # Step 2: Combine text for LLM context
        combined_text = self.cleaner.combine_results(cleaned, max_total_chars=4000)
        safe_text = self.chunker.fit(combined_text, max_tokens=config.MAX_INPUT_TOKENS)

        # Step 3: Extract key themes via LLM
        key_themes = self._extract_themes(query, safe_text)

        return {
            "original_query": query,
            "sub_queries": research_output.get("sub_queries", [query]),
            "filtered_results": cleaned,
            "combined_text": safe_text,
            "key_themes": key_themes,
            "source_count": len(cleaned),
        }

    def _extract_themes(self, query: str, text: str) -> List[str]:
        """Use LLM to identify key themes from the search data."""
        system = """You are a retail analyst. Extract 4-6 key themes or trends from the text.
Return ONLY a JSON array of short strings (each under 10 words).
Example: ["theme 1", "theme 2", "theme 3"]"""

        try:
            import json
            response = self.llm.call(
                system_prompt=system,
                user_prompt=f"Query: {query}\n\nData:\n{text[:2000]}",
                max_tokens=200,
            )
            cleaned = response.strip()
            if "[" in cleaned:
                start = cleaned.index("[")
                themes = json.loads(cleaned[start:])
                if isinstance(themes, list):
                    return themes[:6]
        except Exception as e:
            self.log(f"Theme extraction failed: {e}")

        return ["Retail market analysis", "Consumer trends", "Competitive landscape"]


# ─── Agent 3: Summary Agent ───────────────────────────────────────────────────

class SummaryAgent(BaseAgent):
    """
    Generates a structured research report using:
    - Cleaned + filtered data from AnalysisAgent
    - Relevant past research injected via RAG context
    """

    SYSTEM_PROMPT = """You are a senior retail research analyst. Generate a comprehensive,
structured research report based on the provided data and context.

Your report MUST include:
1. Executive Summary (2-3 sentences)
2. Key Findings (4-6 bullet points, each starting with "•")
3. Market Trends (3-4 current trends)
4. Competitive Intelligence (2-3 observations)
5. Strategic Recommendations (3-4 actionable items)

Rules:
- Be factual, cite data points where available
- Do NOT hallucinate statistics without source basis
- Keep each section concise but substantive
- Use professional business language
- Format output as clean structured text (no markdown headers, use section labels)"""

    def __init__(self, llm: LLMProvider, rag: RAGMemory):
        super().__init__("SummaryAgent", "Research Report Generation Specialist", llm)
        self.rag = rag

    def run(self, analysis_output: Dict) -> Dict:
        """
        Generate the final structured research report.
        Injects RAG context from past similar research.
        """
        query = analysis_output["original_query"]
        combined_text = analysis_output["combined_text"]
        key_themes = analysis_output["key_themes"]
        source_count = analysis_output["source_count"]

        self.log(f"Generating summary for: {query[:60]}")

        # Retrieve relevant past research from RAG memory
        rag_context = self.rag.get_context(query)
        if rag_context:
            self.log("RAG context injected from past research")

        # Build the full LLM prompt
        user_prompt = self._build_prompt(query, combined_text, key_themes, rag_context)

        # Generate the report
        report_text = self.llm.call(
            system_prompt=self.SYSTEM_PROMPT,
            user_prompt=user_prompt,
            max_tokens=config.MAX_OUTPUT_TOKENS,
        )

        # Parse the report into structured sections
        sections = self._parse_report(report_text)
        sources = self._extract_sources(analysis_output["filtered_results"])

        self.log("Summary generation complete")

        return {
            "original_query": query,
            "summary": report_text,
            "sections": sections,
            "key_themes": key_themes,
            "sources": sources,
            "source_count": source_count,
            "rag_context_used": bool(rag_context),
            "sub_queries_used": analysis_output.get("sub_queries", []),
        }

    def _build_prompt(self, query: str, text: str,
                      themes: List[str], rag_context: str) -> str:
        parts = [f"Research Query: {query}\n"]

        if rag_context:
            parts.append(f"{rag_context}\n")

        parts.append(f"Key Themes Identified: {', '.join(themes)}\n")
        parts.append(f"Research Data:\n{text}")

        return "\n".join(parts)

    def _parse_report(self, report: str) -> Dict[str, str]:
        """
        Extract named sections from the report text.
        Returns a dict mapping section names to content.
        """
        sections = {}
        section_keywords = [
            "Executive Summary",
            "Key Findings",
            "Market Trends",
            "Competitive Intelligence",
            "Strategic Recommendations",
        ]

        lines = report.split("\n")
        current_section = "Overview"
        current_content = []

        for line in lines:
            matched = False
            for keyword in section_keywords:
                if keyword.lower() in line.lower() and len(line) < 60:
                    if current_content:
                        sections[current_section] = "\n".join(current_content).strip()
                    current_section = keyword
                    current_content = []
                    matched = True
                    break
            if not matched:
                current_content.append(line)

        if current_content:
            sections[current_section] = "\n".join(current_content).strip()

        return sections

    def _extract_sources(self, results: List[Dict]) -> List[Dict]:
        """Extract source metadata for citation display."""
        sources = []
        for r in results[:5]:  # top 5 sources
            if r.get("url"):
                sources.append({
                    "title": r.get("title", "Source"),
                    "url": r.get("url", ""),
                    "score": round(r.get("tavily_score", 0), 3),
                })
        return sources


# ─── Agent 4: Storage Agent ───────────────────────────────────────────────────

class StorageAgent(BaseAgent):
    """
    Coordinates final storage:
    - Formats output for API response
    - Triggers DB + RAG storage (done in main.py after this returns)
    - Validates and enriches final output
    """

    def __init__(self, llm: LLMProvider):
        super().__init__("StorageAgent", "Result Storage and Output Coordinator", llm)

    def run(self, summary_output: Dict) -> Dict:
        """
        Validate and format the final output structure.
        """
        self.log("Formatting final output")

        result = {
            "query": summary_output["original_query"],
            "summary": summary_output.get("summary", ""),
            "sections": summary_output.get("sections", {}),
            "key_themes": summary_output.get("key_themes", []),
            "sources": summary_output.get("sources", []),
            "source_count": summary_output.get("source_count", 0),
            "sub_queries_used": summary_output.get("sub_queries_used", []),
            "rag_context_used": summary_output.get("rag_context_used", False),
            "status": "success",
        }

        # Validate we have actual content
        if not result["summary"] or len(result["summary"]) < 50:
            result["status"] = "partial"
            result["warning"] = "Summary may be incomplete"

        self.log("Output formatted and validated")
        return result


# ─── Orchestrator ─────────────────────────────────────────────────────────────

class ResearchOrchestrator:
    """
    Coordinates all four agents in sequence:
      ResearchAgent → AnalysisAgent → SummaryAgent → StorageAgent

    Each agent receives the output of the previous one.
    The orchestrator handles errors, timeouts, and agent status tracking.
    """

    def __init__(self, rag: RAGMemory):
        self.rag = rag
        self.llm = LLMProvider()

        # Instantiate agents
        self.research_agent = ResearchAgent(self.llm)
        self.analysis_agent = AnalysisAgent(self.llm)
        self.summary_agent = SummaryAgent(self.llm, self.rag)
        self.storage_agent = StorageAgent(self.llm)

        logger.info("ResearchOrchestrator initialized with 4 agents")

    async def run(self, query: str) -> Dict:
        """
        Execute the full multi-agent research pipeline asynchronously.
        Returns the final structured result dict.
        """
        logger.info(f"Orchestrator starting pipeline for: {query[:60]}")

        pipeline_start = {}  # shared state between agents

        try:
            # ── Agent 1: Research ─────────────────────────────────────────────
            logger.info("Running Agent 1: ResearchAgent")
            research_output = await asyncio.wait_for(
                self.research_agent.run_async(query),
                timeout=config.REQUEST_TIMEOUT,
            )

            # ── Agent 2: Analysis ─────────────────────────────────────────────
            logger.info("Running Agent 2: AnalysisAgent")
            analysis_output = await asyncio.wait_for(
                self.analysis_agent.run_async(research_output),
                timeout=config.REQUEST_TIMEOUT,
            )

            # ── Agent 3: Summary ──────────────────────────────────────────────
            logger.info("Running Agent 3: SummaryAgent")
            summary_output = await asyncio.wait_for(
                self.summary_agent.run_async(analysis_output),
                timeout=config.REQUEST_TIMEOUT,
            )

            # ── Agent 4: Storage ──────────────────────────────────────────────
            logger.info("Running Agent 4: StorageAgent")
            final_output = await asyncio.wait_for(
                self.storage_agent.run_async(summary_output),
                timeout=10,
            )

            logger.info("Pipeline completed successfully")
            return final_output

        except asyncio.TimeoutError:
            raise TimeoutError(f"Pipeline timed out after {config.REQUEST_TIMEOUT}s")
        except Exception as e:
            logger.exception(f"Pipeline error: {e}")
            return {
                "query": query,
                "summary": f"Research failed: {str(e)}",
                "sections": {},
                "key_themes": [],
                "sources": [],
                "source_count": 0,
                "status": "error",
                "error": str(e),
            }