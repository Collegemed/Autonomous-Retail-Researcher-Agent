"""
rag.py - RAG (Retrieval-Augmented Generation) memory using FAISS + sentence-transformers.

Flow:
  1. store(query, summary) → encodes text → adds to FAISS index
  2. retrieve(query) → finds top-K similar past entries
  3. get_context(query) → returns formatted context for LLM prompt injection
"""

import os
import json
import logging
import pickle
from typing import List, Dict, Optional
from config import config

logger = logging.getLogger("rag")


class RAGMemory:
    """
    Vector-based memory using FAISS for similarity search.
    Embeds research summaries and retrieves relevant past context
    to augment new LLM queries (RAG pattern).
    """

    def __init__(self):
        self.index_path = config.FAISS_INDEX_PATH
        self.top_k = config.RAG_TOP_K
        self.embedding_model = None
        self.index = None
        self.metadata: List[Dict] = []  # stores {query, summary} per vector

        self._init_embeddings()
        self._load_or_create_index()

    def _init_embeddings(self):
        """Load sentence-transformer embedding model."""
        try:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
            logger.info(f"Embedding model loaded: {config.EMBEDDING_MODEL}")
        except ImportError:
            logger.warning("sentence-transformers not installed. RAG disabled.")
        except Exception as e:
            logger.error(f"Embedding model init failed: {e}")

    def _load_or_create_index(self):
        """Load existing FAISS index from disk or create a new empty one."""
        if self.embedding_model is None:
            return

        try:
            import faiss
            dim = self.embedding_model.get_sentence_embedding_dimension()

            meta_file = f"{self.index_path}.meta.pkl"
            idx_file = f"{self.index_path}.faiss"

            if os.path.exists(idx_file) and os.path.exists(meta_file):
                self.index = faiss.read_index(idx_file)
                with open(meta_file, "rb") as f:
                    self.metadata = pickle.load(f)
                logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")
            else:
                # L2 flat index (exact search, good for small-medium datasets)
                self.index = faiss.IndexFlatL2(dim)
                self.metadata = []
                logger.info(f"Created new FAISS index (dim={dim})")

        except ImportError:
            logger.warning("faiss-cpu not installed. RAG disabled.")
        except Exception as e:
            logger.error(f"FAISS init failed: {e}")

    def store(self, query: str, summary: str):
        """
        Encode query+summary and add to the FAISS index.
        Also persists the index to disk so it survives restarts.
        """
        if not self._is_ready():
            return

        try:
            import numpy as np

            text = f"Query: {query}\nSummary: {summary}"
            embedding = self.embedding_model.encode([text], convert_to_numpy=True)
            embedding = embedding.astype("float32")

            self.index.add(embedding)
            self.metadata.append({"query": query, "summary": summary})

            self._save_index()
            logger.info(f"Stored in RAG ({self.index.ntotal} total vectors)")

        except Exception as e:
            logger.error(f"RAG store failed: {e}")

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        """
        Find top-K most similar past research entries for the given query.
        Returns list of {query, summary, distance} dicts.
        """
        if not self._is_ready() or self.index.ntotal == 0:
            return []

        top_k = top_k or self.top_k

        try:
            import numpy as np

            embedding = self.embedding_model.encode([query], convert_to_numpy=True)
            embedding = embedding.astype("float32")

            k = min(top_k, self.index.ntotal)
            distances, indices = self.index.search(embedding, k)

            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx < len(self.metadata):
                    entry = self.metadata[idx].copy()
                    entry["distance"] = float(dist)
                    results.append(entry)

            logger.info(f"RAG retrieved {len(results)} relevant memories")
            return results

        except Exception as e:
            logger.error(f"RAG retrieve failed: {e}")
            return []

    def get_context(self, query: str) -> str:
        """
        Build a formatted context string from retrieved memories.
        This gets injected into the LLM prompt as prior knowledge.
        """
        memories = self.retrieve(query)
        if not memories:
            return ""

        lines = ["=== Relevant Past Research (from memory) ==="]
        for i, m in enumerate(memories, 1):
            lines.append(f"\n[Memory {i}]")
            lines.append(f"Previous Query: {m['query']}")
            lines.append(f"Summary: {m['summary'][:400]}")

        lines.append("=" * 44)
        return "\n".join(lines)

    def _save_index(self):
        """Persist FAISS index and metadata to disk."""
        try:
            import faiss
            faiss.write_index(self.index, f"{self.index_path}.faiss")
            with open(f"{self.index_path}.meta.pkl", "wb") as f:
                pickle.dump(self.metadata, f)
        except Exception as e:
            logger.error(f"FAISS save failed: {e}")

    def _is_ready(self) -> bool:
        return self.embedding_model is not None and self.index is not None