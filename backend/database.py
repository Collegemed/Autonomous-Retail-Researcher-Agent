"""
database.py - SQLite persistence layer.

Tables:
  - research_queries: stores all user queries + results
  - query_cache: short-lived cache to avoid repeat LLM calls

All functions use thread-safe connections.
"""

import sqlite3
import json
import logging
import time
from typing import Optional, List, Dict
from config import config

logger = logging.getLogger("database")


def _get_conn() -> sqlite3.Connection:
    """Return a thread-local SQLite connection with row factory."""
    conn = sqlite3.connect(config.DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")  # better concurrency
    return conn


def init_db():
    """Create tables if they don't exist. Safe to call on every startup."""
    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS research_queries (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                query       TEXT    NOT NULL,
                result_json TEXT    NOT NULL,
                created_at  REAL    NOT NULL DEFAULT (unixepoch('now'))
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_query_time
            ON research_queries (created_at DESC)
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS query_cache (
                query_hash  TEXT PRIMARY KEY,
                query       TEXT NOT NULL,
                result_json TEXT NOT NULL,
                expires_at  REAL NOT NULL
            )
        """)
        conn.commit()
    logger.info("Database initialized")


def save_query(query: str, result: dict):
    """Persist a completed research result to the database."""
    try:
        with _get_conn() as conn:
            conn.execute(
                "INSERT INTO research_queries (query, result_json, created_at) VALUES (?, ?, ?)",
                (query, json.dumps(result), time.time()),
            )
            conn.commit()
        logger.info(f"Saved query to DB: {query[:60]}")
    except Exception as e:
        logger.error(f"DB save failed: {e}")


def get_history(limit: int = 20, offset: int = 0) -> List[Dict]:
    """Return paginated list of past queries from newest to oldest."""
    try:
        with _get_conn() as conn:
            rows = conn.execute(
                """
                SELECT id, query, result_json, created_at
                FROM research_queries
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()

        history = []
        for row in rows:
            try:
                result = json.loads(row["result_json"])
            except json.JSONDecodeError:
                result = {}
            history.append({
                "id": row["id"],
                "query": row["query"],
                "result": result,
                "created_at": row["created_at"],
            })
        return history

    except Exception as e:
        logger.error(f"DB history fetch failed: {e}")
        return []


# ─── Cache ─────────────────────────────────────────────────────────────────────

def _query_hash(query: str) -> str:
    """Simple normalized hash for cache key."""
    import hashlib
    normalized = " ".join(query.lower().split())
    return hashlib.sha256(normalized.encode()).hexdigest()


def get_cached_result(query: str) -> Optional[Dict]:
    """Return cached result if it exists and hasn't expired."""
    try:
        qhash = _query_hash(query)
        with _get_conn() as conn:
            row = conn.execute(
                "SELECT result_json, expires_at FROM query_cache WHERE query_hash = ?",
                (qhash,),
            ).fetchone()

        if row and row["expires_at"] > time.time():
            return json.loads(row["result_json"])

        # Expired — delete it
        if row:
            with _get_conn() as conn:
                conn.execute("DELETE FROM query_cache WHERE query_hash = ?", (qhash,))
                conn.commit()

        return None

    except Exception as e:
        logger.error(f"Cache get failed: {e}")
        return None


def save_cache(query: str, result: dict):
    """Store a result in the cache with TTL expiry."""
    try:
        qhash = _query_hash(query)
        expires_at = time.time() + config.CACHE_TTL_SECONDS
        with _get_conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO query_cache
                    (query_hash, query, result_json, expires_at)
                VALUES (?, ?, ?, ?)
                """,
                (qhash, query, json.dumps(result), expires_at),
            )
            conn.commit()
    except Exception as e:
        logger.error(f"Cache save failed: {e}")


def clear_cache():
    """Delete all cache entries."""
    try:
        with _get_conn() as conn:
            conn.execute("DELETE FROM query_cache")
            conn.commit()
        logger.info("Cache cleared")
    except Exception as e:
        logger.error(f"Cache clear failed: {e}")