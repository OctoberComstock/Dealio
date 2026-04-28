import json
import logging
import uuid
from datetime import datetime, timezone

import aiosqlite

logger = logging.getLogger(__name__)


async def init_db(db_path: str) -> None:
    async with aiosqlite.connect(db_path) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS research_runs (
                id TEXT PRIMARY KEY,
                normalized_url TEXT NOT NULL,
                result_payload TEXT NOT NULL,
                checked_at TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_research_runs_normalized_url
            ON research_runs(normalized_url)
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS research_cache (
                id TEXT PRIMARY KEY,
                artifact_type TEXT NOT NULL,
                cache_key TEXT NOT NULL,
                payload TEXT NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL
            )
        """)
        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_research_cache_cache_key
            ON research_cache(cache_key)
        """)
        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_research_cache_expires_at
            ON research_cache(expires_at)
        """)
        await db.commit()


async def save_research_run(
    db_path: str,
    normalized_url: str,
    result_payload: dict,
    checked_at: datetime,
) -> str:
    run_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc)
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            """
            INSERT INTO research_runs (id, normalized_url, result_payload, checked_at, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                run_id,
                normalized_url,
                json.dumps(result_payload),
                checked_at.isoformat(),
                created_at.isoformat(),
            ),
        )
        await db.commit()
    logger.info("Research run saved: run_id=%s url=%s", run_id, normalized_url)
    return run_id


async def save_cache_artifact(
    db_path: str,
    artifact_type: str,
    cache_key: str,
    payload: dict,
    expires_at: datetime,
) -> str:
    artifact_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc)
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            """
            INSERT INTO research_cache
                (id, artifact_type, cache_key, payload, created_at, expires_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                artifact_id,
                artifact_type,
                cache_key,
                json.dumps(payload),
                created_at.isoformat(),
                expires_at.isoformat(),
            ),
        )
        await db.commit()
    return artifact_id


async def load_research_run(db_path: str, run_id: str) -> dict | None:
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute(
            "SELECT id, result_payload, checked_at FROM research_runs WHERE id = ?",
            (run_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return {"id": row[0], "result_payload": json.loads(row[1]), "checked_at": row[2]}


async def delete_expired_cache_artifacts(db_path: str) -> int:
    now = datetime.now(timezone.utc).isoformat()
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute(
            "DELETE FROM research_cache WHERE expires_at < ?",
            (now,),
        )
        await db.commit()
        count = cursor.rowcount
    if count:
        logger.info("Deleted %d expired cache artifacts", count)
    return count
