import json
import logging
import uuid
from datetime import datetime, timezone

import aiosqlite

logger = logging.getLogger(__name__)


async def _migrate_research_runs_add_status(db) -> None:
    await db.execute("""
        CREATE TABLE research_runs_new (
            id TEXT PRIMARY KEY,
            normalized_url TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'running',
            result_payload TEXT,
            checked_at TEXT,
            created_at TEXT NOT NULL
        )
    """)
    await db.execute("""
        INSERT INTO research_runs_new
            (id, normalized_url, status, result_payload, checked_at, created_at)
        SELECT id, normalized_url, 'completed', result_payload, checked_at, created_at
        FROM research_runs
    """)
    await db.execute("DROP TABLE research_runs")
    await db.execute("ALTER TABLE research_runs_new RENAME TO research_runs")
    await db.execute("""
        CREATE INDEX IF NOT EXISTS idx_research_runs_normalized_url
        ON research_runs(normalized_url)
    """)
    logger.info("Migrated research_runs to add status column")


async def init_db(db_path: str) -> None:
    async with aiosqlite.connect(db_path) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS research_runs (
                id TEXT PRIMARY KEY,
                normalized_url TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'running',
                result_payload TEXT,
                checked_at TEXT,
                created_at TEXT NOT NULL
            )
        """)
        cursor = await db.execute("PRAGMA table_info(research_runs)")
        rows = await cursor.fetchall()
        columns = {row[1] for row in rows}
        if rows and "status" not in columns:
            await _migrate_research_runs_add_status(db)
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


async def create_research_run(db_path: str, normalized_url: str) -> str:
    run_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc)
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            """
            INSERT INTO research_runs (id, normalized_url, status, created_at)
            VALUES (?, ?, 'running', ?)
            """,
            (run_id, normalized_url, created_at.isoformat()),
        )
        await db.commit()
    logger.info("Research run created: run_id=%s url=%s", run_id, normalized_url)
    return run_id


async def complete_research_run(
    db_path: str,
    run_id: str,
    result_payload: dict,
    checked_at: datetime,
) -> None:
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            """
            UPDATE research_runs
            SET status = 'completed', result_payload = ?, checked_at = ?
            WHERE id = ?
            """,
            (json.dumps(result_payload), checked_at.isoformat(), run_id),
        )
        await db.commit()
    logger.info("Research run completed: run_id=%s", run_id)


async def mark_research_run_failed(db_path: str, run_id: str) -> None:
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "UPDATE research_runs SET status = 'failed' WHERE id = ?",
            (run_id,),
        )
        await db.commit()
    logger.info("Research run marked failed: run_id=%s", run_id)


async def load_research_run(db_path: str, run_id: str) -> dict | None:
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute(
            "SELECT id, status, result_payload, checked_at FROM research_runs WHERE id = ?",
            (run_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        result_payload = json.loads(row[2]) if row[2] is not None else None
        return {
            "id": row[0],
            "status": row[1],
            "result_payload": result_payload,
            "checked_at": row[3],
        }


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
