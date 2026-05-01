import json
import uuid
from datetime import datetime, timedelta, timezone

import aiosqlite
import pytest

from app.database import (
    complete_research_run,
    create_research_run,
    delete_expired_cache_artifacts,
    init_db,
    load_research_run,
    mark_research_run_failed,
    save_cache_artifact,
)

CHECKED_AT = datetime.now(timezone.utc) - timedelta(hours=1)
FUTURE_EXPIRES_AT = datetime.now(timezone.utc) + timedelta(days=1)


@pytest.fixture
async def db_path(tmp_path):
    path = str(tmp_path / "test.db")
    await init_db(path)
    return path


async def test_init_db_creates_research_runs_table(db_path):
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] async for row in cursor}
    assert "research_runs" in tables


async def test_init_db_creates_research_cache_table(db_path):
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] async for row in cursor}
    assert "research_cache" in tables


async def test_create_research_run_returns_uuid(db_path):
    run_id = await create_research_run(db_path, "https://example.com/product")
    assert uuid.UUID(run_id)


async def test_create_research_run_stores_uuid_as_id(db_path):
    run_id = await create_research_run(db_path, "https://example.com/product")
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute("SELECT id FROM research_runs")
        row = await cursor.fetchone()
    assert row[0] == run_id


async def test_create_research_run_persists_normalized_url(db_path):
    await create_research_run(db_path, "https://example.com/product")
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute("SELECT normalized_url FROM research_runs")
        row = await cursor.fetchone()
    assert row[0] == "https://example.com/product"


async def test_create_research_run_sets_status_running(db_path):
    await create_research_run(db_path, "https://example.com/product")
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute("SELECT status FROM research_runs")
        row = await cursor.fetchone()
    assert row[0] == "running"


async def test_create_research_run_has_null_result_payload(db_path):
    await create_research_run(db_path, "https://example.com/product")
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute("SELECT result_payload FROM research_runs")
        row = await cursor.fetchone()
    assert row[0] is None


async def test_create_research_run_has_null_checked_at(db_path):
    await create_research_run(db_path, "https://example.com/product")
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute("SELECT checked_at FROM research_runs")
        row = await cursor.fetchone()
    assert row[0] is None


async def test_create_research_run_persists_created_at(db_path):
    await create_research_run(db_path, "https://example.com/product")
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute("SELECT created_at FROM research_runs")
        row = await cursor.fetchone()
    assert row[0] is not None and row[0] != ""


async def test_complete_research_run_sets_status_completed(db_path):
    payload = {"verdict": "good_deal"}
    run_id = await create_research_run(db_path, "https://example.com/product")
    await complete_research_run(db_path, run_id, payload, CHECKED_AT)
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute("SELECT status FROM research_runs WHERE id = ?", (run_id,))
        row = await cursor.fetchone()
    assert row[0] == "completed"


async def test_complete_research_run_stores_result_payload(db_path):
    payload = {"verdict": "good_deal", "confidence": "high"}
    run_id = await create_research_run(db_path, "https://example.com/product")
    await complete_research_run(db_path, run_id, payload, CHECKED_AT)
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute(
            "SELECT result_payload FROM research_runs WHERE id = ?", (run_id,)
        )
        row = await cursor.fetchone()
    assert json.loads(row[0]) == payload


async def test_complete_research_run_stores_checked_at(db_path):
    run_id = await create_research_run(db_path, "https://example.com/product")
    await complete_research_run(db_path, run_id, {}, CHECKED_AT)
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute("SELECT checked_at FROM research_runs WHERE id = ?", (run_id,))
        row = await cursor.fetchone()
    assert row[0] == CHECKED_AT.isoformat()


async def test_mark_research_run_failed_sets_status_failed(db_path):
    run_id = await create_research_run(db_path, "https://example.com/product")
    await mark_research_run_failed(db_path, run_id)
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute("SELECT status FROM research_runs WHERE id = ?", (run_id,))
        row = await cursor.fetchone()
    assert row[0] == "failed"


async def test_mark_research_run_failed_preserves_null_result_payload(db_path):
    run_id = await create_research_run(db_path, "https://example.com/product")
    await mark_research_run_failed(db_path, run_id)
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute(
            "SELECT result_payload FROM research_runs WHERE id = ?", (run_id,)
        )
        row = await cursor.fetchone()
    assert row[0] is None


async def test_load_research_run_returns_none_for_missing_id(db_path):
    result = await load_research_run(db_path, "nonexistent-id")
    assert result is None


async def test_load_research_run_returns_status(db_path):
    run_id = await create_research_run(db_path, "https://example.com/product")
    await complete_research_run(db_path, run_id, {"verdict": "good_deal"}, CHECKED_AT)
    run = await load_research_run(db_path, run_id)
    assert run is not None
    assert run["status"] == "completed"


async def test_load_research_run_returns_running_status_before_completion(db_path):
    run_id = await create_research_run(db_path, "https://example.com/product")
    run = await load_research_run(db_path, run_id)
    assert run is not None
    assert run["status"] == "running"


async def test_load_research_run_returns_payload_by_id(db_path):
    payload = {"verdict": "good_deal", "confidence": "high"}
    run_id = await create_research_run(db_path, "https://example.com/product")
    await complete_research_run(db_path, run_id, payload, CHECKED_AT)
    run = await load_research_run(db_path, run_id)
    assert run is not None
    assert run["result_payload"] == payload


async def test_load_research_run_returns_null_payload_for_running_run(db_path):
    run_id = await create_research_run(db_path, "https://example.com/product")
    run = await load_research_run(db_path, run_id)
    assert run is not None
    assert run["result_payload"] is None


async def test_load_research_run_returns_correct_id(db_path):
    run_id = await create_research_run(db_path, "https://example.com/product")
    await complete_research_run(db_path, run_id, {"verdict": "fair"}, CHECKED_AT)
    run = await load_research_run(db_path, run_id)
    assert run["id"] == run_id


async def test_load_research_run_returns_checked_at(db_path):
    run_id = await create_research_run(db_path, "https://example.com/product")
    await complete_research_run(db_path, run_id, {}, CHECKED_AT)
    run = await load_research_run(db_path, run_id)
    assert run["checked_at"] == CHECKED_AT.isoformat()


async def test_load_research_run_does_not_return_other_run(db_path):
    payload_a = {"verdict": "good_deal"}
    payload_b = {"verdict": "overpriced"}
    run_id_a = await create_research_run(db_path, "https://example.com/a")
    await complete_research_run(db_path, run_id_a, payload_a, CHECKED_AT)
    run_id_b = await create_research_run(db_path, "https://example.com/b")
    await complete_research_run(db_path, run_id_b, payload_b, CHECKED_AT)
    run = await load_research_run(db_path, run_id_a)
    assert run["result_payload"] == payload_a


async def test_init_db_migrates_old_schema_to_add_status(tmp_path):
    db_path = str(tmp_path / "old.db")
    async with aiosqlite.connect(db_path) as db:
        await db.execute("""
            CREATE TABLE research_runs (
                id TEXT PRIMARY KEY,
                normalized_url TEXT NOT NULL,
                result_payload TEXT NOT NULL,
                checked_at TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        await db.execute(
            """
            INSERT INTO research_runs
                (id, normalized_url, result_payload, checked_at, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                "existing-id",
                "https://example.com/product",
                '{"verdict":"good_deal"}',
                CHECKED_AT.isoformat(),
                CHECKED_AT.isoformat(),
            ),
        )
        await db.commit()

    await init_db(db_path)

    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute(
            "SELECT id, status, result_payload FROM research_runs WHERE id = 'existing-id'"
        )
        row = await cursor.fetchone()

    assert row is not None
    assert row[0] == "existing-id"
    assert row[1] == "completed"
    assert json.loads(row[2]) == {"verdict": "good_deal"}


async def test_save_cache_artifact_returns_uuid(db_path):
    artifact_id = await save_cache_artifact(
        db_path, "search_result", "widget-query", {}, FUTURE_EXPIRES_AT
    )
    assert uuid.UUID(artifact_id)


async def test_save_cache_artifact_stores_uuid_as_id(db_path):
    artifact_id = await save_cache_artifact(
        db_path, "search_result", "widget-query", {}, FUTURE_EXPIRES_AT
    )
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute("SELECT id FROM research_cache")
        row = await cursor.fetchone()
    assert row[0] == artifact_id


async def test_save_cache_artifact_persists_all_fields(db_path):
    await save_cache_artifact(
        db_path, "search_result", "widget-query", {"results": []}, FUTURE_EXPIRES_AT
    )
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute(
            "SELECT artifact_type, cache_key, payload, created_at, expires_at FROM research_cache"
        )
        row = await cursor.fetchone()
    assert row[0] == "search_result"
    assert row[1] == "widget-query"
    assert json.loads(row[2]) == {"results": []}
    assert row[3] is not None and row[3] != ""
    assert row[4] == FUTURE_EXPIRES_AT.isoformat()


async def test_save_cache_artifact_serializes_dict_payload(db_path):
    payload = {"url": "https://example.com", "content": "some text"}
    await save_cache_artifact(
        db_path, "fetched_page", "https://example.com", payload, FUTURE_EXPIRES_AT
    )
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute("SELECT payload FROM research_cache")
        row = await cursor.fetchone()
    assert json.loads(row[0]) == payload


async def test_delete_expired_cache_artifacts_removes_expired(db_path):
    past = datetime.now(timezone.utc) - timedelta(hours=1)
    await save_cache_artifact(db_path, "search_result", "old-query", {}, past)
    deleted = await delete_expired_cache_artifacts(db_path)
    assert deleted == 1
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute("SELECT COUNT(*) FROM research_cache")
        row = await cursor.fetchone()
    assert row[0] == 0


async def test_delete_expired_cache_artifacts_preserves_non_expired(db_path):
    future = datetime.now(timezone.utc) + timedelta(hours=1)
    await save_cache_artifact(db_path, "search_result", "fresh-query", {}, future)
    deleted = await delete_expired_cache_artifacts(db_path)
    assert deleted == 0
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute("SELECT COUNT(*) FROM research_cache")
        row = await cursor.fetchone()
    assert row[0] == 1


async def test_delete_expired_cache_artifacts_does_not_delete_research_runs(db_path):
    run_id = await create_research_run(db_path, "https://example.com/product")
    await complete_research_run(db_path, run_id, {}, CHECKED_AT)
    past = datetime.now(timezone.utc) - timedelta(hours=1)
    await save_cache_artifact(db_path, "search_result", "old-query", {}, past)
    await delete_expired_cache_artifacts(db_path)
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute("SELECT COUNT(*) FROM research_runs")
        row = await cursor.fetchone()
    assert row[0] == 1
