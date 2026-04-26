import json
import uuid
from datetime import datetime, timedelta, timezone

import aiosqlite
import pytest

from app.database import (
    delete_expired_cache_artifacts,
    init_db,
    save_cache_artifact,
    save_research_run,
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


async def test_save_research_run_returns_uuid(db_path):
    run_id = await save_research_run(db_path, "https://example.com/product", {}, CHECKED_AT)
    assert uuid.UUID(run_id)


async def test_save_research_run_stores_uuid_as_id(db_path):
    run_id = await save_research_run(db_path, "https://example.com/product", {}, CHECKED_AT)
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute("SELECT id FROM research_runs")
        row = await cursor.fetchone()
    assert row[0] == run_id


async def test_save_research_run_persists_normalized_url(db_path):
    await save_research_run(db_path, "https://example.com/product", {}, CHECKED_AT)
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute("SELECT normalized_url FROM research_runs")
        row = await cursor.fetchone()
    assert row[0] == "https://example.com/product"


async def test_save_research_run_serializes_dict_payload(db_path):
    payload = {"verdict": "good_deal", "confidence": "high"}
    await save_research_run(db_path, "https://example.com/product", payload, CHECKED_AT)
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute("SELECT result_payload FROM research_runs")
        row = await cursor.fetchone()
    assert json.loads(row[0]) == payload


async def test_save_research_run_persists_checked_at(db_path):
    await save_research_run(db_path, "https://example.com/product", {}, CHECKED_AT)
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute("SELECT checked_at FROM research_runs")
        row = await cursor.fetchone()
    assert row[0] == CHECKED_AT.isoformat()


async def test_save_research_run_persists_created_at(db_path):
    await save_research_run(db_path, "https://example.com/product", {}, CHECKED_AT)
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute("SELECT created_at FROM research_runs")
        row = await cursor.fetchone()
    assert row[0] is not None and row[0] != ""


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
    await save_research_run(db_path, "https://example.com/product", {}, CHECKED_AT)
    past = datetime.now(timezone.utc) - timedelta(hours=1)
    await save_cache_artifact(db_path, "search_result", "old-query", {}, past)
    await delete_expired_cache_artifacts(db_path)
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute("SELECT COUNT(*) FROM research_runs")
        row = await cursor.fetchone()
    assert row[0] == 1
