import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


async def test_healthz_returns_200(client):
    response = await client.get("/healthz")
    assert response.status_code == 200


async def test_healthz_returns_ok_true(client):
    response = await client.get("/healthz")
    assert response.json() == {"ok": True}
