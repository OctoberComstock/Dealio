import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


async def test_homepage_renders(client):
    response = await client.get("/")
    assert response.status_code == 200
    assert "Dealio" in response.text
    assert 'name="product_url"' in response.text


async def test_submit_empty_url_returns_error(client):
    response = await client.post("/", data={"product_url": ""})
    assert response.status_code == 422
    assert "Please enter a product URL" in response.text


async def test_submit_invalid_url_returns_error(client):
    response = await client.post("/", data={"product_url": "not-a-url"})
    assert response.status_code == 422
    assert "valid" in response.text.lower()


async def test_submit_valid_url_shows_confirmation(client):
    url = "https://www.amazon.com/dp/B08N5WRWNW"
    response = await client.post("/", data={"product_url": url})
    assert response.status_code == 200
    assert url in response.text
    assert "received" in response.text.lower()


async def test_submit_invalid_url_preserves_input(client):
    bad_url = "htp://bad"
    response = await client.post("/", data={"product_url": bad_url})
    assert response.status_code == 422
    assert bad_url in response.text
