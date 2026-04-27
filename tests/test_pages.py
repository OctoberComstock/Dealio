from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.schemas import Confidence, EvidenceItem, ResearchResult, Verdict

FAKE_RUN_ID = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
FAKE_CHECKED_AT = datetime.now(timezone.utc)

FAKE_RESULT = ResearchResult(
    product_name="Test Widget",
    merchant="example.com",
    listed_price="$29.99",
    verdict=Verdict.good_deal,
    confidence=Confidence.high,
    summary="This is a good deal based on market research.",
    evidence=[
        EvidenceItem(text="Listed below market average.", source_url="https://example.com/product"),
        EvidenceItem(text="Positive reviews across sources.", source_url="https://example.com/reviews"),
        EvidenceItem(text="Widely available at similar price.", source_url="https://example.com/other"),
    ],
    alternative=None,
    last_checked=FAKE_CHECKED_AT,
)

FAKE_RUN_DATA = {
    "id": FAKE_RUN_ID,
    "result_payload": FAKE_RESULT.model_dump(mode="json"),
    "checked_at": FAKE_CHECKED_AT.isoformat(),
}


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


async def test_submit_invalid_url_preserves_input(client):
    bad_url = "htp://bad"
    response = await client.post("/", data={"product_url": bad_url})
    assert response.status_code == 422
    assert bad_url in response.text


async def test_submit_valid_url_runs_research_and_redirects(client):
    url = "https://www.amazon.com/dp/B08N5WRWNW"
    with (
        patch(
            "app.routers.pages.run_research_agent",
            new_callable=AsyncMock,
            return_value=FAKE_RESULT,
        ),
        patch(
            "app.routers.pages.save_research_run",
            new_callable=AsyncMock,
            return_value=FAKE_RUN_ID,
        ),
    ):
        response = await client.post("/", data={"product_url": url})
    assert response.status_code == 303
    assert response.headers["location"] == f"/result/{FAKE_RUN_ID}"


async def test_submit_valid_url_strips_tracking_params_before_agent(client):
    url = "https://www.amazon.com/dp/B08N5WRWNW?utm_source=google"
    mock_agent = AsyncMock(return_value=FAKE_RESULT)
    with (
        patch("app.routers.pages.run_research_agent", mock_agent),
        patch(
            "app.routers.pages.save_research_run",
            new_callable=AsyncMock,
            return_value=FAKE_RUN_ID,
        ),
    ):
        await client.post("/", data={"product_url": url})
    called_url = mock_agent.call_args[0][0]
    assert "utm_source" not in called_url


async def test_submit_research_failure_rerenders_homepage_with_error(client):
    url = "https://www.amazon.com/dp/B08N5WRWNW"
    with patch(
        "app.routers.pages.run_research_agent",
        new_callable=AsyncMock,
        side_effect=Exception("API down"),
    ):
        response = await client.post("/", data={"product_url": url})
    assert response.status_code == 500
    assert "went wrong" in response.text.lower()
    assert 'name="product_url"' in response.text


async def test_result_page_renders_for_known_run(client):
    with patch(
        "app.routers.pages.load_research_run", new_callable=AsyncMock, return_value=FAKE_RUN_DATA
    ):
        response = await client.get(f"/result/{FAKE_RUN_ID}")
    assert response.status_code == 200


async def test_result_page_displays_product_name(client):
    with patch(
        "app.routers.pages.load_research_run", new_callable=AsyncMock, return_value=FAKE_RUN_DATA
    ):
        response = await client.get(f"/result/{FAKE_RUN_ID}")
    assert "Test Widget" in response.text


async def test_result_page_displays_verdict(client):
    with patch(
        "app.routers.pages.load_research_run", new_callable=AsyncMock, return_value=FAKE_RUN_DATA
    ):
        response = await client.get(f"/result/{FAKE_RUN_ID}")
    assert "Good Deal" in response.text


async def test_result_page_displays_confidence(client):
    with patch(
        "app.routers.pages.load_research_run", new_callable=AsyncMock, return_value=FAKE_RUN_DATA
    ):
        response = await client.get(f"/result/{FAKE_RUN_ID}")
    assert "High" in response.text


async def test_result_page_displays_summary(client):
    with patch(
        "app.routers.pages.load_research_run", new_callable=AsyncMock, return_value=FAKE_RUN_DATA
    ):
        response = await client.get(f"/result/{FAKE_RUN_ID}")
    assert "good deal based on market research" in response.text


async def test_result_page_displays_evidence_bullets(client):
    with patch(
        "app.routers.pages.load_research_run", new_callable=AsyncMock, return_value=FAKE_RUN_DATA
    ):
        response = await client.get(f"/result/{FAKE_RUN_ID}")
    assert "Listed below market average" in response.text
    assert "Positive reviews across sources" in response.text


async def test_result_page_displays_evidence_links(client):
    with patch(
        "app.routers.pages.load_research_run", new_callable=AsyncMock, return_value=FAKE_RUN_DATA
    ):
        response = await client.get(f"/result/{FAKE_RUN_ID}")
    assert "https://example.com/product" in response.text


async def test_result_page_uses_checked_at_not_payload_last_checked(client):
    checked_at = datetime.now(timezone.utc)
    payload_last_checked = checked_at - timedelta(days=400)
    run_data = {
        "id": FAKE_RUN_ID,
        "result_payload": {
            **FAKE_RESULT.model_dump(mode="json"),
            "last_checked": payload_last_checked.isoformat(),
        },
        "checked_at": checked_at.isoformat(),
    }
    with patch(
        "app.routers.pages.load_research_run", new_callable=AsyncMock, return_value=run_data
    ):
        response = await client.get(f"/result/{FAKE_RUN_ID}")
    assert str(checked_at.year) in response.text
    assert str(payload_last_checked.year) not in response.text


async def test_result_page_returns_404_for_missing_run(client):
    with patch(
        "app.routers.pages.load_research_run", new_callable=AsyncMock, return_value=None
    ):
        response = await client.get("/result/nonexistent-id")
    assert response.status_code == 404
