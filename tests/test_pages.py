from datetime import datetime, timedelta, timezone
from unittest.mock import ANY, AsyncMock, patch

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
    "status": "completed",
    "result_payload": FAKE_RESULT.model_dump(mode="json"),
    "checked_at": FAKE_CHECKED_AT.isoformat(),
    "failure_reason": None,
}

FAKE_RUN_DATA_RUNNING = {
    "id": FAKE_RUN_ID,
    "status": "running",
    "result_payload": None,
    "checked_at": None,
    "failure_reason": None,
}

FAKE_RUN_DATA_FAILED = {
    "id": FAKE_RUN_ID,
    "status": "failed",
    "result_payload": None,
    "checked_at": None,
    "failure_reason": "Something went wrong while researching this product. Please try again.",
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


async def test_homepage_loading_message_is_initially_hidden(client):
    response = await client.get("/")
    assert 'id="loading-message"' in response.text
    assert "hidden" in response.text


async def test_error_page_loading_message_is_initially_hidden(client):
    response = await client.post("/", data={"product_url": "not-a-url"})
    assert 'id="loading-message"' in response.text
    assert "hidden" in response.text


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


async def test_submit_valid_url_redirects_immediately(client):
    url = "https://www.amazon.com/dp/B08N5WRWNW"
    with (
        patch(
            "app.routers.pages.create_research_run",
            new_callable=AsyncMock,
            return_value=FAKE_RUN_ID,
        ),
        patch(
            "app.routers.pages.run_research_agent",
            new_callable=AsyncMock,
            return_value=FAKE_RESULT,
        ),
        patch(
            "app.routers.pages.complete_research_run",
            new_callable=AsyncMock,
        ),
    ):
        response = await client.post("/", data={"product_url": url})
    assert response.status_code == 303
    assert response.headers["location"] == f"/result/{FAKE_RUN_ID}"


async def test_submit_creates_run_before_agent_executes(client):
    url = "https://www.amazon.com/dp/B08N5WRWNW"
    call_order = []

    async def mock_create(*args, **kwargs):
        call_order.append("create")
        return FAKE_RUN_ID

    async def mock_agent(*args, **kwargs):
        call_order.append("agent")
        return FAKE_RESULT

    with (
        patch("app.routers.pages.create_research_run", side_effect=mock_create),
        patch("app.routers.pages.run_research_agent", side_effect=mock_agent),
        patch("app.routers.pages.complete_research_run", new_callable=AsyncMock),
    ):
        await client.post("/", data={"product_url": url})

    assert call_order == ["create", "agent"]


async def test_submit_valid_url_strips_tracking_params_before_agent(client):
    url = "https://www.amazon.com/dp/B08N5WRWNW?utm_source=google"
    mock_agent = AsyncMock(return_value=FAKE_RESULT)
    with (
        patch(
            "app.routers.pages.create_research_run",
            new_callable=AsyncMock,
            return_value=FAKE_RUN_ID,
        ),
        patch("app.routers.pages.run_research_agent", mock_agent),
        patch("app.routers.pages.complete_research_run", new_callable=AsyncMock),
    ):
        await client.post("/", data={"product_url": url})
    called_url = mock_agent.call_args[0][0]
    assert "utm_source" not in called_url


async def test_submit_agent_failure_marks_run_as_failed(client):
    url = "https://www.amazon.com/dp/B08N5WRWNW"
    mock_fail = AsyncMock()
    with (
        patch(
            "app.routers.pages.create_research_run",
            new_callable=AsyncMock,
            return_value=FAKE_RUN_ID,
        ),
        patch(
            "app.routers.pages.run_research_agent",
            new_callable=AsyncMock,
            side_effect=Exception("API down"),
        ),
        patch("app.routers.pages.mark_research_run_failed", mock_fail),
    ):
        response = await client.post("/", data={"product_url": url})
    assert response.status_code == 303
    mock_fail.assert_awaited_once_with(ANY, FAKE_RUN_ID, ANY)


async def test_submit_agent_failure_still_redirects(client):
    url = "https://www.amazon.com/dp/B08N5WRWNW"
    with (
        patch(
            "app.routers.pages.create_research_run",
            new_callable=AsyncMock,
            return_value=FAKE_RUN_ID,
        ),
        patch(
            "app.routers.pages.run_research_agent",
            new_callable=AsyncMock,
            side_effect=Exception("API down"),
        ),
        patch("app.routers.pages.mark_research_run_failed", new_callable=AsyncMock),
    ):
        response = await client.post("/", data={"product_url": url})
    assert response.status_code == 303
    assert response.headers["location"] == f"/result/{FAKE_RUN_ID}"


async def test_submit_extraction_failure_returns_error_page(client):
    url = "https://www.amazon.com/dp/B08N5WRWNW"
    with patch(
        "app.routers.pages.create_research_run",
        new_callable=AsyncMock,
        side_effect=Exception("DB down"),
    ):
        response = await client.post("/", data={"product_url": url})
    assert response.status_code == 500
    assert "went wrong" in response.text.lower()


async def test_result_page_renders_for_completed_run(client):
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
        "status": "completed",
        "result_payload": {
            **FAKE_RESULT.model_dump(mode="json"),
            "last_checked": payload_last_checked.isoformat(),
        },
        "checked_at": checked_at.isoformat(),
        "failure_reason": None,
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


async def test_result_page_renders_loading_for_running_run(client):
    with patch(
        "app.routers.pages.load_research_run",
        new_callable=AsyncMock,
        return_value=FAKE_RUN_DATA_RUNNING,
    ):
        response = await client.get(f"/result/{FAKE_RUN_ID}")
    assert response.status_code == 200
    assert "progress" in response.text.lower() or "researching" in response.text.lower()


async def test_result_page_renders_error_for_failed_run(client):
    with patch(
        "app.routers.pages.load_research_run",
        new_callable=AsyncMock,
        return_value=FAKE_RUN_DATA_FAILED,
    ):
        response = await client.get(f"/result/{FAKE_RUN_ID}")
    assert response.status_code == 200
    assert FAKE_RUN_DATA_FAILED["failure_reason"] in response.text
