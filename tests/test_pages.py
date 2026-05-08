from datetime import datetime, timedelta, timezone
from html import unescape
from unittest.mock import ANY, AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.config import settings
from app.main import app
from app.schemas import Confidence, EvidenceItem, ResearchResult, Verdict
from app.tools.extract_product import ProductExtractionResult, ProductPageExtraction

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

FAKE_EXTRACTION_RESULT = ProductExtractionResult(
    extraction=ProductPageExtraction(
        product_name="Test Widget",
        listed_price="$29.99",
        merchant="example.com",
    ),
    fetched_page=None,
)

FAKE_RUN_DATA = {
    "id": FAKE_RUN_ID,
    "status": "completed",
    "result_payload": FAKE_RESULT.model_dump(mode="json"),
    "checked_at": FAKE_CHECKED_AT.isoformat(),
    "failure_reason": None,
}


def make_completed_run_data_with_verdict(verdict: Verdict) -> dict:
    result = FAKE_RESULT.model_copy(update={"verdict": verdict})
    return make_completed_run_data(result)


def make_completed_run_data(result: ResearchResult) -> dict:
    return {
        "id": FAKE_RUN_ID,
        "status": "completed",
        "result_payload": result.model_dump(mode="json"),
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

TEST_DEMO_PASSWORD = "test-demo-password"


@pytest.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


@pytest.fixture
async def authenticated_client():
    with patch.object(settings, "dealio_demo_password", TEST_DEMO_PASSWORD):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            await ac.post("/unlock", data={"password": TEST_DEMO_PASSWORD})
            yield ac


# --- Landing page ---

async def test_landing_page_renders(client):
    response = await client.get("/")
    assert response.status_code == 200
    assert "dealio" in response.text


async def test_landing_page_has_github_link(client):
    response = await client.get("/")
    assert "github.com/OctoberComstock/Dealio" in response.text


async def test_landing_page_has_password_form(client):
    response = await client.get("/")
    assert 'name="password"' in response.text
    assert 'action="/unlock"' in response.text


async def test_landing_page_does_not_have_url_form(client):
    response = await client.get("/")
    assert 'name="product_url"' not in response.text


async def test_landing_page_shows_start_analysis_link_when_already_unlocked(authenticated_client):
    response = await authenticated_client.get("/")
    assert response.status_code == 200
    assert 'href="/analyze"' in response.text
    assert "demo access" in response.text.lower()


# --- Demo access gate ---

async def test_analyze_page_redirects_to_landing_without_demo_access(client):
    response = await client.get("/analyze")
    assert response.status_code == 303
    assert response.headers["location"] == "/"


async def test_submit_redirects_to_landing_without_demo_access(client):
    response = await client.post("/analyze", data={"product_url": "https://example.com/product"})
    assert response.status_code == 303
    assert response.headers["location"] == "/"


async def test_unlock_with_wrong_password_redirects_with_error(client):
    with patch.object(settings, "dealio_demo_password", TEST_DEMO_PASSWORD):
        response = await client.post("/unlock", data={"password": "wrong-password"})
    assert response.status_code == 303
    assert "error=invalid_password" in response.headers["location"]


async def test_unlock_with_correct_password_redirects_to_analyze(client):
    with patch.object(settings, "dealio_demo_password", TEST_DEMO_PASSWORD):
        response = await client.post("/unlock", data={"password": TEST_DEMO_PASSWORD})
    assert response.status_code == 303
    assert response.headers["location"] == "/analyze"


async def test_unlock_with_empty_configured_password_rejects_any_password(client):
    with patch.object(settings, "dealio_demo_password", ""):
        response = await client.post("/unlock", data={"password": ""})
    assert response.status_code == 303
    assert "error=invalid_password" in response.headers["location"]


async def test_landing_page_shows_error_for_invalid_password(client):
    response = await client.get("/?error=invalid_password")
    assert response.status_code == 200
    assert "isn't right" in response.text


async def test_analyze_page_renders_with_demo_access(authenticated_client):
    response = await authenticated_client.get("/analyze")
    assert response.status_code == 200
    assert 'name="product_url"' in response.text


async def test_analyze_page_loading_message_is_initially_hidden(authenticated_client):
    response = await authenticated_client.get("/analyze")
    assert 'id="loading-message"' in response.text
    assert "hidden" in response.text


# --- Submit ---

async def test_submit_empty_url_returns_error(authenticated_client):
    response = await authenticated_client.post("/analyze", data={"product_url": ""})
    assert response.status_code == 422
    assert "Please enter a product URL" in response.text


async def test_submit_invalid_url_returns_error(authenticated_client):
    response = await authenticated_client.post("/analyze", data={"product_url": "not-a-url"})
    assert response.status_code == 422
    assert "valid" in response.text.lower()


async def test_submit_invalid_url_preserves_input(authenticated_client):
    bad_url = "htp://bad"
    response = await authenticated_client.post("/analyze", data={"product_url": bad_url})
    assert response.status_code == 422
    assert bad_url in response.text


async def test_submit_error_page_loading_message_is_initially_hidden(authenticated_client):
    response = await authenticated_client.post("/analyze", data={"product_url": "not-a-url"})
    assert 'id="loading-message"' in response.text
    assert "hidden" in response.text


async def test_submit_valid_url_redirects_to_loading_page(authenticated_client):
    url = "https://www.amazon.com/dp/B08N5WRWNW"
    with (
        patch(
            "app.routers.pages.create_research_run",
            new_callable=AsyncMock,
            return_value=FAKE_RUN_ID,
        ),
        patch(
            "app.routers.pages._run_research_in_background",
            new_callable=AsyncMock,
        ),
    ):
        response = await authenticated_client.post("/analyze", data={"product_url": url})
    assert response.status_code == 303
    assert response.headers["location"] == f"/research/{FAKE_RUN_ID}/loading"


async def test_submit_creates_run_before_agent_executes(authenticated_client):
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
        patch(
            "app.routers.pages.extract_product",
            new_callable=AsyncMock,
            return_value=FAKE_EXTRACTION_RESULT,
        ),
        patch("app.routers.pages.run_research_agent", side_effect=mock_agent),
        patch("app.routers.pages.complete_research_run", new_callable=AsyncMock),
    ):
        await authenticated_client.post("/analyze", data={"product_url": url})

    assert call_order == ["create", "agent"]


async def test_submit_valid_url_strips_tracking_params_before_agent(authenticated_client):
    url = "https://www.amazon.com/dp/B08N5WRWNW?utm_source=google"
    mock_agent = AsyncMock(return_value=FAKE_RESULT)
    with (
        patch(
            "app.routers.pages.create_research_run",
            new_callable=AsyncMock,
            return_value=FAKE_RUN_ID,
        ),
        patch(
            "app.routers.pages.extract_product",
            new_callable=AsyncMock,
            return_value=FAKE_EXTRACTION_RESULT,
        ),
        patch("app.routers.pages.run_research_agent", mock_agent),
        patch("app.routers.pages.complete_research_run", new_callable=AsyncMock),
    ):
        await authenticated_client.post("/analyze", data={"product_url": url})
    called_url = mock_agent.call_args[0][0]
    assert "utm_source" not in called_url


async def test_submit_agent_failure_marks_run_as_failed(authenticated_client):
    url = "https://www.amazon.com/dp/B08N5WRWNW"
    mock_fail = AsyncMock()
    with (
        patch(
            "app.routers.pages.create_research_run",
            new_callable=AsyncMock,
            return_value=FAKE_RUN_ID,
        ),
        patch(
            "app.routers.pages.extract_product",
            new_callable=AsyncMock,
            return_value=FAKE_EXTRACTION_RESULT,
        ),
        patch(
            "app.routers.pages.run_research_agent",
            new_callable=AsyncMock,
            side_effect=Exception("API down"),
        ),
        patch("app.routers.pages.mark_research_run_failed", mock_fail),
    ):
        response = await authenticated_client.post("/analyze", data={"product_url": url})
    assert response.status_code == 303
    mock_fail.assert_awaited_once_with(ANY, FAKE_RUN_ID, ANY)


async def test_submit_create_run_failure_returns_error_page(authenticated_client):
    url = "https://www.amazon.com/dp/B08N5WRWNW"
    with patch(
        "app.routers.pages.create_research_run",
        new_callable=AsyncMock,
        side_effect=Exception("DB down"),
    ):
        response = await authenticated_client.post("/analyze", data={"product_url": url})
    assert response.status_code == 500
    assert "went wrong" in response.text.lower()


# --- Status endpoint ---

async def test_status_returns_running_for_running_run(client):
    with patch(
        "app.routers.pages.load_research_run",
        new_callable=AsyncMock,
        return_value=FAKE_RUN_DATA_RUNNING,
    ):
        response = await client.get(f"/research/{FAKE_RUN_ID}/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "running"
    assert data["result_url"] == f"/result/{FAKE_RUN_ID}"
    assert data["error_message"] is None


async def test_status_returns_completed_for_completed_run(client):
    with patch(
        "app.routers.pages.load_research_run",
        new_callable=AsyncMock,
        return_value=FAKE_RUN_DATA,
    ):
        response = await client.get(f"/research/{FAKE_RUN_ID}/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "completed"
    assert data["result_url"] == f"/result/{FAKE_RUN_ID}"
    assert data["error_message"] is None


async def test_status_returns_failed_with_error_message(client):
    with patch(
        "app.routers.pages.load_research_run",
        new_callable=AsyncMock,
        return_value=FAKE_RUN_DATA_FAILED,
    ):
        response = await client.get(f"/research/{FAKE_RUN_ID}/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "failed"
    assert data["error_message"] == FAKE_RUN_DATA_FAILED["failure_reason"]


async def test_status_returns_404_for_missing_run(client):
    with patch(
        "app.routers.pages.load_research_run", new_callable=AsyncMock, return_value=None
    ):
        response = await client.get("/research/nonexistent-id/status")
    assert response.status_code == 404


# --- Loading page ---

async def test_loading_page_renders_for_running_run(client):
    with patch(
        "app.routers.pages.load_research_run",
        new_callable=AsyncMock,
        return_value=FAKE_RUN_DATA_RUNNING,
    ):
        response = await client.get(f"/research/{FAKE_RUN_ID}/loading")
    assert response.status_code == 200
    assert "research" in response.text.lower()


async def test_loading_page_displays_progress_steps(client):
    with patch(
        "app.routers.pages.load_research_run",
        new_callable=AsyncMock,
        return_value=FAKE_RUN_DATA_RUNNING,
    ):
        response = await client.get(f"/research/{FAKE_RUN_ID}/loading")
    assert "Checking the product page" in response.text
    assert "Searching major retailers" in response.text
    assert "Comparing prices across stores" in response.text
    assert "Verifying the best alternative" in response.text
    assert "Preparing your verdict" in response.text


async def test_loading_page_displays_supporting_copy(client):
    with patch(
        "app.routers.pages.load_research_run",
        new_callable=AsyncMock,
        return_value=FAKE_RUN_DATA_RUNNING,
    ):
        response = await client.get(f"/research/{FAKE_RUN_ID}/loading")
    assert "marketplace" in response.text.lower()


async def test_loading_page_renders_error_for_failed_run(client):
    with patch(
        "app.routers.pages.load_research_run",
        new_callable=AsyncMock,
        return_value=FAKE_RUN_DATA_FAILED,
    ):
        response = await client.get(f"/research/{FAKE_RUN_ID}/loading")
    assert response.status_code == 200
    assert FAKE_RUN_DATA_FAILED["failure_reason"] in response.text


async def test_loading_page_redirects_to_result_for_completed_run(client):
    with patch(
        "app.routers.pages.load_research_run",
        new_callable=AsyncMock,
        return_value=FAKE_RUN_DATA,
    ):
        response = await client.get(f"/research/{FAKE_RUN_ID}/loading")
    assert response.status_code == 303
    assert response.headers["location"] == f"/result/{FAKE_RUN_ID}"


async def test_loading_page_returns_404_for_missing_run(client):
    with patch(
        "app.routers.pages.load_research_run", new_callable=AsyncMock, return_value=None
    ):
        response = await client.get("/research/nonexistent-id/loading")
    assert response.status_code == 404


# --- Result page ---

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


@pytest.mark.parametrize(
    ("verdict", "expected_label"),
    [
        (Verdict.good_deal, "Buy"),
        (Verdict.fair, "Wait"),
        (Verdict.overpriced, "Don't Buy"),
        (Verdict.insufficient_data, "Not Enough Data"),
    ],
)
async def test_result_page_displays_verdict_label(client, verdict, expected_label):
    run_data = make_completed_run_data_with_verdict(verdict)
    with patch(
        "app.routers.pages.load_research_run", new_callable=AsyncMock, return_value=run_data
    ):
        response = await client.get(f"/result/{FAKE_RUN_ID}")
    assert expected_label in unescape(response.text)


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


async def test_result_page_displays_purchase_decision_sections(client):
    with patch(
        "app.routers.pages.load_research_run", new_callable=AsyncMock, return_value=FAKE_RUN_DATA
    ):
        response = await client.get(f"/result/{FAKE_RUN_ID}")

    assert "example.com" in response.text
    assert "Listed price" in response.text
    assert "Dealio recommendation" in response.text
    assert "Price vs market" in response.text
    assert "Prices and availability change frequently" in response.text


async def test_result_page_displays_market_snapshot_from_evidence_prices(client):
    result = FAKE_RESULT.model_copy(
        update={
            "listed_price": "$19.99",
            "evidence": [
                EvidenceItem(
                    text="Amazon lists the same item at $19.99.",
                    source_url="https://example.com/amazon",
                ),
                EvidenceItem(
                    text="Target lists the same item at $24.99.",
                    source_url="https://example.com/target",
                ),
                EvidenceItem(
                    text="Brand store lists the same item at $34.99.",
                    source_url="https://example.com/brand",
                ),
            ],
        }
    )
    run_data = make_completed_run_data(result)

    with patch(
        "app.routers.pages.load_research_run", new_callable=AsyncMock, return_value=run_data
    ):
        response = await client.get(f"/result/{FAKE_RUN_ID}")

    assert "Submitted" in response.text
    assert "$19.99" in response.text
    assert "$24.99" in response.text
    assert "$34.99" in response.text


async def test_result_page_hides_market_modules_for_insufficient_data(client):
    result = FAKE_RESULT.model_copy(update={"verdict": Verdict.insufficient_data})
    run_data = make_completed_run_data(result)

    with patch(
        "app.routers.pages.load_research_run", new_callable=AsyncMock, return_value=run_data
    ):
        response = await client.get(f"/result/{FAKE_RUN_ID}")

    assert "Not Enough Data" in response.text
    assert "Price vs market" not in response.text


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


async def test_result_page_redirects_to_loading_for_running_run(client):
    with patch(
        "app.routers.pages.load_research_run",
        new_callable=AsyncMock,
        return_value=FAKE_RUN_DATA_RUNNING,
    ):
        response = await client.get(f"/result/{FAKE_RUN_ID}")
    assert response.status_code == 303
    assert response.headers["location"] == f"/research/{FAKE_RUN_ID}/loading"


async def test_result_page_redirects_to_loading_for_failed_run(client):
    with patch(
        "app.routers.pages.load_research_run",
        new_callable=AsyncMock,
        return_value=FAKE_RUN_DATA_FAILED,
    ):
        response = await client.get(f"/result/{FAKE_RUN_ID}")
    assert response.status_code == 303
    assert response.headers["location"] == f"/research/{FAKE_RUN_ID}/loading"
