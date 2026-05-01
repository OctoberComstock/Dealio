import logging
import time
from datetime import datetime

from fastapi import APIRouter, BackgroundTasks, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import HttpUrl, TypeAdapter, ValidationError

from app.agent import run_research_agent
from app.config import settings
from app.database import (
    complete_research_run,
    create_research_run,
    load_research_run,
    mark_research_run_failed,
)
from app.schemas import ResearchResult
from app.tools.extract_product import ProductPageExtraction, extract_product
from app.tools.fetch_page import FetchedPage
from app.tools.normalize_url import normalize_url
from app.tools.product_identity import ProductIdentity, infer_product_identity

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")
logger = logging.getLogger(__name__)

_url_validator = TypeAdapter(HttpUrl)

_RESEARCH_FAILURE_MESSAGE = (
    "Something went wrong while researching this product. Please try again."
)


def _validate_product_url(raw_url: str) -> str | None:
    stripped = raw_url.strip()
    if not stripped:
        return "Please enter a product URL."
    try:
        _url_validator.validate_python(stripped)
    except ValidationError:
        return "Please enter a valid product URL starting with http:// or https://"
    return None


async def _run_research_in_background(
    run_id: str,
    normalized: str,
    extraction: ProductPageExtraction,
    identity: ProductIdentity,
    fetched_page: FetchedPage | None,
    t_start: float,
) -> None:
    try:
        result = await run_research_agent(
            normalized, extraction, identity, initial_fetched_page=fetched_page
        )
        await complete_research_run(
            settings.database_path,
            run_id,
            result.model_dump(mode="json"),
            result.last_checked,
        )
        logger.info(
            "Research completed: verdict=%s confidence=%s run_id=%s elapsed=%.1fs url=%s",
            result.verdict.value,
            result.confidence.value,
            run_id,
            time.perf_counter() - t_start,
            normalized,
        )
    except Exception:
        logger.exception("Background research failed: run_id=%s url=%s", run_id, normalized)
        await mark_research_run_failed(
            settings.database_path,
            run_id,
            _RESEARCH_FAILURE_MESSAGE,
        )


@router.get("/", response_class=HTMLResponse)
async def homepage(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@router.post("/", response_class=HTMLResponse)
async def submit_product_url(
    request: Request,
    background_tasks: BackgroundTasks,
    product_url: str = Form(default=""),
):
    error = _validate_product_url(product_url)
    if error is not None:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={"error": error, "submitted_url": product_url},
            status_code=422,
        )

    try:
        normalized = normalize_url(product_url.strip())
        t_start = time.perf_counter()
        logger.info("Research starting: url=%s", normalized)

        extraction_result = await extract_product(normalized)
        extraction = extraction_result.extraction
        fetched_page = extraction_result.fetched_page
        identity = infer_product_identity(extraction, normalized)
        logger.info(
            "Product extracted: name=%r price=%r merchant=%r identity=%r source=%s",
            extraction.product_name,
            extraction.listed_price,
            extraction.merchant,
            identity.value,
            identity.source,
        )

        run_id = await create_research_run(settings.database_path, normalized)
    except Exception:
        logger.exception("Submit failed for URL: %s", product_url)
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "error": "Something went wrong while researching that product. Please try again.",
                "submitted_url": product_url,
            },
            status_code=500,
        )

    background_tasks.add_task(
        _run_research_in_background,
        run_id,
        normalized,
        extraction,
        identity,
        fetched_page,
        t_start,
    )
    return RedirectResponse(url=f"/result/{run_id}", status_code=303)


@router.get("/result/{run_id}", response_class=HTMLResponse)
async def result_page(request: Request, run_id: str):
    run = await load_research_run(settings.database_path, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Result not found")

    if run["status"] in ("running", "failed"):
        return templates.TemplateResponse(
            request=request,
            name="loading.html",
            context={
                "run_id": run_id,
                "status": run["status"],
                "failure_reason": run["failure_reason"],
            },
        )

    result = ResearchResult.model_validate(run["result_payload"])
    checked_at = datetime.fromisoformat(run["checked_at"])
    return templates.TemplateResponse(
        request=request,
        name="result.html",
        context={"result": result, "checked_at": checked_at},
    )
