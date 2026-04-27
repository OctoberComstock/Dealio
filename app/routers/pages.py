import logging
from datetime import datetime

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import HttpUrl, TypeAdapter, ValidationError

from app.agent import run_research_agent
from app.config import settings
from app.database import load_research_run, save_research_run
from app.schemas import ResearchResult
from app.tools.extract_product import extract_product
from app.tools.normalize_url import normalize_url
from app.tools.product_identity import infer_product_identity

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")
logger = logging.getLogger(__name__)

_url_validator = TypeAdapter(HttpUrl)


def _validate_product_url(raw_url: str) -> str | None:
    stripped = raw_url.strip()
    if not stripped:
        return "Please enter a product URL."
    try:
        _url_validator.validate_python(stripped)
    except ValidationError:
        return "Please enter a valid product URL starting with http:// or https://"
    return None


@router.get("/", response_class=HTMLResponse)
async def homepage(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@router.post("/", response_class=HTMLResponse)
async def submit_product_url(request: Request, product_url: str = Form(default="")):
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
        extraction = await extract_product(normalized)
        identity = infer_product_identity(extraction, normalized)
        result = await run_research_agent(normalized, extraction, identity)
        run_id = await save_research_run(
            settings.database_path,
            normalized,
            result.model_dump(mode="json"),
            result.last_checked,
        )
    except Exception:
        logger.exception("Research flow failed for URL: %s", product_url)
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "error": "Something went wrong while researching that product. Please try again.",
                "submitted_url": product_url,
            },
            status_code=500,
        )

    return RedirectResponse(url=f"/result/{run_id}", status_code=303)


@router.get("/result/{run_id}", response_class=HTMLResponse)
async def result_page(request: Request, run_id: str):
    run = await load_research_run(settings.database_path, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Result not found")
    result = ResearchResult.model_validate(run["result_payload"])
    checked_at = datetime.fromisoformat(run["checked_at"])
    return templates.TemplateResponse(
        request=request,
        name="result.html",
        context={"result": result, "checked_at": checked_at},
    )
