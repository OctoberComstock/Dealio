from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import HttpUrl, TypeAdapter, ValidationError

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

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
    return templates.TemplateResponse(request, "index.html")


@router.post("/", response_class=HTMLResponse)
async def submit_product_url(request: Request, product_url: str = Form(default="")):
    error = _validate_product_url(product_url)
    if error is not None:
        return templates.TemplateResponse(
            request,
            "index.html",
            {"error": error, "submitted_url": product_url},
            status_code=422,
        )
    return templates.TemplateResponse(
        request,
        "index.html",
        {"submitted_url": product_url, "received": True},
    )
