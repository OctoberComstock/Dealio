import logging
import logging.config
from contextlib import asynccontextmanager

from fastapi import FastAPI
from starlette.middleware.sessions import SessionMiddleware
from starlette.staticfiles import StaticFiles

from app.config import settings
from app.database import init_db
from app.routers import health, pages


def _configure_logging() -> None:
    logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s %(levelname)-8s %(name)s: %(message)s",
                "datefmt": "%H:%M:%S",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "standard",
            },
        },
        "loggers": {
            "app": {
                "handlers": ["console"],
                "level": settings.log_level.upper(),
                "propagate": False,
            },
        },
    })


@asynccontextmanager
async def lifespan(app: FastAPI):
    _configure_logging()
    await init_db(settings.database_path)
    yield


app = FastAPI(title="Dealio", description="Product deal research tool", lifespan=lifespan)

app.add_middleware(SessionMiddleware, secret_key=settings.session_secret_key, max_age=86400)
app.mount("/static", StaticFiles(directory="app/static"), name="static")
app.include_router(health.router)
app.include_router(pages.router)
