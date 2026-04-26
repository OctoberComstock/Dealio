from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import settings
from app.database import init_db
from app.routers import health, pages


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db(settings.database_path)
    yield


app = FastAPI(title="Dealio", description="Product deal research tool", lifespan=lifespan)

app.include_router(health.router)
app.include_router(pages.router)
