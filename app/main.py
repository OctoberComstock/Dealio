from fastapi import FastAPI

from app.routers import health, pages

app = FastAPI(title="Dealio", description="Product deal research tool")

app.include_router(health.router)
app.include_router(pages.router)
