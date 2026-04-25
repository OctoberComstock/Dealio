from fastapi import FastAPI

from app.routers import pages

app = FastAPI(title="Dealio", description="Product deal research tool")

app.include_router(pages.router)
