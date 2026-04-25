from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI(title="Dealio", description="Product deal research tool")

templates = Jinja2Templates(directory="app/templates")
