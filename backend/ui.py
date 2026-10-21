"""Shared server-rendered UI configuration. No database/model imports."""
from pathlib import Path
from fastapi import Request
from fastapi.templating import Jinja2Templates

PROJECT_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = PROJECT_DIR / "frontend"
templates = Jinja2Templates(directory=str(FRONTEND_DIR))


def render_page(request: Request, filename: str, active_page: str):
    return templates.TemplateResponse(
        request=request,
        name=filename,
        context={"active_page": active_page},
    )
