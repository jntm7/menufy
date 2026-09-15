"""FastAPI application entry point."""

from fastapi import FastAPI
from menufy.api.health import router as health_router

# FastAPI app
app = FastAPI(title="menufy")

# Routes
app.include_router(health_router)
