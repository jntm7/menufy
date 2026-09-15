"""Liveness endpoint."""

from typing import Annotated, Literal
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from menufy.config import LLMProvider, Settings, get_settings

router = APIRouter()

# Health response model
class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    llm_provider: LLMProvider
    llm_model: str

# Health route
@router.get("/health")
def health(settings: Annotated[Settings, Depends(get_settings)]) -> HealthResponse:
    return HealthResponse(llm_provider=settings.llm_provider, llm_model=settings.llm_model)
