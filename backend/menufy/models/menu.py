"""Structured menu data produced by the extraction stage."""

from pydantic import BaseModel, Field

# Dish model
class Dish(BaseModel):
    name_original: str
    name_translated: str | None = None
    description_original: str | None = None
    description_translated: str | None = None
    # Kept as text: menus mix currencies, ranges, and "market price".
    price: str | None = None
    section: str | None = None

# Menu result model
class MenuResult(BaseModel):
    source_language: str | None = None
    target_language: str = "en"
    dishes: list[Dish] = Field(default_factory=list)
