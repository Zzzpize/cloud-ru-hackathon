from typing import List, Optional, Literal
from pydantic import BaseModel, Field

class Criterion(BaseModel):
    criterion_id: str
    category: str
    text: str
    is_mandatory: bool
    check_type: Literal["keyword", "semantic"]
    keywords: List[str] = Field(default_factory=list)


class Checklist(BaseModel):
    checklist_id: str
    name: str
    criteria: List[Criterion]

class ChecklistSummary(BaseModel):
    id: str
    name: str

DbStructure = dict[str, dict[str, Checklist]]

