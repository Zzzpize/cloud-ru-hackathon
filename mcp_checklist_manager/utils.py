from typing import Any, List, Optional
from pydantic import BaseModel
from mcp.types import TextContent

class ToolResult(BaseModel):

    content: List[TextContent]
    structured_content: Optional[Any] = None
    meta: Optional[dict] = {}
