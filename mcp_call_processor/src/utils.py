import os
from typing import Any, List, Optional
from pydantic import BaseModel
from mcp.types import TextContent
from mcp.shared.exceptions import McpError, ErrorData

class ToolResult(BaseModel):
    content: List[TextContent]
    structured_content: Optional[Any] = None
    meta: Optional[dict] = {}

def _require_env_vars(vars_list: list[str]):
    """Проверяет наличие переменных окружения."""
    missing = [v for v in vars_list if not os.getenv(v)]
    if missing:
        raise McpError(ErrorData(
            code=-32602, 
            message=f"Missing required environment variables: {', '.join(missing)}"
        ))