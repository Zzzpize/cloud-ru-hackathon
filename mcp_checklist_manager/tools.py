import json
from typing import List, Dict

from mcp_instance import mcp

from mcp.server.fastmcp import Context
from mcp.types import TextContent, ErrorData
from mcp.shared.exceptions import McpError
from utils import ToolResult
from pydantic import Field

from models import Checklist, ChecklistSummary, Criterion
from database import db
from metrics import checklist_operations_total



@mcp.tool()
async def get_checklist(
    ctx: Context,
    tenant_id: str = Field(..., description="Идентификатор клиента"),
    checklist_id: str = Field(..., description="Идентификатор чек-листа")
) -> ToolResult:
    try:
        db_data = await db._read_db()

        if tenant_id not in db_data:
            raise McpError(ErrorData(code=-32602, message=f"Клиент {tenant_id} не найден"))

        if checklist_id not in db_data[tenant_id]:
            raise McpError(ErrorData(code=-32602, message=f"Чеклист {checklist_id} не найден для клиента {tenant_id}"))
        
        checklist = db_data[tenant_id][checklist_id]
        checklist_operations_total.labels(operation="get_checklist", status="success").inc()

        return ToolResult(
            content=[TextContent(
                type="text",
                text=checklist.model_dump_json(indent=2)
            )]
        )
    except McpError:
        raise
    except Exception as e:
        raise McpError(ErrorData(code=-32603, message=str(e)))
    

@mcp.tool()
async def list_checklists(
    ctx: Context,
    tenant_id: str = Field(..., description="Идентификатор клиента")
) -> ToolResult:
    try:
        db_data = await db._read_db()

        if tenant_id not in db_data:
            return ToolResult(
                content=[TextContent(
                    type="text",
                    text="[]"
                )]
            )
        
        checklists_summary = [
            ChecklistSummary(id=checklist_id, name=checklist.name)
            for checklist_id, checklist in db_data[tenant_id].items()
        ]

        checklist_operations_total.labels(operation="list_checklists", status="success").inc()
        return ToolResult(
            content=[TextContent(
                type="text",
                text=json.dumps([item.model_dump() for item in checklists_summary], indent=2, ensure_ascii=False)
            )]
        )
    except McpError:
        raise
    except Exception as e:
        raise McpError(ErrorData(code=-32603, message=f"Ошибка при получении списка чеклистов: {str(e)}"))
    

@mcp.tool()
async def create_or_update_checklist(
    ctx: Context, 
    tenant_id: str = Field(..., description="Идентификатор клиента"),
    checklist_id: str = Field(..., description="Идентификатор чек-листа"),
    name: str = Field(..., description="Название чек-листа"),
    criteria: List[Dict] = Field(..., description="Список критериев чек-листа")
) -> ToolResult:
    try:
        validated_criteria = [Criterion(**criterion) for criterion in criteria]

        checklist = Checklist(
            checklist_id=checklist_id,
            name=name,
            criteria=validated_criteria
        )

        db_data = await db._read_db()

        if tenant_id not in db_data:
            db_data[tenant_id] = {}

        db_data[tenant_id][checklist_id] = checklist

        await db._write_db(db_data)
        checklist_operations_total.labels(operation="create_or_update_checklist", status="success").inc()
        return ToolResult(
            content=[TextContent(
                type="text",
                text=checklist.model_dump_json(indent=2)
            )]
        )
    except ValueError as ve:
        raise McpError(ErrorData(code=-32602, message=f"Ошибка валидации критериев: {str(ve)}"))
    except McpError:
        raise
    except Exception as e:
        raise McpError(ErrorData(code=-32603, message=f"Ошибка при создании/обновлении чек-листа: {str(e)}"))
    

@mcp.tool()
async def delete_checklist(
    ctx: Context,
    tenant_id: str = Field(..., description="Идентификатор клиента"),
    checklist_id: str = Field(..., description="Идентификатор чек-листа")
) -> ToolResult:
    try:
        db_data = await db._read_db()
        if tenant_id not in db_data:
            raise McpError(ErrorData(code=-32602, message=f"Клиент {tenant_id} не найден"))
        
        if checklist_id not in db_data[tenant_id]:
            raise McpError(ErrorData(code=-32602, message=f"Чеклист {checklist_id} не найден для клиента {tenant_id}"))
        
        del db_data[tenant_id][checklist_id]

        if not db_data[tenant_id]:
            del db_data[tenant_id]

        await db._write_db(db_data)

        responce = {
            "status" : "deleted",
            "checklist_id": checklist_id
        }
        checklist_operations_total.labels(operation="delete_checklist", status="success").inc()
        return ToolResult(
            content=[TextContent(
                type="text",
                text=json.dumps(responce, indent=2, ensure_ascii=False)
            )]
        )
    except McpError:
        raise
    except Exception as e:
        raise McpError(ErrorData(code=-32603, message=f"Ошибка при удалении чек-листа: {str(e)}"))
    