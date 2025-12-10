import os
import httpx
from datetime import datetime, timedelta
from typing import Any, Dict, List

from pydantic import Field
from fastmcp import Context
from mcp.types import TextContent
from mcp.shared.exceptions import McpError, ErrorData

# Импорты проекта
from mcp_instance import mcp
from b24 import b
from utils import ToolResult

# Телеметрия
from opentelemetry import trace
from prometheus_client import Counter

# Инициализация трейсинга и метрик
tracer = trace.get_tracer(__name__)
API_CALLS = Counter(
    "api_calls_total", 
    "Total API calls", 
    ["service", "endpoint", "status"]
)

@mcp.tool()
async def list_managers(ctx: Context = None) -> ToolResult:
    """Возвращает список всех активных менеджеров (пользователей) из CRM Битрикс24."""
    with tracer.start_as_current_span("list_managers") as span:
        try:
            if ctx: await ctx.info("Получаю список менеджеров из Битрикс24...")

            API_CALLS.labels(service="bitrix24", endpoint="user.get", status="started").inc()
            
            users = await b.get_all("user.get", {"ACTIVE": "true"})
            
            managers = [
                {"id": user["ID"], "name": f"{user.get('NAME', '')} {user.get('LAST_NAME', '')}".strip()} 
                for user in users
            ]

            API_CALLS.labels(service="bitrix24", endpoint="user.get", status="success").inc()
            span.set_attribute("managers_count", len(managers))
            
            if ctx: await ctx.info(f"Найдено {len(managers)} активных менеджеров.")
            
            return ToolResult(
                content=[TextContent(type="text", text=f"Найдено {len(managers)} менеджеров.")],
                structured_content=managers
            )
        except Exception as e:
            API_CALLS.labels(service="bitrix24", endpoint="user.get", status="error").inc()
            span.record_exception(e)
            
            if ctx: await ctx.error(f"Ошибка при получении списка менеджеров: {e}")
            raise McpError(ErrorData(code=-32603, message=f"Внутренняя ошибка при работе с API Битрикс24: {e}"))


@mcp.tool()
async def search_calls(
    user_id: str = Field(..., description="ID пользователя (менеджера) в Битрикс24"),
    start_date: str = Field(..., description="Дата начала поиска в формате YYYY-MM-DD."),
    end_date: str = Field(..., description="Дата конца поиска в формате YYYY-MM-DD."),
    ctx: Context = None
) -> ToolResult:
    """Ищет звонки через CRM (сущность Activity) по ID менеджера и диапазону дат."""
    with tracer.start_as_current_span("search_calls") as span:
        span.set_attribute("user_id", user_id)
        span.set_attribute("period", f"{start_date}-{end_date}")
        
        try:
            if ctx: await ctx.info(f"Ищу звонки в CRM для пользователя {user_id} с {start_date} по {end_date}...")
            
            API_CALLS.labels(service="bitrix24", endpoint="crm.activity.list", status="started").inc()
            
            params = {
                "filter": {
                    "RESPONSIBLE_ID": user_id,
                    "TYPE_ID": 2,
                    ">=START_TIME": f"{start_date}T00:00:00",
                    "<=START_TIME": f"{end_date}T23:59:59",
                },
                "select": ["*", "UFIELDS"],
            }

            raw_result = await b.get_all("crm.activity.list", params)
            
            activities = []
            if isinstance(raw_result, list):
                activities = raw_result
            elif isinstance(raw_result, dict):
                if "result" in raw_result:
                    activities = raw_result["result"]
                elif "ID" in raw_result:
                    activities = [raw_result]
                elif all(k.isdigit() for k in list(raw_result.keys())[:3]):
                     activities = list(raw_result.values())

            if not activities:
                API_CALLS.labels(service="bitrix24", endpoint="crm.activity.list", status="success").inc()
                return ToolResult(
                    content=[TextContent(type="text", text="Звонки (дела) за указанный период не найдены.")],
                    structured_content=[]
                )

            result_data = []
            for act in activities:
                record_id = None

                storage_ids = act.get("STORAGE_ELEMENT_IDS")
                if storage_ids:
                    if isinstance(storage_ids, list) and len(storage_ids) > 0:
                        record_id = storage_ids[0]
                    elif isinstance(storage_ids, dict):
                        record_id = list(storage_ids.values())[0]
                    elif isinstance(storage_ids, (str, int)):
                        record_id = storage_ids

                if not record_id:
                    files = act.get("FILES")
                    if files:
                        if isinstance(files, list) and len(files) > 0:
                            record_id = files[0].get("id") if isinstance(files[0], dict) else files[0]

                if not record_id:
                    webdav = act.get("WEBDAV_ELEMENTS")
                    if webdav:
                         if isinstance(webdav, list) and len(webdav) > 0:
                            record_id = webdav[0]

                if record_id:
                    result_data.append({
                        "call_id": act["ID"],
                        "user_id": act["RESPONSIBLE_ID"],
                        "start_time": act["START_TIME"],
                        "client_phone": act.get("PHONE_NUMBER", "Неизвестно"), 
                        "record_file_id": record_id
                    })

            API_CALLS.labels(service="bitrix24", endpoint="crm.activity.list", status="success").inc()
            span.set_attribute("found_calls_with_records", len(result_data))

            if ctx: await ctx.info(f"Найдено {len(result_data)} звонков с записью.")
            
            return ToolResult(
                content=[TextContent(type="text", text=f"Найдено {len(result_data)} звонков с записью.")],
                structured_content=result_data
            )
        except Exception as e:
            API_CALLS.labels(service="bitrix24", endpoint="crm.activity.list", status="error").inc()
            span.record_exception(e)
            if ctx: await ctx.error(f"Ошибка при поиске звонков в CRM: {e}")
            raise McpError(ErrorData(code=-32603, message=f"Внутренняя ошибка: {e}"))

@mcp.tool()
async def get_call_record_link(
    call_id: str = Field(..., description="ID звонка (дела) из search_calls"), 
    ctx: Context = None
) -> ToolResult:
    """
    Возвращает прямую ссылку на скачивание файла записи звонка.
    ВНИМАНИЕ: Ссылка требует, чтобы пользователь был авторизован в Битрикс24 в браузере.
    """
    with tracer.start_as_current_span("get_call_record_link") as span:
        try:
            if ctx: await ctx.info(f"Формирую ссылку на запись для звонка {call_id}...")
            
            API_CALLS.labels(service="bitrix24", endpoint="crm.activity.get", status="started").inc()
            
            raw_act = await b.call("crm.activity.get", {"id": call_id})
            activity = list(raw_act.values())[0] if isinstance(raw_act, dict) and 'order' in list(raw_act.keys())[0] else raw_act
            
            files = activity.get("FILES")
            if not files or not isinstance(files, list) or len(files) == 0:
                raise ValueError("В деле не найдено прикрепленных файлов.")
            
            relative_url = files[0].get('url')
            if not relative_url:
                raise ValueError("Ссылка на файл не найдена.")

            host = os.getenv("BITRIX24_WEBHOOK_URL").split("/rest")[0]
            full_url = host + relative_url if relative_url.startswith("/") else relative_url

            API_CALLS.labels(service="bitrix24", endpoint="crm.activity.get", status="success").inc()
            
            return ToolResult(
                content=[TextContent(type="text", text=f"Ссылка на запись звонка успешно создана.")],
                structured_content={
                    "status": "link_generated",
                    "download_url": full_url
                }
            )

        except Exception as e:
            API_CALLS.labels(service="bitrix24", endpoint="crm.activity.get", status="error").inc()
            raise McpError(ErrorData(code=-32603, message=f"Ошибка при получении ссылки: {e}"))