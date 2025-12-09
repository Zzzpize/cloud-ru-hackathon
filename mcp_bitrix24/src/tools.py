import os
import base64
import httpx
from datetime import datetime, timedelta
from pydantic import Field
from mcp_instance import mcp
from fastmcp import Context
from mcp.types import TextContent
from utils import ToolResult
from mcp.shared.exceptions import McpError, ErrorData

from b24 import b

@mcp.tool()
async def list_managers(ctx: Context = None) -> ToolResult:
    """Возвращает список всех активных менеджеров (пользователей) из CRM Битрикс24."""
    try:
        if ctx:
            await ctx.info("Получаю список менеджеров из Битрикс24...")
        
        users = await b.get_all("user.get", {"ACTIVE": "true"})
        managers = [{"id": user["ID"], "name": f"{user.get('NAME', '')} {user.get('LAST_NAME', '')}".strip()} for user in users]
        
        if ctx:
            await ctx.info(f"Найдено {len(managers)} активных менеджеров.")
            
        return ToolResult(
            content=[TextContent(type="text", text=f"Найдено {len(managers)} менеджеров.")],
            structured_content=managers
        )
    except Exception as e:
        if ctx:
            await ctx.error(f"Ошибка при получении списка менеджеров: {e}")
        raise McpError(ErrorData(code=-32603, message=f"Внутренняя ошибка при работе с API Битрикс24: {e}"))


@mcp.tool()
async def search_calls(
    user_id: str = Field(..., description="ID пользователя (менеджера) в Битрикс24"),
    start_date: str = Field(..., description="Дата начала поиска в формате YYYY-MM-DD."),
    end_date: str = Field(..., description="Дата конца поиска в формате YYYY-MM-DD."),
    ctx: Context = None
) -> ToolResult:
    """Ищет звонки через CRM (сущность Activity) по ID менеджера и диапазону дат."""
    try:
        if ctx:
            await ctx.info(f"Ищу звонки в CRM для пользователя {user_id} с {start_date} по {end_date}...")
        
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

        if ctx:
            await ctx.info(f"Найдено {len(result_data)} звонков с записью.")
            
        return ToolResult(
            content=[TextContent(type="text", text=f"Найдено {len(result_data)} звонков с записью.")],
            structured_content=result_data
        )
    except Exception as e:
        if ctx:
            await ctx.error(f"Ошибка при поиске звонков в CRM: {e}")
        raise McpError(ErrorData(code=-32603, message=f"Внутренняя ошибка: {e}"))

@mcp.tool()
async def get_call_record(
    call_id: str = Field(..., description="ID звонка (дела) из search_calls"),
    record_file_id: str = Field(..., description="ID файла записи из search_calls"),
    ctx: Context = None
) -> ToolResult:
    """
    Пытается скачать файл записи звонка. Если стандартный метод API
    недоступен, возвращает прямую ссылку для ручного скачивания.
    """
    try:
        if ctx: await ctx.info(f"Начинаю скачивание файла для дела {call_id} (файл ID: {record_file_id})...")
        if ctx: await ctx.info("Попытка #1: Использование стандартного метода API...")
        
        params = { "ownerTypeId": 2, "ownerId": call_id, "fileId": record_file_id }
        link_result = await b.call("crm.activity.file.getDownloadUrl", params)

        if not link_result or "result" not in link_result or "downloadUrl" not in link_result["result"]:
            error_details = ""
            if isinstance(link_result, dict):
                error_value = next(iter(link_result.values()), {})
                if isinstance(error_value, dict) and "error" in error_value:
                    error_details = error_value["error"]
            
            if "ERROR_METHOD_NOT_FOUND" in error_details:
                raise ValueError("METHOD_NOT_FOUND")
            else:
                 raise McpError(ErrorData(code=-32602, message=f"Не удалось получить ссылку на файл: {error_details}"))

        download_url = link_result["result"]["downloadUrl"]
        if ctx: await ctx.info(f"Стандартный метод API сработал! Скачиваю по ссылке...")

        async with httpx.AsyncClient() as client:
            response = await client.get(download_url, follow_redirects=True, timeout=60.0)
            response.raise_for_status()
            audio_bytes = response.content

        if not audio_bytes: raise McpError(ErrorData(code=-32603, message="Скачанный файл пуст."))

        if ctx: await ctx.info(f"✅ Файл успешно скачан, размер: {len(audio_bytes)} байт.")

        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        return ToolResult(
            content=[TextContent(type="text", text="Аудиозапись успешно скачана.")],
            structured_content={
                "status": "downloaded",
                "content_base64": audio_base64
            }
        )

    except ValueError as e:
        if str(e) == "METHOD_NOT_FOUND":
            if ctx: await ctx.warning("Стандартный метод API не найден. Откатываюсь к генерации прямой ссылки...")

            try:
                raw_act = await b.call("crm.activity.get", {"id": call_id})
                activity_data = next(iter(raw_act.values()))
                files = activity_data.get("FILES")
                relative_url = files[0].get('url')
                host = os.getenv("BITRIX24_WEBHOOK_URL").split("/rest")[0]
                full_url = host + relative_url

                if ctx: await ctx.info(f"⚠️ Прямая ссылка сгенерирована. Дальнейшая обработка требует 'демо-режима'.")

                return ToolResult(
                    content=[TextContent(type="text", text="Не удалось скачать файл автоматически, но была сгенерирована прямая ссылка.")],
                    structured_content={
                        "status": "link_only",
                        "download_url": full_url,
                        "message": "Для скачивания файла нужна авторизация в браузере. Используйте демо-режим для продолжения."
                    }
                )
            except Exception as fallback_e:
                raise McpError(ErrorData(code=-32603, message=f"Ошибка даже в запасном сценарии: {fallback_e}"))
        else:
             raise McpError(ErrorData(code=-32603, message=f"Внутренняя ошибка: {e}"))

    except Exception as e:
        if ctx: await ctx.error(f"Непредвиденная ошибка: {e}")
        raise McpError(ErrorData(code=-32603, message=f"Внутренняя ошибка: {e}"))