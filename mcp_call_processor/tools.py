"""
Инструмент: transcribe_audio

Транскрибирует аудиозапись по URL и возвращает структуру согласно API-контракту:
{
  "status": "completed",
  "full_text": "...",
  "duration_seconds": 185,
  "segments": [ { "speaker": "manager", "start_time": 0.5, "end_time": 4.2, "text": "..." }, ... ]
}

Требования проекта выполнены:
- async def
- строгая типизация
- pydantic.Field для параметров
- Context (ctx) используется для логов и прогресса
- raise McpError(ErrorData(...)) при пользовательских ошибках
- OpenTelemetry span (with tracer.start_as_current_span)
- Prometheus метрики: transcription_requests_total, transcription_duration_seconds
- возвращает ToolResult
"""

import os
import tempfile
import time
from typing import Any, Dict

import aiofiles
import httpx
from dotenv import load_dotenv, find_dotenv
from fastmcp import Context
from mcp.types import TextContent
from opentelemetry import trace
from pydantic import Field

from mcp_instance import mcp
from mcp.shared.exceptions import McpError, ErrorData

# утилиты проекта (предполагается, что в проекте есть utils с ToolResult и _require_env_vars)
from .utils import ToolResult, _require_env_vars, format_api_error

# Prometheus
from prometheus_client import Counter, Histogram

load_dotenv(find_dotenv())

tracer = trace.get_tracer(__name__)

# Prometheus metrics
TRANSCRIPTION_REQUESTS = Counter(
    "transcription_requests_total",
    "Total transcription requests",
    ["status"],
)
TRANSCRIPTION_DURATION = Histogram(
    "transcription_duration_seconds",
    "Histogram of total transcription duration (seconds)"
)

# Environment variable names expected
_ENV_VARS = [
    "STT_API_URL",     # URL сервиса STT, например https://stt.example.com/transcribe
    "STT_API_KEY",     # (опционально) ключ для доступа
    "STT_TIMEOUT"      # (опционально) таймаут в секундах для httpx (float)
]


@mcp.tool(
    name="transcribe_audio",
    description="""Транскрибировать аудиозапись по URL и вернуть JSON в заранее согласованном формате:
- скачивает audio_url
- отправляет файл в STT-сервис
- формирует выдачу с сегментами по спикерам
"""
)
async def transcribe_audio(
    audio_url: str = Field(..., description="URL до аудиофайла (http/https)"),
    ctx: Context = None,
) -> ToolResult:
    """
    Транскрибирует аудио по ссылке.

    Args:
        audio_url: URL на аудиофайл
        ctx: контекст для логирования и прогресса

    Returns:
        ToolResult: результат в формате API-контракта (structured_content)

    Raises:
        McpError: при ошибках (валидация, сеть, ответ STT)
    """
    # Валидация обязательных переменных окружения
    try:
        env = _require_env_vars(_ENV_VARS)
    except Exception as e:
        # _require_env_vars сам поднимает McpError, но на всякий случай ловим
        await (ctx.error(f"❌ Ошибка конфигурации: {e}") if ctx else None)
        raise

    stt_url = env.get("STT_API_URL")  # обязателен
    stt_key = env.get("STT_API_KEY", "")
    stt_timeout = float(os.getenv("STT_TIMEOUT", "120.0"))

    await ctx.info(f"🚀 Начинаем транскрибацию: {audio_url}")
    await ctx.report_progress(progress=0, total=100)

    start_time = time.time()
    status_label = "error"

    with tracer.start_as_current_span("call_processor.transcribe") as span:
        span.set_attribute("audio_url", audio_url)

        # шаг 1: скачать файл асинхронно во временный файл
        await ctx.info("⬇️ Скачиваем аудиофайл")
        await ctx.report_progress(progress=5, total=100)
        tmp_path = None
        try:
            timeout = httpx.Timeout(timeout=stt_timeout)
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream("GET", audio_url) as resp:
                    resp.raise_for_status()
                    # создаём временный файл
                    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".audio")
                    tmp_path = tmp_file.name
                    tmp_file.close()  # будем писать через aiofiles
                    # записываем поток в файл
                    async with aiofiles.open(tmp_path, "wb") as f:
                        async for chunk in resp.aiter_bytes():
                            await f.write(chunk)
            await ctx.report_progress(progress=20, total=100)
            await ctx.info("✅ Файл скачан")
            span.set_attribute("downloaded", True)
        except httpx.HTTPStatusError as e:
            await ctx.error(f"❌ Ошибка скачивания файла: {e.response.status_code}")
            span.set_attribute("error", "download_http_error")
            TRANSCRIPTION_REQUESTS.labels(status="error").inc()
            raise McpError(ErrorData(code=-32603, message=f"Ошибка скачивания аудиофайла: {e.response.status_code}"))
        except Exception as e:
            await ctx.error(f"❌ Ошибка при скачивании: {e}")
            span.set_attribute("error", str(e))
            TRANSCRIPTION_REQUESTS.labels(status="error").inc()
            raise McpError(ErrorData(code=-32603, message=f"Не удалось скачать аудиофайл: {e}"))

        # шаг 2: отправить файл в STT
        await ctx.info("📡 Отправляем файл в STT-сервис")
        await ctx.report_progress(progress=30, total=100)
        try:
            files = {"file": (os.path.basename(tmp_path), open(tmp_path, "rb"))}
            headers = {}
            if stt_key:
                headers["Authorization"] = f"Bearer {stt_key}"

            # Универсально: посылаем multipart/form-data на STT URL и ожидаем JSON
            # Пример ожидаемого от STT ответа:
            # {
            #   "status": "completed",
            #   "full_text": "...",
            #   "duration_seconds": 185,
            #   "segments": [ { "speaker": "manager", "start_time": 0.5, "end_time": 4.2, "text": "..." }, ... ]
            # }
            # В случае другого формата — нужно заменить парсер.
            async with httpx.AsyncClient(timeout=timeout) as client:
                # "files" не поддерживается в async context-manager way with open handle easily,
                # поэтому отправим в потоковом режиме: используем client.post с files (httpx умеет)
                with open(tmp_path, "rb") as fh:
                    files_payload = {"file": (os.path.basename(tmp_path), fh, "application/octet-stream")}
                    response = await client.post(stt_url, files=files_payload, headers=headers)
                response.raise_for_status()
                stt_result = response.json()

            await ctx.report_progress(progress=70, total=100)
            await ctx.info("✅ Получен ответ от STT")
            span.set_attribute("stt_response_ok", True)
        except httpx.HTTPStatusError as e:
            await ctx.error(f"❌ HTTP ошибка от STT: {getattr(e.response, 'status_code', 'unknown')}")
            span.set_attribute("error", "stt_http_status")
            TRANSCRIPTION_REQUESTS.labels(status="error").inc()
            raise McpError(ErrorData(code=-32603, message=f"STT сервис вернул ошибку: {format_api_error(getattr(e.response, 'text', ''), getattr(e.response, 'status_code', 0))}"))
        except Exception as e:
            await ctx.error(f"❌ Ошибка при обращении к STT: {e}")
            span.set_attribute("error", str(e))
            TRANSCRIPTION_REQUESTS.labels(status="error").inc()
            raise McpError(ErrorData(code=-32603, message=f"Ошибка при вызове STT: {e}"))
        finally:
            # убираем временный файл (если есть)
            try:
                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                # не фатализируем, просто логируем в ctx
                await ctx.debug("⚠️ Не удалось удалить временный файл")

        # шаг 3: проверка/нормализация ответа STT в нужный контракт
        await ctx.info("🔎 Форматируем результат")
        await ctx.report_progress(progress=85, total=100)
        try:
            # Вариант: если STT уже отдаёт контракт — используем напрямую.
            # Если формат другой — попытаться извлечь минимальные поля.
            status = stt_result.get("status", "completed")
            full_text = stt_result.get("full_text") or stt_result.get("transcript") or ""
            duration_seconds = float(stt_result.get("duration_seconds") or stt_result.get("duration") or 0.0)
            segments = stt_result.get("segments") or []

            # Нормализуем segments: гарантируем поля speaker, start_time, end_time, text
            normalized_segments = []
            for seg in segments:
                s_speaker = seg.get("speaker", seg.get("role", "unknown"))
                s_start = float(seg.get("start_time", seg.get("start", 0.0)))
                s_end = float(seg.get("end_time", seg.get("end", s_start + 0.0)))
                s_text = seg.get("text", seg.get("content", ""))
                normalized_segments.append({
                    "speaker": s_speaker,
                    "start_time": s_start,
                    "end_time": s_end,
                    "text": s_text
                })

            result_json: Dict[str, Any] = {
                "status": status,
                "full_text": full_text,
                "duration_seconds": duration_seconds,
                "segments": normalized_segments,
            }

            await ctx.report_progress(progress=100, total=100)
            await ctx.info("🎉 Транскрибация завершена")
            span.set_attribute("success", True)
            span.set_attribute("duration_seconds", duration_seconds)
            status_label = "success"
        except Exception as e:
            await ctx.error(f"❌ Ошибка обработки ответа STT: {e}")
            span.set_attribute("error", str(e))
            TRANSCRIPTION_REQUESTS.labels(status="error").inc()
            raise McpError(ErrorData(code=-32603, message=f"Ошибка обработки ответа STT: {e}"))
        finally:
            total_time = time.time() - start_time
            # Обновляем метрики
            TRANSCRIPTION_DURATION.observe(total_time)
            TRANSCRIPTION_REQUESTS.labels(status=status_label).inc()
            span.set_attribute("operation_elapsed_seconds", total_time)

        # Формируем ToolResult
        # Человеко-читаемая часть — краткий summary (первые 300 символов)
        human_readable = (result_json["full_text"][:300] + "...") if len(result_json["full_text"]) > 300 else result_json["full_text"]
        content = [TextContent(type="text", text=f"Transcription status: {result_json['status']}\n{human_readable}")]

        return ToolResult(
            content=content,
            structured_content=result_json,
            meta={"audio_url": audio_url, "duration_seconds": result_json["duration_seconds"]}
        )