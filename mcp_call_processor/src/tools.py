
# mcp_call_processor/tools.py
"""
Final MCP tool: transcribe_audio using SaluteSpeech (Smartspeech) async REST flow.
This file is intended to be saved as mcp_call_processor/tools.py in your project.

Flow:
 - download audio by URL (async)
 - convert to WAV PCM_S16LE 16kHz mono via ffmpeg (async)
 - obtain OAuth token (cached)
 - upload file to /data:upload -> get request_file_id
 - create async recognition task /speech:async_recognize with speaker_separation_options
 - poll /task:get until DONE -> (use uploaded file_id as response_file_id)
 - download result via /data:download?response_file_id=<uploaded_file_id>
 - handle ZIP or octet-stream payloads (extract JSON)
 - normalize SaluteSpeech result -> API contract
 - return ToolResult(structured_content=contract)
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import tempfile
import time
import uuid
import zipfile
from typing import Any, Dict, List, Optional

import aiofiles
import httpx
from dotenv import load_dotenv, find_dotenv
from fastmcp import Context
from mcp.types import TextContent
from opentelemetry import trace
from pydantic import Field

from mcp_instance import mcp
from mcp.shared.exceptions import McpError, ErrorData
from prometheus_client import Counter, Histogram

from utils import ToolResult, _require_env_vars

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
    "Histogram of transcription duration",
)

# Required env vars
_ENV_VARS = [
    "SBER_OAUTH_CLIENT_ID",
    "SBER_OAUTH_CLIENT_SECRET",
]

# Defaults (can be overridden via .env)
DEFAULT_OAUTH_URL = os.getenv("SBER_OAUTH_URL", "https://ngw.devices.sberbank.ru:9443/api/v2/oauth")
DEFAULT_SALUTESPEECH_BASE = os.getenv("SBER_SALUTESPEECH_BASE_URL", "https://smartspeech.sber.ru/rest/v1")
DEFAULT_UPLOAD_PATH = os.getenv("SBER_UPLOAD_PATH", "/data:upload")
DEFAULT_CREATE_TASK_PATH = os.getenv("SBER_CREATE_TASK_PATH", "/speech:async_recognize")
DEFAULT_TASK_STATUS_PATH = os.getenv("SBER_TASK_STATUS_PATH", "/task:get")
DEFAULT_DATA_DOWNLOAD_PATH = os.getenv("SBER_TASK_RESULT_PATH", "/data:download")

# Token cache + lock
_token_lock: asyncio.Lock = asyncio.Lock()
_cached_token: Optional[str] = None
_cached_expiry: Optional[float] = None

# Helpers


def _parse_bool_env(name: str, default: str = "true") -> bool:
    v = os.getenv(name, default)
    if v is None:
        return default.lower() in ("1", "true", "yes")
    return str(v).strip().lower() in ("1", "true", "yes")


async def _ctx_info(ctx: Optional[Context], msg: str) -> None:
    if ctx:
        await ctx.info(msg)


async def _ctx_debug(ctx: Optional[Context], msg: str) -> None:
    if ctx:
        await ctx.debug(msg)


async def _ctx_error(ctx: Optional[Context], msg: str) -> None:
    if ctx:
        await ctx.error(msg)


# ---- audio download & convert ----


async def _download_to_temp(audio_url: str, ctx: Optional[Context], timeout: float = 120.0) -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False)
    tmp_path = tmp.name
    tmp.close()
    await _ctx_debug(ctx, f"Downloading audio to temporary file: {tmp_path}")
    verify_ssl = _parse_bool_env("SBER_VERIFY_SSL", "true")
    try:
        async with httpx.AsyncClient(timeout=timeout, verify=verify_ssl) as client:
            async with client.stream("GET", audio_url) as resp:
                try:
                    resp.raise_for_status()
                except httpx.HTTPStatusError as e:
                    await _ctx_error(ctx, f"HTTP error while downloading audio: {e.response.status_code}")
                    raise McpError(ErrorData(code=-32001, message=f"Failed to download audio: status {e.response.status_code}"))
                async with aiofiles.open(tmp_path, "wb") as f:
                    async for chunk in resp.aiter_bytes():
                        await f.write(chunk)
    except McpError:
        raise
    except Exception as e:
        await _ctx_error(ctx, f"Error downloading audio: {e}")
        raise McpError(ErrorData(code=-32002, message=f"Error downloading audio: {e}"))
    return tmp_path


async def _ensure_wav(input_path: str, ctx: Optional[Context]) -> str:
    """
    Convert to WAV PCM_S16LE 16kHz mono via ffmpeg if needed.
    Returns path to wav file (may be same as input_path).
    """
    _, ext = os.path.splitext(input_path)
    ext = ext.lower()
    if ext == ".wav":
        await _ctx_debug(ctx, "Input already WAV — skipping conversion")
        return input_path

    await _ctx_info(ctx, "Converting audio to WAV (16kHz mono, PCM_S16LE) via ffmpeg")
    fd, output_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-acodec",
        "pcm_s16le",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "wav",
        output_path,
    ]

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            err_text = stderr.decode("utf-8", errors="ignore")[:1000]
            await _ctx_error(ctx, f"ffmpeg failed: {err_text}")
            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except Exception:
                    pass
            raise McpError(ErrorData(code=-32010, message=f"ffmpeg failed to convert audio: {err_text}"))
        await _ctx_debug(ctx, "ffmpeg conversion completed")
        return output_path
    except FileNotFoundError:
        await _ctx_error(ctx, "ffmpeg not found in system")
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except Exception:
                pass
        raise McpError(ErrorData(code=-32011, message="ffmpeg not installed"))
    except McpError:
        raise
    except Exception as e:
        await _ctx_error(ctx, f"Unexpected ffmpeg error: {e}")
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except Exception:
                pass
        raise McpError(ErrorData(code=-32012, message=str(e)))


# ---- OAuth token (cached) ----


async def _get_sber_token(ctx: Optional[Context]) -> str:
    """
    Get NGW OAuth token with caching.
    Expects env: SBER_OAUTH_CLIENT_ID, SBER_OAUTH_CLIENT_SECRET, optional SBER_OAUTH_URL, SBER_OAUTH_SCOPE
    """
    global _cached_token, _cached_expiry
    async with _token_lock:
        now = time.time()
        if _cached_token and _cached_expiry and now < _cached_expiry:
            await _ctx_debug(ctx, "Using cached Sber token")
            return _cached_token

        await _ctx_info(ctx, "Requesting new Sber OAuth token")
        client_id = os.getenv("SBER_OAUTH_CLIENT_ID")
        client_secret = os.getenv("SBER_OAUTH_CLIENT_SECRET")
        if not client_id or not client_secret:
            raise McpError(ErrorData(code=-32020, message="SBER_OAUTH_CLIENT_ID or SBER_OAUTH_CLIENT_SECRET not set"))

        credentials = f"{client_id}:{client_secret}"
        encoded = base64.b64encode(credentials.encode()).decode()
        headers = {
            "Authorization": f"Basic {encoded}",
            "RqUID": str(uuid.uuid4()),
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        }
        data = {"scope": os.getenv("SBER_OAUTH_SCOPE", "SPEECH_RECOGNIZER")}
        oauth_url = os.getenv("SBER_OAUTH_URL", DEFAULT_OAUTH_URL)
        verify_ssl = _parse_bool_env("SBER_VERIFY_SSL", "true")

        try:
            async with httpx.AsyncClient(timeout=30.0, verify=verify_ssl) as client:
                resp = await client.post(oauth_url, headers=headers, data=data)
                try:
                    resp.raise_for_status()
                except httpx.HTTPStatusError as e:
                    body = await resp.aread()
                    await _ctx_error(ctx, f"Sber OAuth HTTP error: {e.response.status_code} body={body[:1000]}")
                    raise McpError(ErrorData(code=-32021, message=f"Sber OAuth failed: {e.response.status_code}"))
                js = resp.json()
                token = js.get("access_token")
                expires_in = int(js.get("expires_in", 3600))
                if not token:
                    await _ctx_error(ctx, "Sber OAuth did not return access_token")
                    raise McpError(ErrorData(code=-32022, message="No access_token in OAuth response"))
                _cached_token = token
                _cached_expiry = time.time() + expires_in - 30  # safety margin
                await _ctx_debug(ctx, "Cached new Sber token")
                return token
        except McpError:
            raise
        except Exception as e:
            await _ctx_error(ctx, f"Error obtaining Sber OAuth token: {e}")
            raise McpError(ErrorData(code=-32023, message=f"Error obtaining Sber token: {e}"))


# ---- Upload / create task / poll / download ----


async def _upload_file_to_salute(token: str, wav_path: str, ctx: Optional[Context], timeout: float = 120.0) -> str:
    """
    Upload file body to /data:upload -> returns request_file_id (or file_id)
    """
    base = os.getenv("SBER_SALUTESPEECH_BASE_URL", DEFAULT_SALUTESPEECH_BASE)
    upload_path = os.getenv("SBER_UPLOAD_PATH", DEFAULT_UPLOAD_PATH)
    url = base.rstrip("/") + upload_path
    headers = {"Authorization": f"Bearer {token}"}
    await _ctx_debug(ctx, f"Uploading file to SaluteSpeech: {url}")
    verify_ssl = _parse_bool_env("SBER_VERIFY_SSL", "true")

    try:
        async with httpx.AsyncClient(timeout=timeout, verify=verify_ssl) as client:
            async with aiofiles.open(wav_path, "rb") as f:
                data = await f.read()
            resp = await client.post(url, headers=headers, content=data)
            try:
                resp.raise_for_status()
            except httpx.HTTPStatusError as e:
                body = resp.text[:1000]
                await _ctx_error(ctx, f"Upload HTTP error {e.response.status_code}: {body}")
                raise McpError(ErrorData(code=-32050, message=f"Upload failed: {e.response.status_code}"))
            js = resp.json()
            # support multiple shapes
            result_obj = js.get("result", {}) or {}
            file_id = js.get("file_id") or js.get("id") or result_obj.get("request_file_id") or result_obj.get("file_id")
            if not file_id:
                await _ctx_error(ctx, f"Upload response missing file id: {js}")
                raise McpError(ErrorData(code=-32051, message="Upload did not return file id"))
            await _ctx_debug(ctx, f"Uploaded file id: {file_id}")
            return file_id
    except McpError:
        raise
    except Exception as e:
        await _ctx_error(ctx, f"Error uploading file: {e}")
        raise McpError(ErrorData(code=-32052, message=f"Error uploading file: {e}"))


async def _create_recognition_task(
    token: str,
    request_file_id: str,
    ctx: Optional[Context],
    enable_speaker_separation: bool = True,
    expected_speakers: int = 2,
    timeout: float = 30.0,
) -> str:
    """
    Create async recognition task and return task id.
    Payload follows SaluteSpeech async API contract (options + speaker_separation_options).
    """
    base = os.getenv("SBER_SALUTESPEECH_BASE_URL", DEFAULT_SALUTESPEECH_BASE)
    create_path = os.getenv("SBER_CREATE_TASK_PATH", DEFAULT_CREATE_TASK_PATH)
    url = base.rstrip("/") + create_path
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload: Dict[str, Any] = {
        "request_file_id": request_file_id,
        "options": {
            "model": os.getenv("SBER_RECOGNITION_MODEL", "general"),
            "audio_encoding": os.getenv("SBER_AUDIO_ENCODING", "PCM_S16LE"),
            "sample_rate": int(os.getenv("SBER_SAMPLE_RATE", "16000")),
            "language": os.getenv("SBER_RECOGNITION_LANGUAGE", "ru-RU"),
            "hypotheses_count": int(os.getenv("SBER_HYPOTHESES_COUNT", "1")),
            "normalization_options": {"enable": True, "punctuation": True, "capitalization": True},
            "enable_profanity_filter": _parse_bool_env("SBER_PROFANITY_FILTER", "false"),
            "no_speech_timeout": str(os.getenv("SBER_NO_SPEECH_TIMEOUT", "2")) + "s",
            "max_speech_timeout": str(os.getenv("SBER_MAX_SPEECH_TIMEOUT", "2")) + "s",
            "channels_count": int(os.getenv("SBER_CHANNELS_COUNT", "1")),
        },
    }
    if enable_speaker_separation:
        payload["options"]["speaker_separation_options"] = {
            "enable": True,
            "count": int(os.getenv("SBER_SPEAKER_COUNT", str(expected_speakers))),
            "enable_only_main_speaker": False,
        }

    await _ctx_debug(ctx, f"Creating recognition task at {url} payload keys: {list(payload.keys())}")
    verify_ssl = _parse_bool_env("SBER_VERIFY_SSL", "true")

    try:
        async with httpx.AsyncClient(timeout=timeout, verify=verify_ssl) as client:
            resp = await client.post(url, headers=headers, json=payload)
            try:
                resp.raise_for_status()
            except httpx.HTTPStatusError as e:
                body = resp.text[:1000]
                await _ctx_error(ctx, f"Create task HTTP error {e.response.status_code}: {body}")
                raise McpError(ErrorData(code=-32060, message=f"Create task failed: {e.response.status_code}"))
            js = resp.json()
            result_obj = js.get("result", {}) or {}
            task_id = js.get("task_id") or js.get("id") or result_obj.get("task_id") or result_obj.get("id")
            if not task_id:
                await _ctx_error(ctx, f"Create task response missing task id: {js}")
                raise McpError(ErrorData(code=-32061, message="Create task did not return task id"))
            await _ctx_debug(ctx, f"Created task id: {task_id}")
            return task_id
    except McpError:
        raise
    except Exception as e:
        await _ctx_error(ctx, f"Error creating recognition task: {e}")
        raise McpError(ErrorData(code=-32062, message=f"Error creating task: {e}"))


async def _poll_task_until_done(
    token: str,
    task_id: str,
    ctx: Optional[Context],
    max_poll_time: int = 300,
    poll_interval: int = 3,
) -> Dict[str, Any]:
    """
    Poll task:get until status DONE. Returns dict containing at least:
      {"task_id": ..., "response_file_id": "..."} — response_file_id here will be filled from uploaded file_id by caller.
    """
    base = os.getenv("SBER_SALUTESPEECH_BASE_URL", DEFAULT_SALUTESPEECH_BASE)
    status_path = os.getenv("SBER_TASK_STATUS_PATH", DEFAULT_TASK_STATUS_PATH)
    url = base.rstrip("/") + status_path
    headers = {"Authorization": f"Bearer {token}"}
    verify_ssl = _parse_bool_env("SBER_VERIFY_SSL", "true")
    start_time = time.time()
    params = {"id": task_id}

    await _ctx_debug(ctx, f"Polling task status at {url} with ID={task_id}")
    try:
        async with httpx.AsyncClient(timeout=20.0, verify=verify_ssl) as client:
            while time.time() - start_time < max_poll_time:
                resp = await client.get(url, headers=headers, params=params)
                try:
                    resp.raise_for_status()
                except httpx.HTTPStatusError as e:
                    body = resp.text[:1000]
                    await _ctx_error(ctx, f"Task status HTTP error {e.response.status_code}: {body}")
                    raise McpError(ErrorData(code=-32063, message=f"Task status failed: {e.response.status_code}"))
                js = resp.json()
                # status likely in js["result"]["status"] or js["result"]["state"]
                task_status = (js.get("result") or {}).get("status") or (js.get("result") or {}).get("state") or js.get("status")
                await _ctx_debug(ctx, f"Task {task_id} status: {task_status} (raw response keys: {list(js.keys())})")
                if not task_status:
                    await _ctx_debug(ctx, f"No explicit recognition status found in task response; full response: {js}")
                else:
                    norm = str(task_status).upper()
                    if norm in ("DONE", "COMPLETED", "FINISHED"):
                        # Caller will use uploaded file_id as response_file_id
                        return {"task_id": task_id, "task_response": js, "response_file_id": js.get("result").get("response_file_id")}
                    if norm in ("FAILED", "ERROR", "CANCELED"):
                        await _ctx_error(ctx, f"Recognition task {task_id} failed: {js}")
                        raise McpError(ErrorData(code=-32064, message=f"Recognition task failed: {js}"))
                await asyncio.sleep(poll_interval)
    except McpError:
        raise
    except Exception as e:
        await _ctx_error(ctx, f"Error polling task status: {e}")
        raise McpError(ErrorData(code=-32073, message=f"Error polling task: {e}"))

    raise McpError(ErrorData(code=-32065, message=f"Recognition task {task_id} timed out after {max_poll_time} seconds"))


async def _download_task_result(
    token: str,
    response_file_id: str,
    ctx: Optional[Context],
    timeout: float = 120.0,
) -> Dict[str, Any]:
    """
    Download the result file by response_file_id using /data:download?response_file_id=<response_file_id>.
    Handles:
      - application/zip (extract first .json inside)
      - application/octet-stream (try to decode as utf-8 JSON)
      - application/json
    Returns parsed JSON dict (the SaluteSpeech result).
    """

    base = os.getenv("SBER_SALUTESPEECH_BASE_URL", DEFAULT_SALUTESPEECH_BASE)
    download_path = os.getenv("SBER_TASK_RESULT_PATH", DEFAULT_DATA_DOWNLOAD_PATH)
    url = base.rstrip("/") + download_path
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/octet-stream"}
    params = {"response_file_id": response_file_id}
    verify_ssl = _parse_bool_env("SBER_VERIFY_SSL", "true")

    await _ctx_debug(ctx, f"Downloading result file for response_file_id={response_file_id} from {url}")

    try:
        async with httpx.AsyncClient(timeout=timeout, verify=verify_ssl) as client:
            resp = await client.get(url, headers=headers, params=params)

            try:
                resp.raise_for_status()
            except httpx.HTTPStatusError as e:
                body = resp.text[:1000]
                await _ctx_error(ctx, f"Result download HTTP error {e.response.status_code}: {body}")
                raise McpError(ErrorData(code=-32081, message=f"Result download failed: {e.response.status_code}: {body}"))

            content_type = (resp.headers.get("Content-Type") or "").lower()
            content = resp.content

            await _ctx_debug(
                ctx,
                f"Sber raw content-type={content_type}, size={len(content)} bytes"
            )

            #
            # ---------- ZIP ----------
            #
            if "application/zip" in content_type or (len(content) > 4 and content[:4] == b'PK\x03\x04'):
                await _ctx_debug(ctx, "Result is ZIP archive; extracting JSON inside")

                try:
                    zf = zipfile.ZipFile(io.BytesIO(content))

                    json_name = None
                    for name in zf.namelist():
                        if name.endswith(".json"):
                            json_name = name
                            break

                    # fallback: grab first file
                    if not json_name:
                        json_name = zf.namelist()[0]

                    raw = zf.read(json_name)
                    parsed = resp.json()

                    await _ctx_debug(
                        ctx,
                        f"Sber parsed JSON from ZIP: {json.dumps(parsed, ensure_ascii=False)[:4000]}"
                    )

                    return parsed

                except Exception as e:
                    await _ctx_error(ctx, f"Failed to extract JSON from zip result: {e}")
                    raise McpError(ErrorData(code=-32083, message=f"Zip received but cannot extract JSON: {e}"))

            #
            # ---------- OCTET-STREAM ----------
            #
            if "application/octet-stream" in content_type or content_type == "" or content_type.startswith("application/"):
                try:
                    return resp.json()
                except Exception as e:
                    await _ctx_error(ctx, f"Failed to parse octet-stream JSON: {e}")
                    raise McpError(ErrorData(code=-32084, message=f"Downloaded result is not valid JSON: {e}"))

            #
            # ---------- JSON ----------
            #
            try:
                parsed = resp.json()

                await _ctx_debug(
                    ctx,
                    f"Sber parsed JSON (resp.json): {json.dumps(parsed, ensure_ascii=False)[:4000]}"
                )

                return parsed

            except Exception:
                await _ctx_error(ctx, "Result download returned unknown content-type and cannot parse JSON")
                raise McpError(ErrorData(code=-32085, message="Unknown result format"))

    except McpError:
        raise
    except Exception as e:
        await _ctx_error(ctx, f"Error downloading result: {e}")
        raise McpError(ErrorData(code=-32082, message=f"Error downloading result: {e}"))


# ---- normalize result -> API contract ----


def _normalize_salute_result(result_json: Any) -> Dict[str, Any]:
    """
    Normalizes SaluteSpeech async result (list of chunks) into canonical API format:
    {
        "status": "completed",
        "full_text": "...",
        "duration_seconds": float,
        "segments": [
            { speaker, start_time, end_time, text }
        ]
    }
    """

    # Result is ALWAYS a list of chunks
    if not isinstance(result_json, list):
        # fallback: try inside "result"
        if isinstance(result_json, dict) and "result" in result_json:
            result_json = result_json["result"]
        else:
            result_json = [result_json]

    segments = []
    full_text_parts = []

    for chunk in result_json:
        results = chunk.get("results", [])
        speaker_id = (chunk.get("speaker_info") or {}).get("speaker_id", -1)

        for r in results:
            text = r.get("normalized_text") or r.get("text") or ""
            start_raw = r.get("start") or "0s"
            end_raw = r.get("end") or start_raw

            # convert "2.673999872s" → 2.673999872
            try:
                start = float(start_raw.replace("s", ""))
                end = float(end_raw.replace("s", ""))
            except:
                start, end = 0.0, 0.0

            segments.append({
                "speaker": f"speaker_{speaker_id}",
                "start_time": start,
                "end_time": end,
                "text": text,
            })

            if text:
                full_text_parts.append(text)

    duration_seconds = 0.0
    if segments:
        duration_seconds = max(s["end_time"] for s in segments)

    full_text = " ".join(full_text_parts).strip()

    return {
        "status": "completed",
        "full_text": full_text,
        "duration_seconds": duration_seconds,
        "segments": segments,
    }


# ---- MCP tool ----


@mcp.tool(
    name="transcribe_audio",
    description="Transcribe audio via SaluteSpeech async REST API (upload -> create task -> poll -> download). Normalizes output to canonical contract."
)
async def transcribe_audio(
    audio_url: str = Field(..., description="URL to audio file (http/https)"),
    enable_speaker_separation: bool = Field(True, description="Enable speaker separation (diarization)"),
    expected_speakers: int = Field(2, description="Hint: expected number of speakers"),
    ctx: Optional[Context] = None,
) -> ToolResult:
    # validate envs
    try:
        _require_env_vars(_ENV_VARS)
    except Exception:
        raise

    start_time = time.time()
    status_label = "error"
    await _ctx_info(ctx, f"Start transcription for {audio_url}")
    if ctx:
        await ctx.report_progress(progress=0, total=100)

    input_path: Optional[str] = None
    wav_path: Optional[str] = None
    file_id: Optional[str] = None
    task_id: Optional[str] = None
    response_file_id: Optional[str] = None

    try:
        with tracer.start_as_current_span("call_processor.transcribe") as span:
            span.set_attribute("audio_url", audio_url)

            # 1) download
            await _ctx_info(ctx, "⬇️ Downloading audio")
            input_path = await _download_to_temp(audio_url, ctx)
            if ctx:
                await ctx.report_progress(progress=10, total=100)
            span.set_attribute("downloaded", True)

            # 2) convert to wav
            wav_path = await _ensure_wav(input_path, ctx)
            if ctx:
                await ctx.report_progress(progress=25, total=100)
            span.set_attribute("wav_path", wav_path)

            # 3) token
            token = await _get_sber_token(ctx)
            if ctx:
                await ctx.report_progress(progress=35, total=100)

            # 4) upload
            await _ctx_info(ctx, "📤 Uploading file to SaluteSpeech")
            file_id = await _upload_file_to_salute(token, wav_path, ctx)
            if ctx:
                await ctx.report_progress(progress=50, total=100)
            span.set_attribute("file_id", file_id)

            # 5) create task
            await _ctx_info(ctx, "🧾 Creating recognition task")
            task_id = await _create_recognition_task(token, file_id, ctx, enable_speaker_separation, expected_speakers)
            if ctx:
                await ctx.report_progress(progress=60, total=100)
            span.set_attribute("task_id", task_id)

            # 6) poll
            await _ctx_info(ctx, "⏳ Waiting for recognition to finish")
            poll_result = await _poll_task_until_done(token, task_id, ctx)
            # Per REST API, response_file_id for downloading result is the uploaded file id
            response_file_id = poll_result.get("response_file_id")
            if ctx:
                await ctx.report_progress(progress=85, total=100)
            span.set_attribute("response_file_id", response_file_id)

            # 7) download result
            await _ctx_info(ctx, "📥 Downloading result")
            result_json = await _download_task_result(token, response_file_id, ctx)
            if ctx:
                await ctx.report_progress(progress=95, total=100)

            # 8) normalize
            result_contract = _normalize_salute_result(result_json)

            await _ctx_info(ctx, "✅ Transcription completed")
            if ctx:
                await ctx.report_progress(progress=100, total=100)
            span.set_attribute("duration_seconds", result_contract.get("duration_seconds", 0.0))
            status_label = "success"

            return ToolResult(
                content=[TextContent(
                    type="text",
                    text="Transcription completed"
                )],
                structured_content=result_contract,
            )

    
    except McpError:
        TRANSCRIPTION_REQUESTS.labels(status="error").inc()
        raise
    except Exception as e:
        await _ctx_error(ctx, f"Unexpected error in transcription: {e}")
        TRANSCRIPTION_REQUESTS.labels(status="error").inc()
        raise McpError(ErrorData(code=-32099, message=f"Unexpected error: {e}"))
    finally:
        elapsed = time.time() - start_time
        TRANSCRIPTION_DURATION.observe(elapsed)
        if status_label == "success":
            TRANSCRIPTION_REQUESTS.labels(status="success").inc()
        # cleanup
        try:
            if input_path and os.path.exists(input_path):
                os.remove(input_path)
            if wav_path and wav_path != input_path and os.path.exists(wav_path):
                os.remove(wav_path)
        except Exception:
            await _ctx_debug(ctx, "Warning: failed to remove temp files")