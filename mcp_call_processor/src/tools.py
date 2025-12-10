# mcp_call_processor/tools.py
"""
Transcribe tool for SaluteSpeech (Smartspeech) — async REST flow:
- upload audio file (POST {BASE_URL}{UPLOAD_PATH})
- create async recognition task with speaker_separation_options enabled
- poll task status until finished
- download result JSON and convert to ToolResult (segments with speakers)

Requirements preserved:
- all functions async
- strict typing
- pydantic.Field for tool params
- ToolResult return
- ctx usage for logging
- McpError(ErrorData(...)) on user/expected errors
- oauth token caching
- Prometheus + OpenTelemetry instrumentation
"""

import asyncio
import base64
import json
import os
import time
import uuid
import tempfile
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
    "Histogram of transcription duration"
)

# Required env vars (must include oauth client credentials)
_ENV_VARS = [
    "SBER_OAUTH_CLIENT_ID",
    "SBER_OAUTH_CLIENT_SECRET",
    # optional: override base url / paths
]

# Default endpoints / paths (configurable via env)
DEFAULT_OAUTH_URL = os.getenv("SBER_OAUTH_URL", "https://ngw.devices.sberbank.ru:9443/api/v2/oauth")
DEFAULT_SALUTESPEECH_BASE = os.getenv("SBER_SALUTESPEECH_BASE_URL", "https://smartspeech.sber.ru/rest/v1")
DEFAULT_UPLOAD_PATH = os.getenv("SBER_UPLOAD_PATH", "/data:upload")
DEFAULT_CREATE_TASK_PATH = os.getenv("SBER_CREATE_TASK_PATH", "/speech:async_recognize")
DEFAULT_TASK_STATUS_PATH = os.getenv("SBER_TASK_STATUS_PATH", "/tasks/{task_id}")
DEFAULT_TASK_RESULT_PATH = os.getenv("SBER_TASK_RESULT_PATH", "/tasks/{task_id}/result")

# Token cache and lock
_token_lock: asyncio.Lock = asyncio.Lock()
_cached_token: Optional[str] = None
_cached_expiry: Optional[float] = None


def _parse_bool_env(name: str, default: str = "true") -> bool:
    v = os.getenv(name, default)
    if v is None:
        return default.lower() in ("1", "true", "yes")
    return str(v).strip().lower() in ("1", "true", "yes")


async def _ctx_safe_info(ctx: Optional[Context], msg: str) -> None:
    if ctx:
        await ctx.info(msg)


async def _ctx_safe_debug(ctx: Optional[Context], msg: str) -> None:
    if ctx:
        await ctx.debug(msg)


async def _ctx_safe_error(ctx: Optional[Context], msg: str) -> None:
    if ctx:
        await ctx.error(msg)


async def _download_to_temp(audio_url: str, ctx: Optional[Context], timeout: float = 120.0) -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False)
    tmp_path = tmp.name
    tmp.close()
    await _ctx_safe_debug(ctx, f"Downloading audio to temporary file: {tmp_path}")
    verify_ssl = _parse_bool_env("SBER_VERIFY_SSL", "true")
    try:
        async with httpx.AsyncClient(timeout=timeout, verify=verify_ssl) as client:
            async with client.stream("GET", audio_url) as resp:
                try:
                    resp.raise_for_status()
                except httpx.HTTPStatusError as e:
                    await _ctx_safe_error(ctx, f"HTTP error while downloading audio: {e.response.status_code}")
                    raise McpError(ErrorData(code=-32001, message=f"Failed to download audio: status {e.response.status_code}"))
                async with aiofiles.open(tmp_path, "wb") as f:
                    async for chunk in resp.aiter_bytes():
                        await f.write(chunk)
    except McpError:
        raise
    except Exception as e:
        await _ctx_safe_error(ctx, f"Error downloading audio: {e}")
        raise McpError(ErrorData(code=-32002, message=f"Error downloading audio: {e}"))
    return tmp_path


async def _ensure_wav(input_path: str, ctx: Optional[Context]) -> str:
    """
    Convert to wav (16kHz mono) asynchronously if needed.
    Returns path to wav file (may be same as input_path).
    """
    _, ext = os.path.splitext(input_path)
    ext = ext.lower()
    if ext == ".wav":
        await _ctx_safe_debug(ctx, "Input already WAV — skipping conversion")
        return input_path

    await _ctx_safe_info(ctx, "Converting audio to WAV (16kHz mono) via ffmpeg")
    fd, output_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-ac",
        "1",
        "-ar",
        "16000",
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
            await _ctx_safe_error(ctx, f"ffmpeg failed: {err_text}")
            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except Exception:
                    pass
            raise McpError(ErrorData(code=-32010, message=f"ffmpeg failed to convert audio: {err_text}"))
        await _ctx_safe_debug(ctx, "ffmpeg conversion completed")
        return output_path
    except FileNotFoundError:
        await _ctx_safe_error(ctx, "ffmpeg not found in system")
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except Exception:
                pass
        raise McpError(ErrorData(code=-32011, message="ffmpeg not installed"))
    except McpError:
        raise
    except Exception as e:
        await _ctx_safe_error(ctx, f"Unexpected ffmpeg error: {e}")
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except Exception:
                pass
        raise McpError(ErrorData(code=-32012, message=str(e)))


async def _get_sber_token(ctx: Optional[Context]) -> str:
    """
    Get OAuth token with caching (async lock).
    Uses env vars: SBER_OAUTH_CLIENT_ID, SBER_OAUTH_CLIENT_SECRET, optional SBER_OAUTH_URL
    """
    global _cached_token, _cached_expiry
    async with _token_lock:
        now = time.time()
        if _cached_token and _cached_expiry and now < _cached_expiry:
            await _ctx_safe_debug(ctx, "Using cached Sber token")
            return _cached_token

        await _ctx_safe_info(ctx, "Requesting new Sber OAuth token")
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
                    await _ctx_safe_error(ctx, f"Sber OAuth HTTP error: {e.response.status_code} body={body[:1000]}")
                    raise McpError(ErrorData(code=-32021, message=f"Sber OAuth failed: {e.response.status_code}"))
                js = resp.json()
                token = js.get("access_token")
                expires_in = int(js.get("expires_in", 3600))
                if not token:
                    await _ctx_safe_error(ctx, "Sber OAuth did not return access_token")
                    raise McpError(ErrorData(code=-32022, message="No access_token in OAuth response"))
                _cached_token = token
                _cached_expiry = time.time() + expires_in - 30  # safety margin
                await _ctx_safe_debug(ctx, "Cached new Sber token")
                return token
        except McpError:
            raise
        except Exception as e:
            await _ctx_safe_error(ctx, f"Error obtaining Sber OAuth token: {e}")
            raise McpError(ErrorData(code=-32023, message=f"Error obtaining Sber token: {e}"))


async def _upload_file_to_salute(token: str, wav_path: str, ctx: Optional[Context], timeout: float = 120.0) -> str:
    """
    Upload the audio file to SaluteSpeech async storage.
    Returns file_id (string) used to create task.
    """
    base = os.getenv("SBER_SALUTESPEECH_BASE_URL", DEFAULT_SALUTESPEECH_BASE)
    upload_path = os.getenv("SBER_UPLOAD_PATH", DEFAULT_UPLOAD_PATH)
    url = base.rstrip("/") + upload_path
    headers = {"Authorization": f"Bearer {token}"}
    await _ctx_safe_debug(ctx, f"Uploading file to SaluteSpeech: {url}")
    verify_ssl = _parse_bool_env("SBER_VERIFY_SSL", "true")

    try:
        async with httpx.AsyncClient(timeout=timeout, verify=verify_ssl) as client:
            async with aiofiles.open(wav_path, "rb") as f:
                data = await f.read()
            # per docs, the upload accepts binary audio in body
            resp = await client.post(url, headers=headers, content=data)
            try:
                resp.raise_for_status()
            except httpx.HTTPStatusError as e:
                body = resp.text[:1000]
                await _ctx_safe_error(ctx, f"Upload HTTP error {e.response.status_code}: {body}")
                raise McpError(ErrorData(code=-32050, message=f"Upload failed: {e.response.status_code}"))
            js = resp.json()
            # docs: returns an identifier for the uploaded file (field name may be 'file_id' or 'id')
            file_id = js.get("file_id") or js.get("id") or js.get("result", {}).get("file_id")
            if not file_id:
                await _ctx_safe_error(ctx, f"Upload response missing file id: {js}")
                raise McpError(ErrorData(code=-32051, message="Upload did not return file id"))
            await _ctx_safe_debug(ctx, f"Uploaded file id: {file_id}")
            return file_id
    except McpError:
        raise
    except Exception as e:
        await _ctx_safe_error(ctx, f"Error uploading file: {e}")
        raise McpError(ErrorData(code=-32052, message=f"Error uploading file: {e}"))


async def _create_recognition_task(token: str, file_id: str, ctx: Optional[Context], enable_speaker_separation: bool = True, expected_speakers: int = 2, timeout: float = 30.0) -> str:
    """
    Create async recognition task. Returns task_id (string).
    Payload includes speaker_separation_options per docs.
    See docs: async-general and POST create task. (speaker_separation_options enable -> diarization).
    """
    base = os.getenv("SBER_SALUTESPEECH_BASE_URL", DEFAULT_SALUTESPEECH_BASE)
    create_path = os.getenv("SBER_CREATE_TASK_PATH", DEFAULT_CREATE_TASK_PATH)
    url = base.rstrip("/") + create_path
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload: Dict[str, Any] = {
        "file_id": file_id,
        "language": os.getenv("SBER_RECOGNITION_LANGUAGE", "ru-RU"),
        "config": {
            "hypotheses_count": int(os.getenv("SBER_HYPOTHESES_COUNT", "1")),
            "normalization_options": {
                "enable": True,
                "punctuation": True,
                "capitalization": True
            },
        },
    }
    # add speaker separation options if requested
    if enable_speaker_separation:
        payload["config"]["speaker_separation_options"] = {
            "enable": True,
            "count": int(os.getenv("SBER_SPEAKER_COUNT", str(expected_speakers))),
            "enable_only_main_speaker": False
        }

    await _ctx_safe_debug(ctx, f"Creating recognition task at {url} payload keys: {list(payload.keys())}")
    verify_ssl = _parse_bool_env("SBER_VERIFY_SSL", "true")

    try:
        async with httpx.AsyncClient(timeout=timeout, verify=verify_ssl) as client:
            resp = await client.post(url, headers=headers, json=payload)
            try:
                resp.raise_for_status()
            except httpx.HTTPStatusError as e:
                body = resp.text[:1000]
                await _ctx_safe_error(ctx, f"Create task HTTP error {e.response.status_code}: {body}")
                raise McpError(ErrorData(code=-32060, message=f"Create task failed: {e.response.status_code}"))
            js = resp.json()
            task_id = js.get("task_id") or js.get("id") or js.get("result", {}).get("task_id")
            if not task_id:
                await _ctx_safe_error(ctx, f"Create task response missing task id: {js}")
                raise McpError(ErrorData(code=-32061, message="Create task did not return task id"))
            await _ctx_safe_debug(ctx, f"Created task id: {task_id}")
            return task_id
    except McpError:
        raise
    except Exception as e:
        await _ctx_safe_error(ctx, f"Error creating recognition task: {e}")
        raise McpError(ErrorData(code=-32062, message=f"Error creating task: {e}"))


async def _poll_task_until_done(token: str, task_id: str, ctx: Optional[Context], poll_interval: float = 2.0, timeout: float = 300.0) -> Dict[str, Any]:
    """
    Poll the task status until finished. Returns final task JSON (which should include link to result or result in body).
    """
    base = os.getenv("SBER_SALUTESPEECH_BASE_URL", DEFAULT_SALUTESPEECH_BASE)
    status_path_template = os.getenv("SBER_TASK_STATUS_PATH", DEFAULT_TASK_STATUS_PATH)
    url = base.rstrip("/") + status_path_template.format(task_id=task_id)
    headers = {"Authorization": f"Bearer {token}"}
    verify_ssl = _parse_bool_env("SBER_VERIFY_SSL", "true")
    started = time.time()
    await _ctx_safe_debug(ctx, f"Polling task status at {url}")
    try:
        async with httpx.AsyncClient(timeout=30.0, verify=verify_ssl) as client:
            while True:
                resp = await client.get(url, headers=headers)
                try:
                    resp.raise_for_status()
                except httpx.HTTPStatusError as e:
                    body = resp.text[:1000]
                    await _ctx_safe_error(ctx, f"Task status HTTP error {e.response.status_code}: {body}")
                    raise McpError(ErrorData(code=-32070, message=f"Task status failed: {e.response.status_code}"))
                js = resp.json()
                # docs: status field may be 'pending', 'running', 'completed', 'failed'
                status = js.get("status") or js.get("state") or ""
                await _ctx_safe_debug(ctx, f"Task {task_id} status: {status}")
                if status.lower() in ("completed", "done", "finished"):
                    return js
                if status.lower() in ("failed", "error"):
                    await _ctx_safe_error(ctx, f"Task {task_id} failed: {js}")
                    raise McpError(ErrorData(code=-32071, message="Recognition task failed"))
                if time.time() - started > timeout:
                    await _ctx_safe_error(ctx, f"Task {task_id} polling timeout after {timeout}s")
                    raise McpError(ErrorData(code=-32072, message="Recognition task timeout"))
                await asyncio.sleep(poll_interval)
    except McpError:
        raise
    except Exception as e:
        await _ctx_safe_error(ctx, f"Error polling task status: {e}")
        raise McpError(ErrorData(code=-32073, message=f"Error polling task: {e}"))


async def _download_task_result(token: str, task_json: Dict[str, Any], ctx: Optional[Context], timeout: float = 120.0) -> Dict[str, Any]:
    """
    Obtain final result JSON. The status response may already contain result or a download URL.
    Handles both possibilities.
    """
    # If result included directly
    if "result" in task_json and task_json["result"]:
        return task_json["result"]

    # otherwise, try to download via result path
    base = os.getenv("SBER_SALUTESPEECH_BASE_URL", DEFAULT_SALUTESPEECH_BASE)
    result_path_template = os.getenv("SBER_TASK_RESULT_PATH", DEFAULT_TASK_RESULT_PATH)
    task_id = task_json.get("task_id") or task_json.get("id")
    if not task_id:
        raise McpError(ErrorData(code=-32080, message="Task JSON missing id for result download"))
    url = base.rstrip("/") + result_path_template.format(task_id=task_id)
    headers = {"Authorization": f"Bearer {token}"}
    verify_ssl = _parse_bool_env("SBER_VERIFY_SSL", "true")
    await _ctx_safe_debug(ctx, f"Downloading result from {url}")
    try:
        async with httpx.AsyncClient(timeout=timeout, verify=verify_ssl) as client:
            resp = await client.get(url, headers=headers)
            try:
                resp.raise_for_status()
            except httpx.HTTPStatusError as e:
                body = resp.text[:1000]
                await _ctx_safe_error(ctx, f"Result download HTTP error {e.response.status_code}: {body}")
                raise McpError(ErrorData(code=-32081, message=f"Result download failed: {e.response.status_code}"))
            return resp.json()
    except McpError:
        raise
    except Exception as e:
        await _ctx_safe_error(ctx, f"Error downloading result: {e}")
        raise McpError(ErrorData(code=-32082, message=f"Error downloading result: {e}"))


def _convert_salute_result_to_contract(result_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Map SaluteSpeech result to the Tool API contract:
    {
      status: "completed",
      full_text: "...",
      duration_seconds: 185,
      segments: [
         {speaker, start_time, end_time, text}, ...
      ]
    }
    SaluteSpeech result format may contain word-level timing and speaker segments; handle common variants.
    """
    full_text = ""
    duration_seconds = 0.0
    segments: List[Dict[str, Any]] = []

    # try a few common structures
    # 1) result.hypotheses[0].text + hypotheses[0].words
    try:
        if isinstance(result_json, dict):
            hyp = None
            if "hypotheses" in result_json:
                hyps = result_json.get("hypotheses") or []
                if hyps:
                    hyp = hyps[0]
            elif "result" in result_json and isinstance(result_json["result"], dict):
                res = result_json["result"]
                hyps = res.get("hypotheses") or []
                if hyps:
                    hyp = hyps[0]
            if hyp:
                full_text = hyp.get("text", "") or result_json.get("text", "")
                words = hyp.get("words", []) or []
            else:
                # fallback to top-level text/words
                full_text = result_json.get("text", "") or result_json.get("full_text", "")
                words = result_json.get("words", []) or []
            # speaker-aligned segments often in result_json["segments"] or result_json["speaker_segments"]
            speaker_segs = result_json.get("segments") or result_json.get("speaker_segments") or result_json.get("speakerSeparation", [])
            if speaker_segs:
                # Normalise and map
                for s in speaker_segs:
                    sp = s.get("speaker") or s.get("label") or s.get("id") or "unknown"
                    start = float(s.get("start", s.get("begin", 0.0)))
                    end = float(s.get("end", s.get("to", start)))
                    text = s.get("text") or ""
                    segments.append({"speaker": sp, "start_time": start, "end_time": end, "text": text})
            else:
                # if words exist, aggregate contiguous words with same speaker (if speaker present in words)
                if words:
                    for w in words:
                        w_start = float(w.get("start", 0.0))
                        w_end = float(w.get("end", w_start))
                        w_text = w.get("word") or w.get("text") or ""
                        sp = w.get("speaker") or "unknown"
                        segments.append({"speaker": sp, "start_time": w_start, "end_time": w_end, "text": w_text})
            # compute duration
            if segments:
                try:
                    duration_seconds = max(float(s.get("end_time", 0.0)) for s in segments)
                except Exception:
                    duration_seconds = float(segments[-1].get("end_time", 0.0) or 0.0)
            else:
                duration_seconds = float(result_json.get("duration_seconds", 0.0) or 0.0)
    except Exception:
        # last-resort fallback
        full_text = result_json.get("text", "") or result_json.get("full_text", "")
        segments = []
        duration_seconds = float(result_json.get("duration_seconds", 0.0) or 0.0)

    return {
        "status": "completed",
        "full_text": full_text,
        "duration_seconds": duration_seconds,
        "segments": segments,
    }


@mcp.tool(
    name="transcribe_audio",
    description="Transcribe audio via SaluteSpeech async REST API (upload -> create task -> poll -> download). Uses speaker_separation_options for diarization."
)
async def transcribe_audio(
    audio_url: str = Field(..., description="URL to audio file (http/https)"),
    enable_speaker_separation: bool = Field(True, description="Enable speaker separation (diarization)"),
    expected_speakers: int = Field(2, description="Expected number of speakers (used as hint)"),
    ctx: Optional[Context] = None,
) -> ToolResult:
    # ensure envs
    try:
        _require_env_vars(_ENV_VARS)
    except Exception:
        raise

    start_time = time.time()
    status_label = "error"
    await _ctx_safe_info(ctx, f"Start transcription for {audio_url}")
    await (ctx.report_progress(progress=0, total=100) if ctx else asyncio.sleep(0))

    input_path: Optional[str] = None
    wav_path: Optional[str] = None

    try:
        with tracer.start_as_current_span("call_processor.transcribe") as span:
            span.set_attribute("audio_url", audio_url)

            # 1) download
            await _ctx_safe_info(ctx, "⬇️ Downloading audio")
            input_path = await _download_to_temp(audio_url, ctx)
            await (ctx.report_progress(progress=15, total=100) if ctx else asyncio.sleep(0))
            span.set_attribute("downloaded", True)

            # 2) convert to wav if needed
            wav_path = await _ensure_wav(input_path, ctx)
            await (ctx.report_progress(progress=30, total=100) if ctx else asyncio.sleep(0))
            span.set_attribute("wav_path", wav_path)

            # 3) get token
            token = await _get_sber_token(ctx)
            await (ctx.report_progress(progress=40, total=100) if ctx else asyncio.sleep(0))

            # 4) upload file
            await _ctx_safe_info(ctx, "📤 Uploading file to SaluteSpeech")
            file_id = await _upload_file_to_salute(token, wav_path, ctx)
            await (ctx.report_progress(progress=55, total=100) if ctx else asyncio.sleep(0))
            span.set_attribute("file_id", file_id)

            # 5) create task (with speaker separation options)
            await _ctx_safe_info(ctx, "🧾 Creating recognition task")
            task_id = await _create_recognition_task(token, file_id, ctx, enable_speaker_separation, expected_speakers)
            await (ctx.report_progress(progress=65, total=100) if ctx else asyncio.sleep(0))
            span.set_attribute("task_id", task_id)

            # 6) poll until done
            await _ctx_safe_info(ctx, "⏳ Waiting for recognition to finish")
            task_json = await _poll_task_until_done(token, task_id, ctx)
            await (ctx.report_progress(progress=90, total=100) if ctx else asyncio.sleep(0))

            # 7) download result (if not in status)
            result_json = await _download_task_result(token, task_json, ctx)
            await (ctx.report_progress(progress=98, total=100) if ctx else asyncio.sleep(0))

            # 8) convert result to our API contract
            result_contract = _convert_salute_result_to_contract(result_json)

            await (ctx.report_progress(progress=100, total=100) if ctx else asyncio.sleep(0))
            await _ctx_safe_info(ctx, "✅ Transcription completed")
            span.set_attribute("duration_seconds", result_contract.get("duration_seconds", 0.0))
            status_label = "success"

            return ToolResult(
                content=[TextContent(type="text", text=f"Transcription status: completed\n{(result_contract['full_text'][:300] + '...') if len(result_contract['full_text']) > 300 else result_contract['full_text']}")],
                structured_content=result_contract,
                meta={"audio_url": audio_url, "file_id": file_id, "task_id": task_id},
            )

    except McpError:
        TRANSCRIPTION_REQUESTS.labels(status="error").inc()
        raise
    except Exception as e:
        await _ctx_safe_error(ctx, f"Unexpected error in transcription: {e}")
        TRANSCRIPTION_REQUESTS.labels(status="error").inc()
        raise McpError(ErrorData(code=-32099, message=f"Unexpected error: {e}"))
    finally:
        elapsed = time.time() - start_time
        TRANSCRIPTION_DURATION.observe(elapsed)
        if status_label == "success":
            TRANSCRIPTION_REQUESTS.labels(status="success").inc()
        # cleanup temporary files
        try:
            if input_path and os.path.exists(input_path):
                os.remove(input_path)
            if wav_path and wav_path != input_path and os.path.exists(wav_path):
                os.remove(wav_path)
        except Exception:
            await _ctx_safe_debug(ctx, "Warning: failed to remove temp files")