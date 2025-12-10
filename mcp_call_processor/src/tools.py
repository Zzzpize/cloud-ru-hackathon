# mcp_call_processor/tools.py
"""
Transcribe tool for Sber SaluteSpeech with:
- async ffmpeg conversion to WAV (16kHz mono)
- OAuth token caching with asyncio.Lock
- optional diarization and merging words->speakers
- robust parsing and fallbacks
- Prometheus metrics + OpenTelemetry span
- all logs via ctx (if provided)
- all user errors raised as McpError(ErrorData(...))
"""

import asyncio
import base64
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

# Required env vars
_ENV_VARS = [
    "SBER_CLIENT_ID",
    "SBER_CLIENT_SECRET",
    # Optional overrides:
    # SBER_OAUTH_URL, SBER_STT_URL, SBER_DIARIZATION_URL, SBER_VERIFY_SSL, SBER_DIARIZATION_ENABLED
]

# Default endpoints
DEFAULT_OAUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
DEFAULT_STT_URL = "https://smartspeech.sber.ru/rest/v1/stt:recognize"
DEFAULT_DIAR_URL = "https://smartspeech.sber.ru/rest/v1/diarization:recognize"

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
    """
    global _cached_token, _cached_expiry
    async with _token_lock:
        now = time.time()
        if _cached_token and _cached_expiry and now < _cached_expiry:
            await _ctx_safe_debug(ctx, "Using cached Sber token")
            return _cached_token

        await _ctx_safe_info(ctx, "Requesting new Sber OAuth token")
        client_id = os.getenv("SBER_CLIENT_ID")
        client_secret = os.getenv("SBER_CLIENT_SECRET")
        if not client_id or not client_secret:
            raise McpError(ErrorData(code=-32020, message="SBER_CLIENT_ID or SBER_CLIENT_SECRET not set"))

        credentials = f"{client_id}:{client_secret}"
        encoded = base64.b64encode(credentials.encode()).decode()
        headers = {
            "Authorization": f"Basic {encoded}",
            "RqUID": str(uuid.uuid4()),
            "Content-Type": "application/x-www-form-urlencoded",
        }
        data = {"scope": "SPEECH_RECOGNIZER"}
        oauth_url = os.getenv("SBER_OAUTH_URL", DEFAULT_OAUTH_URL)
        verify_ssl = _parse_bool_env("SBER_VERIFY_SSL", "true")

        try:
            async with httpx.AsyncClient(timeout=30.0, verify=verify_ssl) as client:
                resp = await client.post(oauth_url, headers=headers, data=data)
                try:
                    resp.raise_for_status()
                except httpx.HTTPStatusError as e:
                    await _ctx_safe_error(ctx, f"Sber OAuth HTTP error: {e.response.status_code}")
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


async def _call_sber_stt(token: str, wav_path: str, ctx: Optional[Context], timeout: float = 120.0) -> Dict[str, Any]:
    stt_url = os.getenv("SBER_STT_URL", DEFAULT_STT_URL)
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "audio/wav"}
    await _ctx_safe_debug(ctx, f"Calling STT endpoint: {stt_url}")
    verify_ssl = _parse_bool_env("SBER_VERIFY_SSL", "true")

    try:
        async with httpx.AsyncClient(timeout=timeout, verify=verify_ssl) as client:
            async with aiofiles.open(wav_path, "rb") as f:
                data = await f.read()
            resp = await client.post(stt_url, headers=headers, content=data)
            try:
                resp.raise_for_status()
            except httpx.HTTPStatusError as e:
                await _ctx_safe_error(ctx, f"STT HTTP error: {e.response.status_code}")
                raise McpError(ErrorData(code=-32030, message=f"STT returned HTTP {e.response.status_code}"))
            return resp.json()
    except McpError:
        raise
    except Exception as e:
        await _ctx_safe_error(ctx, f"Error calling STT: {e}")
        raise McpError(ErrorData(code=-32031, message=f"Error calling STT: {e}"))


async def _call_sber_diarization(token: str, wav_path: str, ctx: Optional[Context], timeout: float = 120.0) -> Dict[str, Any]:
    diar_url = os.getenv("SBER_DIARIZATION_URL", DEFAULT_DIAR_URL)
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "audio/wav"}
    await _ctx_safe_debug(ctx, f"Calling Diarization endpoint: {diar_url}")
    verify_ssl = _parse_bool_env("SBER_VERIFY_SSL", "true")

    try:
        async with httpx.AsyncClient(timeout=timeout, verify=verify_ssl) as client:
            async with aiofiles.open(wav_path, "rb") as f:
                data = await f.read()
            resp = await client.post(diar_url, headers=headers, content=data)
            try:
                resp.raise_for_status()
            except httpx.HTTPStatusError as e:
                await _ctx_safe_error(ctx, f"Diarization HTTP error: {e.response.status_code}")
                raise McpError(ErrorData(code=-32040, message=f"Diarization returned HTTP {e.response.status_code}"))
            return resp.json()
    except McpError:
        raise
    except Exception as e:
        await _ctx_safe_error(ctx, f"Error calling Diarization: {e}")
        raise McpError(ErrorData(code=-32041, message=f"Error calling Diarization: {e}"))


def _merge_words_and_diar(words: List[Dict[str, Any]], diar_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Assign speaker for each word by midpoint timestamp membership.
    Normalizes diar_segments and sorts them by start time.
    """
    normalized: List[Dict[str, Any]] = []
    for s in diar_segments:
        # support multiple key names
        start = s.get("start") or s.get("begin") or s.get("from") or 0.0
        end = s.get("end") or s.get("to") or s.get("until") or start
        speaker = s.get("speaker") or s.get("id") or s.get("label") or "S0"
        try:
            normalized.append({"start": float(start), "end": float(end), "speaker": str(speaker)})
        except Exception:
            continue

    # sort by start for robustness
    normalized.sort(key=lambda x: x["start"])

    merged: List[Dict[str, Any]] = []
    for w in words:
        try:
            w_start = float(w.get("start", 0.0))
        except Exception:
            w_start = 0.0
        try:
            w_end = float(w.get("end", w_start))
        except Exception:
            w_end = w_start
        mid = (w_start + w_end) / 2.0
        w_text = w.get("word") or w.get("text") or ""
        assigned = "unknown"
        for seg in normalized:
            if seg["start"] <= mid <= seg["end"]:
                assigned = seg["speaker"]
                break
        merged.append({
            "speaker": assigned,
            "start_time": w_start,
            "end_time": w_end,
            "text": w_text
        })
    return merged


@mcp.tool(
    name="transcribe_audio",
    description="Transcribe audio via Sber SaluteSpeech with optional diarization"
)
async def transcribe_audio(
    audio_url: str = Field(..., description="URL to audio file (http/https)"),
    ctx: Optional[Context] = None,
) -> ToolResult:
    """
    Main entrypoint: download -> convert -> STT -> (diar) -> merge -> ToolResult
    """
    # env presence
    try:
        _require_env_vars(_ENV_VARS)
    except Exception:
        # _require_env_vars should raise McpError; rethrow
        raise

    start_time = time.time()
    status_label = "error"
    await _ctx_safe_info(ctx, f"Start transcription for {audio_url}")
    await (ctx.report_progress(progress=0, total=100) if ctx else asyncio.sleep(0))

    diar_enabled = _parse_bool_env("SBER_DIARIZATION_ENABLED", "false")

    input_path: Optional[str] = None
    wav_path: Optional[str] = None

    try:
        with tracer.start_as_current_span("call_processor.transcribe") as span:
            span.set_attribute("audio_url", audio_url)

            # 1) download
            await _ctx_safe_info(ctx, "⬇️ Downloading audio")
            input_path = await _download_to_temp(audio_url, ctx)
            await (ctx.report_progress(progress=20, total=100) if ctx else asyncio.sleep(0))
            span.set_attribute("downloaded", True)

            # 2) convert to wav if needed
            wav_path = await _ensure_wav(input_path, ctx)
            await (ctx.report_progress(progress=40, total=100) if ctx else asyncio.sleep(0))
            span.set_attribute("wav_path", wav_path)

            # 3) token
            token = await _get_sber_token(ctx)
            await (ctx.report_progress(progress=55, total=100) if ctx else asyncio.sleep(0))

            # 4) STT
            await _ctx_safe_info(ctx, "📡 Calling Sber STT")
            stt_json = await _call_sber_stt(token, wav_path, ctx)
            await (ctx.report_progress(progress=75, total=100) if ctx else asyncio.sleep(0))

            # robust parsing of STT result
            full_text = ""
            words: List[Dict[str, Any]] = []
            try:
                if isinstance(stt_json, dict):
                    res = stt_json.get("result") or stt_json
                    hypotheses = res.get("hypotheses") if isinstance(res, dict) else None
                    if hypotheses and isinstance(hypotheses, list) and len(hypotheses) > 0:
                        hyp0 = hypotheses[0] or {}
                        full_text = hyp0.get("text") or res.get("text") or ""
                        words = hyp0.get("words") or res.get("words") or []
                    else:
                        # fallback to result.text or top-level text
                        full_text = res.get("text", "") or stt_json.get("text", "")
                        words = res.get("words", []) or stt_json.get("words", []) or []
                else:
                    full_text = ""
                    words = []
            except Exception:
                # never fail parsing
                full_text = stt_json.get("result", {}).get("text", "") if isinstance(stt_json, dict) else ""
                words = stt_json.get("result", {}).get("words", []) if isinstance(stt_json, dict) else []

            # Ensure a fallback if full_text still empty
            if not full_text and isinstance(stt_json, dict):
                full_text = stt_json.get("result", {}).get("text", "") or stt_json.get("text", "")

            # 5) diarization if enabled
            diar_segments: List[Dict[str, Any]] = []
            if diar_enabled:
                await _ctx_safe_info(ctx, "🗣️ Calling Sber Diarization")
                diar_json = await _call_sber_diarization(token, wav_path, ctx)
                if isinstance(diar_json, dict):
                    if "result" in diar_json and isinstance(diar_json["result"], dict) and "segments" in diar_json["result"]:
                        diar_segments = diar_json["result"].get("segments", []) or []
                    elif "segments" in diar_json:
                        diar_segments = diar_json.get("segments", []) or []
                    else:
                        # try to look for nested
                        diar_segments = diar_json.get("segments", []) or []
                else:
                    diar_segments = []

            # 6) merge
            if diar_segments and words:
                segments = _merge_words_and_diar(words, diar_segments)
            else:
                # fallback: each word is a segment
                segments = [
                    {
                        "speaker": "unknown",
                        "start_time": float(w.get("start", 0.0)) if w.get("start") is not None else 0.0,
                        "end_time": float(w.get("end", w.get("start", 0.0))) if w.get("end") is not None else float(w.get("start", 0.0) or 0.0),
                        "text": w.get("word") or w.get("text") or ""
                    }
                    for w in words
                ]

            # compute duration from segments if available
            duration_seconds = 0.0
            if segments:
                try:
                    # find max end_time (safer than taking last)
                    duration_seconds = max(float(s.get("end_time", 0.0)) for s in segments)
                except Exception:
                    duration_seconds = float(segments[-1].get("end_time", 0.0) or 0.0)

            result_json: Dict[str, Any] = {
                "status": "completed",
                "full_text": full_text,
                "duration_seconds": duration_seconds,
                "segments": segments,
            }

            await (ctx.report_progress(progress=100, total=100) if ctx else asyncio.sleep(0))
            await _ctx_safe_info(ctx, "✅ Transcription completed")
            span.set_attribute("duration_seconds", duration_seconds)
            status_label = "success"

            return ToolResult(
                content=[TextContent(type="text", text=f"Transcription status: completed\n{(full_text[:300] + '...') if len(full_text) > 300 else full_text}")],
                structured_content=result_json,
                meta={"audio_url": audio_url, "duration_seconds": duration_seconds},
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