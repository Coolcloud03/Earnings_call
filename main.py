import asyncio
import os
from typing import Any


async def consume_transcript_messages(connection: Any, transcript_queue: "asyncio.Queue[str]") -> None:
    try:
        async for message in connection:
            transcript = extract_transcript_text(message)
            if transcript:
                transcript_queue.put_nowait(transcript)
    except asyncio.CancelledError:
        pass
    except Exception as exc:  # pragma: no cover - runtime logging path
        print(f"[deepgram][transcript-error] {exc}")

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pathlib import Path

try:
    from deepgram import AsyncDeepgramClient
except ImportError:  # pragma: no cover - handled gracefully at runtime
    AsyncDeepgramClient = None

app = FastAPI(title="Earnings Call AI Backend")


@app.get("/", include_in_schema=False)
async def index():
    html_path = Path(__file__).parent / "index.html"
    return FileResponse(html_path)

# @app.get("/", include_in_schema=False)
# async def index() -> HTMLResponse:
#     html_path = os.path.join(os.path.dirname(__file__), "index.html")
#     with open(html_path, "r", encoding="utf-8") as handle:
#         return HTMLResponse(handle.read())
# above is the better way

def extract_transcript_text(payload: Any) -> str | None:
    if payload is None:
        return None

    def pick_text(candidate: Any) -> str | None:
        if candidate is None:
            return None
        if isinstance(candidate, dict):
            transcript = candidate.get("transcript")
            if isinstance(transcript, str) and transcript.strip():
                return transcript.strip()
            return None
        transcript = getattr(candidate, "transcript", None)
        if isinstance(transcript, str) and transcript.strip():
            return transcript.strip()
        return None

    if isinstance(payload, dict):
        for key in ("channel", "channels"):
            container = payload.get(key)
            if isinstance(container, dict):
                alternatives = container.get("alternatives") or []
                if isinstance(alternatives, list):
                    for item in alternatives:
                        text = pick_text(item)
                        if text:
                            return text
            elif isinstance(container, list):
                for item in container:
                    if isinstance(item, dict):
                        alternatives = item.get("alternatives") or []
                        if isinstance(alternatives, list):
                            for alt in alternatives:
                                text = pick_text(alt)
                                if text:
                                    return text
        return None

    channel = getattr(payload, "channel", None)
    if channel is not None:
        alternatives = getattr(channel, "alternatives", None) or []
        if isinstance(alternatives, list):
            for item in alternatives:
                text = pick_text(item)
                if text:
                    return text

    for attr_name in ("results", "alternatives"):
        container = getattr(payload, attr_name, None)
        if isinstance(container, list):
            for item in container:
                text = pick_text(item)
                if text:
                    return text

    return None


def load_api_key() -> str | None:
    api_key = os.getenv("DEEPGRAM_API_KEY")
    if api_key:
        return api_key

    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if not os.path.exists(env_path):
        return None

    try:
        with open(env_path, "r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                if key.strip() == "DEEPGRAM_API_KEY":
                    return value.strip().strip('"').strip("'") or None
    except OSError:
        return None

    return None


@app.websocket("/ws/stream")
async def stream_transcription(websocket: WebSocket) -> None:
    await websocket.accept()

    if AsyncDeepgramClient is None:
        await websocket.send_text("Deepgram SDK is not available. Install it with uv sync.")
        try:
            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    break
        except WebSocketDisconnect:
            print("[ws] client disconnected")
        return

    api_key = load_api_key()
    if not api_key:
        await websocket.send_text("No Deepgram API key was found. Add DEEPGRAM_API_KEY to your environment or a local .env file to enable transcription.")
        try:
            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    break
        except WebSocketDisconnect:
            print("[ws] client disconnected")
        return

    shutdown_event = asyncio.Event()
    transcript_queue: "asyncio.Queue[str]" = asyncio.Queue()
    connection: Any = None

    async def relay_audio() -> None:
        try:
            while not shutdown_event.is_set():
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    break
                if message.get("bytes"):
                    if connection is not None:
                        audio_bytes = message["bytes"]
                        print(f"[ws][audio] received {len(audio_bytes)} bytes")
                        await connection.send_media(audio_bytes)
        except WebSocketDisconnect:
            print("[ws] client disconnected")
        except asyncio.CancelledError:
            pass
        except Exception as exc:  # pragma: no cover - runtime logging path
            print(f"[ws][audio-error] {exc}")
        finally:
            shutdown_event.set()

    async def consume_transcripts() -> None:
        nonlocal connection
        await consume_transcript_messages(connection, transcript_queue)

    async def emit_transcripts() -> None:
        while not shutdown_event.is_set():
            try:
                transcript = await asyncio.wait_for(transcript_queue.get(), timeout=0.5)
                if transcript:
                    await websocket.send_text(transcript)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except WebSocketDisconnect:
                break
            except Exception as exc:  # pragma: no cover - runtime logging path
                print(f"[deepgram][stream-error] {exc}")
                break

    try:
        await websocket.send_text("Connected to streaming endpoint. Audio is being accepted.")
        deepgram_client = AsyncDeepgramClient(api_key=api_key)
        async with deepgram_client.listen.v1.connect(
            model="nova-2",
            language="en-US",
            punctuate=True,
            encoding="linear16",
            sample_rate=16000,
        ) as deepgram_connection:
            connection = deepgram_connection
            audio_task = asyncio.create_task(relay_audio())
            transcript_task = asyncio.create_task(consume_transcripts())
            emit_task = asyncio.create_task(emit_transcripts())
            await asyncio.gather(audio_task, transcript_task, emit_task, return_exceptions=True)
    except WebSocketDisconnect:
        print("[ws] client disconnected")
    except Exception as exc:  # pragma no cover - runtime logging path
        print(f"[ws][error] {exc}")
    finally:
        shutdown_event.set()
        if connection is not None:
            try:
                await connection.send_close_stream()
            except Exception:  # pragma: no cover - cleanup path
                pass
        try:
            await websocket.close()
        except Exception:  # pragma: no cover - cleanup path
            pass


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)




