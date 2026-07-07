import asyncio
import io
import json
import os
import re
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
from pinecone import Pinecone

load_dotenv(Path(__file__).with_name(".env"))


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
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

try:
    from deepgram import AsyncDeepgramClient
except ImportError:  # pragma: no cover - handled gracefully at runtime
    AsyncDeepgramClient = None

app = FastAPI(title="Earnings Call AI Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)
DOCS_DIR = DATA_DIR / "documents"
DOCS_DIR.mkdir(exist_ok=True)
INDEX_PATH = DATA_DIR / "faiss.index"
METADATA_PATH = DATA_DIR / "metadata.json"

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "").strip()
PINECONE_ENV = (os.getenv("PINECONE_ENVIRONMENT") or os.getenv("PINECONE_ENV") or "").strip()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

if OPENAI_API_KEY:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
else:
    openai_client = None

pinecone_client = None
if PINECONE_API_KEY and PINECONE_ENV and PINECONE_INDEX:
    pinecone_client = Pinecone(api_key=PINECONE_API_KEY)


def _chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return chunks


def _extract_pdf_text(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(page for page in pages if page).strip()


def _get_embedding(text: str) -> list[float]:
    if openai_client is None:
        raise RuntimeError("OPENAI_API_KEY is not set")
    response = openai_client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return response.data[0].embedding


def _load_index_state() -> tuple[faiss.IndexFlatL2 | None, list[dict[str, Any]]]:
    if not INDEX_PATH.exists() or not METADATA_PATH.exists():
        return None, []
    index = faiss.read_index(str(INDEX_PATH))
    with METADATA_PATH.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    return index, metadata


def _save_index_state(index: faiss.IndexFlatL2, metadata: list[dict[str, Any]]) -> None:
    faiss.write_index(index, str(INDEX_PATH))
    with METADATA_PATH.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def _upsert_to_pinecone(file_name: str, chunks: list[str], embeddings: list[list[float]]) -> None:
    if not pinecone_client or not PINECONE_INDEX:
        return
    try:
        index = pinecone_client.Index(PINECONE_INDEX)
        vectors = []
        for chunk, embedding in zip(chunks, embeddings, strict=False):
            vectors.append((f"{file_name}:{len(vectors)}", embedding, {"source": file_name, "text": chunk}))
        if vectors:
            index.upsert(vectors=vectors)
    except Exception as exc:  # pragma: no cover - runtime logging path
        print(f"[pinecone][upsert-error] {exc}")


def ingest_document(file_name: str, file_bytes: bytes) -> dict[str, Any]:
    if file_name.lower().endswith(".pdf"):
        text = _extract_pdf_text(file_bytes)
    else:
        text = file_bytes.decode("utf-8", errors="ignore")

    chunks = _chunk_text(text)
    if not chunks:
        raise ValueError("No text could be extracted from the uploaded document")

    index, metadata = _load_index_state()
    if index is None:
        dimension = 1536
        index = faiss.IndexFlatL2(dimension)
    embeddings = []
    for chunk in chunks:
        embedding = _get_embedding(chunk)
        embeddings.append(embedding)

    vectors = np.array(embeddings, dtype="float32")
    index.add(vectors)

    for chunk in chunks:
        metadata.append({"source": file_name, "text": chunk})

    _save_index_state(index, metadata)
    _upsert_to_pinecone(file_name, chunks, embeddings)
    return {"file_name": file_name, "chunks_added": len(chunks)}


@app.post("/rag/upload")
async def upload_document(file: UploadFile = File(...)) -> JSONResponse:
    try:
        contents = await file.read()
        result = ingest_document(file.filename or "upload.txt", contents)
        return JSONResponse(status_code=200, content=result)
    except Exception as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})


@app.post("/rag/query")
async def query_documents(payload: dict[str, Any]) -> JSONResponse:
    try:
        question = str(payload.get("question", "")).strip()
        if not question:
            raise ValueError("A question is required")
        if openai_client is None:
            raise RuntimeError("OPENAI_API_KEY is not set")

        if pinecone_client and PINECONE_INDEX:
            query_embedding = _get_embedding(question)
            index = pinecone_client.Index(PINECONE_INDEX)
            response = index.query(vector=query_embedding, top_k=4, include_metadata=True)
            context_chunks = [match["metadata"]["text"] for match in response.get("matches", []) if match.get("metadata")]
        else:
            index, metadata = _load_index_state()
            if index is None or not metadata:
                raise ValueError("No documents have been indexed yet")
            query_embedding = _get_embedding(question)
            query_vector = np.array([query_embedding], dtype="float32")
            _, indices = index.search(query_vector, min(4, len(metadata)))
            context_chunks = [metadata[int(i)]["text"] for i in indices[0] if int(i) < len(metadata)]

        context = "\n\n".join(context_chunks)

        response = openai_client.responses.create(
            model=CHAT_MODEL,
            input=[
                {"role": "system", "content": "You answer questions using the provided document context. If the information is not present, say you do not know."},
                {"role": "user", "content": f"Question: {question}\n\nContext:\n{context}"},
            ],
        )
        answer = response.output_text
        return JSONResponse(status_code=200, content={"answer": answer, "context": context_chunks})
    except Exception as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})


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




