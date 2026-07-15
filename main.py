import asyncio
import io
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, List

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

# --- LangChain & Unstructured Imports ---
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.messages import HumanMessage
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
from openai import OpenAI
from pinecone import Pinecone

try:
    from deepgram import AsyncDeepgramClient
except ImportError:
    AsyncDeepgramClient = None

# Load Environment Variables
load_dotenv(Path(__file__).with_name(".env"))

app = FastAPI(title="Earnings Call AI Backend - Multi-Modal Edition")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Configuration & State ---
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)
TEMP_DIR = DATA_DIR / "temp"
TEMP_DIR.mkdir(exist_ok=True)

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o"
ROUTER_MODEL = "gpt-4o-mini"
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

if OPENAI_API_KEY:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
else:
    openai_client = None
    embeddings = None

# Initialize Global Vector Store if Index is provided
vector_store = None
if PINECONE_INDEX_NAME and embeddings:
    vector_store = PineconeVectorStore(index_name=PINECONE_INDEX_NAME, embedding=embeddings)


# ==========================================
# PART 1: MULTI-MODAL RAG INGESTION PIPELINE
# ==========================================

def partition_document(file_path: str):
    """Extract elements from PDF using unstructured (hi_res)"""
    print(f"📄 Partitioning document: {file_path}")
    elements = partition_pdf(
        filename=file_path,
        strategy="hi_res",
        infer_table_structure=True,
        extract_image_block_types=["Image"],
        extract_image_block_to_payload=True
    )
    print(f"✅ Extracted {len(elements)} elements")
    return elements

def create_chunks_by_title(elements):
    """Create intelligent chunks using title-based strategy"""
    print("🔨 Creating smart chunks...")
    chunks = chunk_by_title(
        elements,
        max_characters=3000,
        new_after_n_chars=2400,
        combine_text_under_n_chars=500
    )
    print(f"✅ Created {len(chunks)} chunks")
    return chunks

def separate_content_types(chunk):
    """Analyze what types of content are in a chunk"""
    content_data = {
        'text': chunk.text,
        'tables': [],
        'images': [],
        'types': ['text']
    }
    if hasattr(chunk, 'metadata') and hasattr(chunk.metadata, 'orig_elements'):
        for element in chunk.metadata.orig_elements:
            element_type = type(element).__name__
            if element_type == 'Table':
                content_data['types'].append('table')
                table_html = getattr(element.metadata, 'text_as_html', element.text)
                content_data['tables'].append(table_html)
            elif element_type == 'Image':
                if hasattr(element, 'metadata') and hasattr(element.metadata, 'image_base64'):
                    content_data['types'].append('image')
                    content_data['images'].append(element.metadata.image_base64)
    content_data['types'] = list(set(content_data['types']))
    return content_data

def create_ai_enhanced_summary(text: str, tables: List[str], images: List[str]) -> str:
    """Create AI-enhanced summary for mixed content"""
    try:
        llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)
        prompt_text = f"You are creating a searchable description for document content retrieval.\n\nTEXT CONTENT:\n{text}\n\n"
        if tables:
            prompt_text += "TABLES:\n"
            for i, table in enumerate(tables):
                prompt_text += f"Table {i+1}:\n{table}\n\n"
        prompt_text += "Generate a comprehensive, searchable description covering key facts, main topics, and visual content analysis. SEARCHABLE DESCRIPTION:"

        message_content = [{"type": "text", "text": prompt_text}]
        for image_base64 in images:
            message_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
            })
        
        message = HumanMessage(content=message_content)
        response = llm.invoke([message])
        return response.content
    except Exception as e:
        print(f"❌ AI summary failed: {e}")
        return f"{text[:300]}..."

def summarise_chunks(chunks):
    """Process all chunks with AI Summaries"""
    print("🧠 Processing chunks with AI Summaries...")
    langchain_documents = []
    
    for i, chunk in enumerate(chunks):
        content_data = separate_content_types(chunk)
        if content_data['tables'] or content_data['images']:
            enhanced_content = create_ai_enhanced_summary(content_data['text'], content_data['tables'], content_data['images'])
        else:
            enhanced_content = content_data['text']
            
        doc = Document(
            page_content=enhanced_content,
            metadata={
                "original_content": json.dumps({
                    "raw_text": content_data['text'],
                    "tables_html": content_data['tables'],
                    "images_base64": content_data['images']
                })
            }
        )
        langchain_documents.append(doc)
    return langchain_documents

def run_complete_ingestion_pipeline(pdf_path: str):
    """Run the complete multi-modal RAG ingestion pipeline"""
    global vector_store
    
    print(f"🚀 Starting RAG Ingestion Pipeline for: {pdf_path}")
    
    # Extract, Chunk & Summarize
    print("🔨 Extracting and chunking document...")
    elements = partition_document(pdf_path)
    chunks = create_chunks_by_title(elements)
    
    print("🧠 Processing chunks with AI Summaries...")
    summarised_chunks = summarise_chunks(chunks)
    
    # Clean Metadata for Pinecone Limits
    print("🧹 Cleaning chunk metadata...")
    cleaned_chunks = []
    for chunk in summarised_chunks: 
        safe_metadata = {
            "source": chunk.metadata.get("source", pdf_path),
            "page_number": chunk.metadata.get("page_number", 1),
            "summary": "AI Summary Applied" 
        }

        if "original_content" in chunk.metadata:
            try:
                orig_data = json.loads(chunk.metadata["original_content"])
                orig_data["images_base64"] = [] 
                safe_metadata["original_content"] = json.dumps(orig_data)
            except Exception:
                pass
        
        cleaned_doc = Document(
            page_content=chunk.page_content,
            metadata=safe_metadata
        )
        cleaned_chunks.append(cleaned_doc)

    # Upload to Pinecone
    try:
        print(f"🔮 Storing {len(cleaned_chunks)} chunks in Pinecone...")
        vector_store.add_documents(documents=cleaned_chunks) 
        print("✅ Successfully stored in Pinecone!")
        return len(cleaned_chunks)
    except Exception as e:
        print(f"\n❌ PINECONE ERROR REVEALED:\n{e}")
        return 0
    

@app.post("/rag/upload")
async def upload_document(file: UploadFile = File(...)) -> JSONResponse:
    try:
        temp_file_path = TEMP_DIR / (file.filename or "upload.pdf")
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        chunks_added = run_complete_ingestion_pipeline(str(temp_file_path))
        os.remove(temp_file_path)
        
        return JSONResponse(status_code=200, content={"file_name": file.filename, "chunks_added": chunks_added})
    except Exception as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})


@app.post("/rag/clear")
async def clear_database() -> JSONResponse:
    """Endpoint to delete all vectors from the Pinecone index."""
    try:
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        if not pinecone_api_key or not PINECONE_INDEX_NAME:
            return JSONResponse(
                status_code=400, 
                content={"error": "Pinecone API Key or Index Name is missing in environment variables."}
            )

        print(f"🗑️ Attempting to clear index: {PINECONE_INDEX_NAME}...")
        pc = Pinecone(api_key=pinecone_api_key)
        index = pc.Index(PINECONE_INDEX_NAME)
        index.delete(delete_all=True, namespace="")
        print("✅ Database cleared successfully!")
        return JSONResponse(
            status_code=200, 
            content={"message": f"Successfully cleared all data from index: {PINECONE_INDEX_NAME}"}
        )
    except Exception as exc:
        print(f"❌ Failed to clear database: {exc}")
        return JSONResponse(status_code=500, content={"error": str(exc)})


# ==========================================
# PART 2: MULTI-MODAL GENERATION LOGIC
# ==========================================

def generate_final_answer(chunks, query: str) -> str:
    """Generate final answer using multimodal content and Vision LLM"""
    try:
        llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)
        prompt_text = f"Based on the following documents, please answer this question: {query}\n\nCONTENT TO ANALYZE:\n"
        
        for i, chunk in enumerate(chunks):
            prompt_text += f"--- Document {i+1} ---\n"
            if "original_content" in chunk.metadata:
                original_data = json.loads(chunk.metadata["original_content"])
                raw_text = original_data.get("raw_text", "")
                if raw_text:
                    prompt_text += f"TEXT:\n{raw_text}\n\n"
                
                tables_html = original_data.get("tables_html", [])
                if tables_html:
                    prompt_text += "TABLES:\n"
                    for j, table in enumerate(tables_html):
                        prompt_text += f"Table {j+1}:\n{table}\n\n"
            else:
                prompt_text += f"TEXT:\n{chunk.page_content}\n\n"
            prompt_text += "\n"
        
        prompt_text += "Please provide a clear, comprehensive answer using the text, tables, and images above. If the documents don't contain sufficient information, say you don't have enough information.\n\nANSWER:"

        message_content = [{"type": "text", "text": prompt_text}]
        for chunk in chunks:
            if "original_content" in chunk.metadata:
                original_data = json.loads(chunk.metadata["original_content"])
                images_base64 = original_data.get("images_base64", [])
                for image_base64 in images_base64:
                    message_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                    })
                    
        message = HumanMessage(content=message_content)
        response = llm.invoke([message])
        return response.content
    except Exception as e:
        print(f"❌ Answer generation failed: {e}")
        return "Sorry, I encountered an error while generating the answer."
    

def answer_question_with_rag(question: str) -> dict[str, Any]:
    question = (question or "").strip()
    if not question:
        raise ValueError("A question is required")
    if not vector_store:
        raise ValueError("Vector store is not initialized. Upload a document first.")

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    retrieved_chunks = retriever.invoke(question)
    
    answer = generate_final_answer(retrieved_chunks, question)
    context = [chunk.page_content for chunk in retrieved_chunks]
    return {"answer": answer, "context": context}

@app.post("/rag/query")
async def query_documents(payload: dict[str, Any]) -> JSONResponse:
    try:
        question = str(payload.get("question", "")).strip()
        result = answer_question_with_rag(question)
        return JSONResponse(status_code=200, content={"answer": result["answer"], "context": result["context"]})
    except Exception as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})


# ==========================================
# PART 3: WEBSOCKET & AUDIO STREAMING
# ==========================================

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

def extract_questions_from_transcript(transcript: str) -> list[str]:
    cleaned = (transcript or "").strip()
    if not cleaned:
        return []

    def normalize_question(raw: str) -> str:
        question = re.sub(r"\s+", " ", raw or "").strip(" -•,;:")
        if not question:
            return ""
        question = re.sub(r"^(?:also|and|so|well|uh|um|let me see|let's see|i mean|i guess|okay|right)\s*,?\s*", "", question, flags=re.IGNORECASE)
        question = question.strip(" -•,;:")
        if not question.endswith("?"):
            question = f"{question}?"
        return question.strip()

    if openai_client is None:
        fallback_questions = []
        for segment in re.split(r"(?<=[?.!])\s+", cleaned):
            if "?" not in segment:
                continue
            normalized = normalize_question(segment)
            if normalized and normalized not in fallback_questions:
                fallback_questions.append(normalized)
        return fallback_questions

    try:
        prompt = (
            "You are an extraction filter. Analyze the full transcript, remove conversational filler, and return a strict JSON array of every distinct question asked by the caller. "
            "Do not include greetings, small talk, or non-question statements. Return only valid JSON, e.g. [\"What is the revenue trend?\"]."
        )
        response = openai_client.chat.completions.create(
            model=ROUTER_MODEL,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": cleaned},
            ],
            temperature=0
        )
        answer_text = (response.choices[0].message.content or "").strip()
        if answer_text.startswith("[") and answer_text.endswith("]"):
            parsed = json.loads(answer_text)
            if isinstance(parsed, list):
                questions = [normalize_question(str(item)) for item in parsed if str(item).strip()]
                return [question for question in questions if question]
    except Exception as exc:
        print(f"[extract-questions][error] {exc}")

    fallback_questions = []
    for segment in re.split(r"(?<=[?.!])\s+", cleaned):
        if "?" not in segment:
            continue
        normalized = normalize_question(segment)
        if normalized and normalized not in fallback_questions:
            fallback_questions.append(normalized)
    return fallback_questions

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
    recording_active = False
    transcript_buffer = ""
    session_lock = asyncio.Lock()
    last_transcript_seen = ""

    async def relay_audio() -> None:
        nonlocal recording_active, transcript_buffer, last_transcript_seen
        try:
            while not shutdown_event.is_set():
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    break
                if message.get("bytes"):
                    if connection is not None and recording_active:
                        audio_bytes = message["bytes"]
                        await connection.send_media(audio_bytes)
                elif message.get("text"):
                    try:
                        payload = json.loads(message["text"])
                    except json.JSONDecodeError:
                        continue
                    event = str(payload.get("event", "")).strip()
                    if event == "start_recording":
                        async with session_lock:
                            transcript_buffer = ""
                            last_transcript_seen = ""
                            recording_active = True
                        await websocket.send_json({"event": "recording_started"})
                    elif event == "stop_recording":
                        async with session_lock:
                            recording_active = False
                        await websocket.send_json({"event": "recording_stopped"})
                    elif event == "process_transcript":
                        transcript_to_process = str(payload.get("transcript", "") or "").strip()
                        async with session_lock:
                            if transcript_to_process:
                                transcript_buffer = transcript_to_process
                            else:
                                transcript_to_process = transcript_buffer.strip()
                            recording_active = False
                        if transcript_to_process:
                            await process_transcript(transcript_to_process)
                        else:
                            await websocket.send_json({"event": "empty_transcript"})
        except WebSocketDisconnect:
            print("[ws] client disconnected")
        except asyncio.CancelledError:
            pass
        except Exception as exc: 
            print(f"[ws][audio-error] {exc}")
        finally:
            shutdown_event.set()

    async def consume_transcripts() -> None:
        nonlocal connection
        await consume_transcript_messages(connection, transcript_queue)

    async def emit_transcripts() -> None:
        nonlocal transcript_buffer, last_transcript_seen
        while not shutdown_event.is_set():
            try:
                transcript = await asyncio.wait_for(transcript_queue.get(), timeout=0.5)
                if transcript:
                    async with session_lock:
                        if transcript != last_transcript_seen:
                            if recording_active:
                                transcript_buffer = f"{transcript_buffer} {transcript}".strip()
                            last_transcript_seen = transcript
                    await websocket.send_json({"event": "transcript", "text": transcript})
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except WebSocketDisconnect:
                break
            except Exception as exc: 
                print(f"[deepgram][stream-error] {exc}")
                break

    async def process_transcript(transcript_text: str) -> None:
        try:
            questions = extract_questions_from_transcript(transcript_text)
            if not questions:
                await websocket.send_json({"event": "pending_questions", "questions": []})
                return
            await websocket.send_json({"event": "pending_questions", "questions": questions})
            for question in questions:
                # Triggers the New Multi-Modal LangChain/Pinecone RAG Pipeline
                result = answer_question_with_rag(question)
                # Sends structured JSON back to the frontend
                await websocket.send_json({"event": "answer", "question": question, "answer": result.get("answer", "")})
        except Exception as exc:  
            await websocket.send_json({"event": "processing_error", "error": str(exc)})

    try:
        await websocket.send_json({"event": "connected", "text": "Connected to streaming endpoint. Audio is being accepted."})
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
    except Exception as exc: 
        print(f"[ws][error] {exc}")
    finally:
        shutdown_event.set()
        if connection is not None:
            try:
                await connection.send_close_stream()
            except Exception: 
                pass
        try:
            await websocket.close()
        except Exception: 
            pass


@app.get("/", include_in_schema=False)
async def index():
    html_path = Path(__file__).parent / "index.html"
    return FileResponse(html_path)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)