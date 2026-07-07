import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from main import consume_transcript_messages, extract_transcript_text, ingest_document, _chunk_text


class RAGTests(unittest.TestCase):
    def test_chunk_text_splits_large_content(self) -> None:
        text = "word " * 1000
        chunks = _chunk_text(text, chunk_size=80, overlap=20)
        self.assertGreaterEqual(len(chunks), 2)

    def test_ingest_document_writes_local_index_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("main.INDEX_PATH", Path(temp_dir) / "faiss.index"), patch("main.METADATA_PATH", Path(temp_dir) / "metadata.json"), patch("main._get_embedding", return_value=[0.0] * 1536):
                result = ingest_document("sample.txt", b"hello world from rag")
            self.assertEqual(result["chunks_added"], 1)
            self.assertTrue((Path(temp_dir) / "faiss.index").exists())
            self.assertTrue((Path(temp_dir) / "metadata.json").exists())


class ConsumeTranscriptMessagesTests(unittest.IsolatedAsyncioTestCase):
    async def test_consume_transcript_messages_ignores_cancellation(self) -> None:
        class StubConnection:
            def __aiter__(self):
                return self

            async def __anext__(self):
                raise asyncio.CancelledError

        queue: "asyncio.Queue[str]" = asyncio.Queue()

        await consume_transcript_messages(StubConnection(), queue)

        self.assertEqual(queue.qsize(), 0)

    def test_extract_transcript_text_handles_nested_payloads(self) -> None:
        payload = {"channel": {"alternatives": [{"transcript": "hello"}]}}
        self.assertEqual(extract_transcript_text(payload), "hello")


if __name__ == "__main__":
    unittest.main()
