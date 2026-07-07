import asyncio
import unittest

from main import consume_transcript_messages, extract_transcript_text


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
