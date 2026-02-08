import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

from chunking import naive_chunk
from generate import build_prompt, generate_answer
from retrieve import ensure_indexed, get_client, get_collection, search


class TestChunking(unittest.TestCase):
    def test_naive_chunk_splits_paragraphs(self):
        text = "A first paragraph.\n\nA second paragraph."
        chunks = naive_chunk(text, source="doc.txt")
        self.assertEqual(len(chunks), 2)
        self.assertIn("id", chunks[0])
        self.assertEqual(chunks[0]["metadata"]["chunk_index"], 0)


class TestRetrievalPersistence(unittest.TestCase):
    def test_persistent_index_and_search(self):
        temp_root = tempfile.mkdtemp()
        try:
            data_dir = os.path.join(temp_root, "data")
            persist_dir = os.path.join(temp_root, "chroma")
            os.makedirs(data_dir, exist_ok=True)

            doc_path = os.path.join(data_dir, "sample.txt")
            with open(doc_path, "w", encoding="utf-8") as f:
                f.write("Photosynthesis needs light.\n\nCarbon dioxide is fixed in the Calvin cycle.")

            client1 = get_client(persist_dir)
            collection1 = get_collection(client1, "test_documents")
            indexed_chunks = ensure_indexed(collection1, data_dir)
            self.assertGreater(indexed_chunks, 0)
            count1 = collection1.count()
            self.assertGreater(count1, 0)
            del collection1
            del client1

            client2 = get_client(persist_dir)
            collection2 = get_collection(client2, "test_documents")
            count2 = collection2.count()
            self.assertEqual(count1, count2)

            result = search(collection2, "What does photosynthesis need?", n_results=1)
            self.assertEqual(len(result["documents"][0]), 1)
            self.assertTrue(result["documents"][0][0])
            del collection2
            del client2
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)


class TestGeneration(unittest.TestCase):
    def test_build_prompt_contains_context(self):
        contexts = [
            {
                "text": "Plants absorb light.",
                "metadata": {"source": "sample.txt", "chunk_index": 0},
                "distance": 0.1,
            }
        ]
        messages = build_prompt("How does it work?", contexts)
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("Plants absorb light.", messages[1]["content"])
        self.assertIn("How does it work?", messages[1]["content"])

    def test_generate_answer_requires_api_key(self):
        contexts = [
            {
                "text": "Context",
                "metadata": {"source": "sample.txt", "chunk_index": 0},
                "distance": 0.1,
            }
        ]
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError):
                generate_answer("Q?", contexts)

    def test_generate_answer_handles_http_error(self):
        contexts = [
            {
                "text": "Context",
                "metadata": {"source": "sample.txt", "chunk_index": 0},
                "distance": 0.1,
            }
        ]

        class FakeResponse:
            status_code = 500
            text = "internal error"

            @staticmethod
            def json():
                return {}

        class FakeClient:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                return False

            def post(self, *args, **kwargs):
                return FakeResponse()

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=True):
            with patch("generate.httpx.Client", FakeClient):
                with self.assertRaises(RuntimeError):
                    generate_answer("Q?", contexts)


if __name__ == "__main__":
    unittest.main()
