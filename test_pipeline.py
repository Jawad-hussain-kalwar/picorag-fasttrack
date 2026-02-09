import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

from src.chunking import naive_chunk
from src.generate import build_prompt, generate_answer
from src.retrieve import ensure_indexed, get_client, get_collection, search


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
            with patch("src.generate.httpx.Client", FakeClient):
                with self.assertRaises(RuntimeError):
                    generate_answer("Q?", contexts)


class TestGate(unittest.TestCase):
    def test_cosine_abstain_low_confidence(self):
        from src.gate import should_abstain
        # distance=1.8 → confidence = 1 - 1.8/2 = 0.1 → below 0.5 threshold
        abstain, conf = should_abstain([1.8, 1.9], threshold=0.5, method="cosine")
        self.assertTrue(abstain)
        self.assertAlmostEqual(conf, 0.1, places=5)

    def test_cosine_no_abstain_high_confidence(self):
        from src.gate import should_abstain
        # distance=0.2 → confidence = 1 - 0.2/2 = 0.9 → above 0.5 threshold
        abstain, conf = should_abstain([0.2, 0.5], threshold=0.5, method="cosine")
        self.assertFalse(abstain)
        self.assertAlmostEqual(conf, 0.9, places=5)

    def test_empty_distances_abstains(self):
        from src.gate import should_abstain
        abstain, conf = should_abstain([], threshold=0.5)
        self.assertTrue(abstain)
        self.assertEqual(conf, 0.0)

    def test_higher_better_method(self):
        from src.gate import should_abstain
        abstain, conf = should_abstain([0.8], threshold=0.5, method="higher_better")
        self.assertFalse(abstain)
        self.assertAlmostEqual(conf, 0.8, places=5)


class TestE3Metrics(unittest.TestCase):
    def test_selective_accuracy(self):
        from src.metrics import selective_accuracy
        preds = ["paris", "wrong", "berlin"]
        golds = [["paris"], ["london"], ["berlin"]]
        mask = [True, False, True]  # skip "wrong"
        self.assertAlmostEqual(selective_accuracy(preds, golds, mask), 1.0)

    def test_selective_accuracy_all_abstained(self):
        from src.metrics import selective_accuracy
        self.assertEqual(selective_accuracy(["a"], [["b"]], [False]), 0.0)

    def test_coverage(self):
        from src.metrics import coverage
        self.assertAlmostEqual(coverage([True, True, False, True]), 0.75)
        self.assertEqual(coverage([]), 0.0)

    def test_auprc_perfect(self):
        from src.metrics import auprc
        # Perfect ranking: all positives at top
        y_true = [1, 1, 0, 0]
        y_scores = [0.9, 0.8, 0.3, 0.1]
        result = auprc(y_true, y_scores)
        self.assertAlmostEqual(result, 1.0, places=2)

    def test_auprc_empty(self):
        from src.metrics import auprc
        self.assertEqual(auprc([], []), 0.0)
        self.assertEqual(auprc([0, 0], [0.5, 0.3]), 0.0)  # no positives

    def test_ece_perfect_calibration(self):
        from src.metrics import ece
        # Perfectly calibrated: conf matches accuracy
        y_true = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        y_conf = [0.95, 0.95, 0.95, 0.95, 0.95, 0.05, 0.05, 0.05, 0.05, 0.05]
        result = ece(y_true, y_conf)
        self.assertLess(result, 0.1)

    def test_ece_empty(self):
        from src.metrics import ece
        self.assertEqual(ece([], []), 0.0)


if __name__ == "__main__":
    unittest.main()

