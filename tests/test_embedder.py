import numpy as np
import pytest

from src.rag.embedder import QUERY_PREFIX, OnnxEmbedder, _revision_from_cache_path


class _Input:
    def __init__(self, name):
        self.name = name


class _Encoding:
    def __init__(self, ids, width):
        self.ids = ids + [0] * (width - len(ids))
        self.attention_mask = [1] * len(ids) + [0] * (width - len(ids))
        self.type_ids = [0] * width


class FakeTokenizer:
    """Turns each character into a token id and pads to the longest text."""

    def __init__(self):
        self.seen: list[list[str]] = []

    def encode_batch(self, texts):
        self.seen.append(list(texts))
        width = max(len(text) for text in texts)

        return [_Encoding([ord(c) for c in text], width) for text in texts]


class FakeSession:
    """Fake model: the [CLS] vector is (sum of token ids, length, 1)."""

    def __init__(self, input_names=("input_ids", "attention_mask", "token_type_ids")):
        self._names = input_names
        self.fed: list[set[str]] = []

    def get_inputs(self):
        return [_Input(name) for name in self._names]

    def run(self, _outputs, feeds):
        self.fed.append(set(feeds))

        ids = feeds["input_ids"]
        mask = feeds["attention_mask"]

        hidden = np.zeros((ids.shape[0], ids.shape[1], 3), dtype=np.float32)
        hidden[:, 0, 0] = ids.sum(axis=1)
        hidden[:, 0, 1] = mask.sum(axis=1)
        hidden[:, 0, 2] = 1.0

        return [hidden]


def _embedder(**kwargs):
    return OnnxEmbedder(FakeSession(), FakeTokenizer(), **kwargs)


def test_vectors_have_unit_length():
    vectors = _embedder().encode_documents(["hello", "a much longer sentence"])

    assert np.linalg.norm(vectors, axis=1) == pytest.approx([1.0, 1.0])


def test_rows_stay_in_input_order_even_though_batches_are_sorted_by_length():
    texts = ["a very long document indeed", "hi", "medium text"]
    embedder = _embedder(batch_size=2)

    together = embedder.encode_documents(texts)
    one_by_one = np.vstack([embedder.encode_documents([t]) for t in texts])

    assert together == pytest.approx(one_by_one)


def test_batch_size_is_respected():
    embedder = _embedder(batch_size=2)

    embedder.encode_documents(["a", "bb", "ccc", "dddd", "eeeee"])

    assert [len(batch) for batch in embedder.tokenizer.seen] == [2, 2, 1]


def test_query_gets_the_instruction_prefix_but_documents_do_not():
    embedder = _embedder()

    embedder.encode_query("late fee")
    embedder.encode_documents(["late fee"])

    assert embedder.tokenizer.seen[0] == [QUERY_PREFIX + "late fee"]
    assert embedder.tokenizer.seen[1] == ["late fee"]


def test_only_inputs_the_model_declares_are_sent():
    session = FakeSession(input_names=("input_ids", "attention_mask"))
    OnnxEmbedder(session, FakeTokenizer()).encode_documents(["hello"])

    assert session.fed == [{"input_ids", "attention_mask"}]


def test_zero_vector_does_not_divide_by_zero():
    class ZeroSession(FakeSession):
        def run(self, _outputs, feeds):
            return [np.zeros((feeds["input_ids"].shape[0], 2, 3), dtype=np.float32)]

    vectors = OnnxEmbedder(ZeroSession(), FakeTokenizer()).encode_documents(["x"])

    assert np.isfinite(vectors).all()


def test_no_documents_gives_empty_result():
    assert _embedder().encode_documents([]).shape[0] == 0


def test_invalid_batch_size_is_rejected():
    with pytest.raises(ValueError):
        _embedder(batch_size=0)


def test_revision_is_read_from_the_hugging_face_cache_path():
    path = "/home/u/.cache/huggingface/hub/models--X--y/snapshots/abc123/onnx/model.onnx"

    assert _revision_from_cache_path(path) == "abc123"
    assert _revision_from_cache_path("/somewhere/else/model.onnx") is None