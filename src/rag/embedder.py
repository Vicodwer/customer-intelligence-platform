"""Local text embeddings with ONNX Runtime (no PyTorch).

Default model: BAAI/bge-small-en-v1.5, exported to ONNX by the Xenova project,
full precision (fp32). bge models use the [CLS] token as the sentence vector and
expect search queries (not documents) to start with a fixed instruction.

The class is split in two so the maths can be tested without the real model:

    OnnxEmbedder(session, tokenizer)   pooling, normalising, batching
    OnnxEmbedder.from_pretrained()     downloads and wires up the real model
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import numpy as np


MODEL_REPO = "Xenova/bge-small-en-v1.5"
MODEL_FILE = "onnx/model.onnx"  # full precision. The quantized file is model_quantized.onnx.
TOKENIZER_FILE = "tokenizer.json"

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
MAX_LENGTH = 512  # the model's limit; longer text is cut off
BATCH_SIZE = 32


class OnnxEmbedder:
    def __init__(
        self,
        session: Any,
        tokenizer: Any,
        *,
        query_prefix: str = QUERY_PREFIX,
        batch_size: int = BATCH_SIZE,
        model_id: str = MODEL_REPO,
        revision: str | None = None,
        max_length: int = MAX_LENGTH,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")

        self.session = session
        self.tokenizer = tokenizer
        self.query_prefix = query_prefix
        self.batch_size = batch_size
        self.model_id = model_id
        self.revision = revision
        self.max_length = max_length

        self._input_names = {item.name for item in session.get_inputs()}

    # ---------- core maths ----------

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        encodings = self.tokenizer.encode_batch(texts)

        feeds = {
            "input_ids": np.array([e.ids for e in encodings], dtype=np.int64),
            "attention_mask": np.array([e.attention_mask for e in encodings], dtype=np.int64),
            "token_type_ids": np.array([e.type_ids for e in encodings], dtype=np.int64),
        }
        # Different exports expect different inputs; send only what the model declares.
        feeds = {name: value for name, value in feeds.items() if name in self._input_names}

        hidden = self.session.run(None, feeds)[0]  # (batch, tokens, dim)

        cls_vectors = np.asarray(hidden)[:, 0, :]  # [CLS] pooling
        norms = np.linalg.norm(cls_vectors, axis=1, keepdims=True)

        # Unit length, so a dot product between two vectors is their cosine similarity.
        return (cls_vectors / np.maximum(norms, 1e-12)).astype(np.float32)

    # ---------- public API ----------

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        """Embed documents. Row i of the result belongs to texts[i]."""
        texts = [str(text) for text in texts]

        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        # Batching texts of similar length wastes less time on padding.
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))

        result: np.ndarray | None = None

        for start in range(0, len(order), self.batch_size):
            indices = order[start : start + self.batch_size]
            vectors = self._embed_batch([texts[i] for i in indices])

            if result is None:
                result = np.zeros((len(texts), vectors.shape[1]), dtype=np.float32)

            result[indices] = vectors

        return result

    def encode_query(self, query: str) -> np.ndarray:
        """Embed one search query (with the bge instruction prefix)."""
        return self._embed_batch([self.query_prefix + str(query)])[0]

    # ---------- real model ----------

    @classmethod
    def from_pretrained(cls, repo: str = MODEL_REPO, **kwargs: Any) -> "OnnxEmbedder":
        try:
            import onnxruntime
            from huggingface_hub import hf_hub_download
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Dense retrieval needs extra packages. "
                "Run: pip install onnxruntime tokenizers huggingface_hub"
            ) from exc

        max_length = kwargs.pop("max_length", MAX_LENGTH)

        model_path = hf_hub_download(repo_id=repo, filename=MODEL_FILE)
        tokenizer_path = hf_hub_download(repo_id=repo, filename=TOKENIZER_FILE)

        tokenizer = Tokenizer.from_file(tokenizer_path)
        tokenizer.enable_truncation(max_length=max_length)
        tokenizer.enable_padding()  # pad each batch to its longest text

        session = onnxruntime.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )

        return cls(
            session,
            tokenizer,
            model_id=repo,
            revision=_revision_from_cache_path(model_path),
            max_length=max_length,
            **kwargs,
        )


def _revision_from_cache_path(path: str | Path) -> str | None:
    """The Hugging Face cache stores files under .../snapshots/<commit hash>/..."""
    parts = Path(path).parts

    if "snapshots" in parts:
        position = parts.index("snapshots")

        if position + 1 < len(parts):
            return parts[position + 1]

    return None


@lru_cache(maxsize=1)
def get_embedder() -> OnnxEmbedder:
    """The shared real embedder, loaded once per process."""
    return OnnxEmbedder.from_pretrained()