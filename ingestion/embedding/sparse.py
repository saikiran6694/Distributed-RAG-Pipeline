"""
ingestion/embedding/sparse.py

BM25 sparse vector encoder.

Converts text into a sparse vector: {token_id: bm25_weight}.
This is the "keyword" half of hybrid search — catches exact matches
that dense semantic search misses (product codes, names, rare terms).

Why BM25 over TF-IDF:
  BM25 adds document length normalization and term frequency saturation.
  A word appearing 100x in a doc doesn't score 100x more than one
  appearing once — BM25 caps the benefit of repetition via k1 parameter.

Qdrant sparse vector format:
  {indices: [token_id, ...], values: [weight, ...]}
  Both lists must be same length, sorted by index ascending.
  Only non-zero weights are stored (hence "sparse").
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import tiktoken

logger = logging.getLogger(__name__)

# Reuse cl100k tokenizer — consistent token IDs with the dense embedding
_TOKENIZER = tiktoken.get_encoding("cl100k_base")

# BM25 hyperparameters
# k1 (1.5): term frequency saturation. Higher = more weight on repeated terms.
#           Range 1.2–2.0. 1.5 is a well-tested default for retrieval.
# b  (0.75): document length normalization. 1.0 = full normalization,
#            0.0 = no normalization. 0.75 is the BM25 standard.
_K1 = 1.5
_B  = 0.75


@dataclass
class SparseVector:
    """
    A sparse vector in Qdrant format.
    indices and values are parallel lists, sorted by index ascending.
    """
    indices: list[int]
    values:  list[float]

    def to_qdrant_dict(self) -> dict:
        return {"indices": self.indices, "values": self.values}

    def __len__(self) -> int:
        return len(self.indices)


class BM25Encoder:
    """
    Corpus-aware BM25 encoder.

    Requires fitting on a document corpus to compute IDF weights.
    For production, fit once on the full corpus and serialize the IDF table.
    For ingestion-time encoding, we use a simplified version that
    estimates IDF from corpus statistics accumulated during ingestion.

    Two modes:
      1. fit(corpus) → encode(text)   — full BM25 with corpus IDF
      2. encode_query(text)           — query-time encoding without corpus
         (uses log(1 + 1/tf) as a proxy for IDF — works well in practice)
    """

    def __init__(self):
        self._idf:        dict[int, float] = {}
        self._avgdl:      float            = 0.0
        self._doc_count:  int              = 0
        self._fitted:     bool             = False

    # ─────────────────────────────────────────────────────────
    #  Fitting (corpus statistics)
    # ─────────────────────────────────────────────────────────

    def fit(self, texts: list[str]) -> BM25Encoder:
        """
        Compute IDF weights from a corpus.
        Call this once before encoding documents.

        IDF(t) = log(1 + (N - df(t) + 0.5) / (df(t) + 0.5))
          where N = total documents, df(t) = documents containing term t.
        """
        if not texts:
            return self

        doc_freq: Counter[int] = Counter()
        total_len = 0

        for text in texts:
            tokens = self._tokenize(text)
            total_len += len(tokens)
            # Count each token once per document (document frequency)
            for token_id in set(tokens):
                doc_freq[token_id] += 1

        n = len(texts)
        self._avgdl = total_len / n
        self._doc_count = n

        # Compute IDF for every term in corpus
        for token_id, df in doc_freq.items():
            self._idf[token_id] = math.log(1 + (n - df + 0.5) / (df + 0.5))

        self._fitted = True
        logger.info(
            "BM25Encoder fitted: %d docs, %d unique tokens, avgdl=%.1f",
            n, len(self._idf), self._avgdl,
        )
        return self

    # ─────────────────────────────────────────────────────────
    #  Encoding
    # ─────────────────────────────────────────────────────────

    def encode_document(self, text: str) -> SparseVector:
        """
        Encode a document into a BM25 sparse vector.
        Requires fit() to have been called first.
        Falls back to TF-only encoding if not fitted.
        """
        tokens = self._tokenize(text)
        if not tokens:
            return SparseVector(indices=[], values=[])

        tf: Counter[int] = Counter(tokens)
        dl = len(tokens)
        avgdl = self._avgdl if self._fitted else dl

        scores: dict[int, float] = {}
        for token_id, freq in tf.items():
            idf = self._idf.get(token_id, self._default_idf(freq, dl))
            # BM25 term score formula
            tf_norm = (freq * (_K1 + 1)) / (freq + _K1 * (1 - _B + _B * dl / max(avgdl, 1)))
            scores[token_id] = idf * tf_norm

        return self._to_sparse_vector(scores)

    def encode_query(self, text: str) -> SparseVector:
        """
        Encode a search query into a sparse vector.
        Simpler than document encoding — queries are short, no length norm needed.
        Uses IDF from fitted corpus if available, otherwise falls back to uniform weights.
        """
        tokens = self._tokenize(text)
        if not tokens:
            return SparseVector(indices=[], values=[])

        tf: Counter[int] = Counter(tokens)
        scores: dict[int, float] = {}

        for token_id, _freq in tf.items():
            if self._fitted and token_id in self._idf:
                scores[token_id] = self._idf[token_id]
            else:
                # Unknown token — assign small uniform weight
                scores[token_id] = 1.0

        return self._to_sparse_vector(scores)

    # ─────────────────────────────────────────────────────────
    #  Serialization (persist fitted IDF table)
    # ─────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Persist IDF table to disk for reuse across restarts."""
        import json
        data = {
            "idf":       {str(k): v for k, v in self._idf.items()},
            "avgdl":     self._avgdl,
            "doc_count": self._doc_count,
        }
        Path(path).write_text(json.dumps(data))
        logger.info("BM25Encoder saved to %s (%d tokens)", path, len(self._idf))

    @classmethod
    def load(cls, path: str) -> BM25Encoder:
        """Load a previously fitted encoder from disk."""
        import json
        data = json.loads(Path(path).read_text())
        enc = cls()
        enc._idf        = {int(k): v for k, v in data["idf"].items()}
        enc._avgdl      = data["avgdl"]
        enc._doc_count  = data["doc_count"]
        enc._fitted     = True
        logger.info("BM25Encoder loaded from %s (%d tokens)", path, len(enc._idf))
        return enc

    # ─────────────────────────────────────────────────────────
    #  Internal helpers
    # ─────────────────────────────────────────────────────────

    def _tokenize(self, text: str) -> list[int]:
        """
        Tokenize text to cl100k token IDs.
        Lowercases first so "Python" and "python" share the same token.
        """
        return _TOKENIZER.encode(text.lower())

    def _default_idf(self, tf: int, dl: int) -> float:
        """Fallback IDF for tokens not seen during fitting."""
        return math.log(1 + 1.0 / max(tf, 1))

    def _to_sparse_vector(self, scores: dict[int, float]) -> SparseVector:
        """
        Convert score dict to sorted parallel lists.
        Drops zero-weight tokens (keep sparse).
        Qdrant requires indices sorted in ascending order.
        """
        filtered = {k: v for k, v in scores.items() if v > 0}
        sorted_items = sorted(filtered.items())   # sort by token_id ascending
        if not sorted_items:
            return SparseVector(indices=[], values=[])
        indices, values = zip(*sorted_items)
        return SparseVector(indices=list(indices), values=list(values))


# Module-level singleton — shared across all workers in a process
_encoder: BM25Encoder | None = None


def get_encoder(idf_path: str | None = None) -> BM25Encoder:
    """
    Return the module-level BM25Encoder singleton.
    If idf_path is provided and exists, loads fitted IDF from disk.
    """
    global _encoder
    if _encoder is None:
        _encoder = BM25Encoder()
        if idf_path and Path(idf_path).exists():
            _encoder = BM25Encoder.load(idf_path)
    return _encoder