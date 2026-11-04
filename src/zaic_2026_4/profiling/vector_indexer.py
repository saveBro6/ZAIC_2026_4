"""Module 4: Offline FAISS semantic vector indexer for schema and rule matching.

Embeds Maximal Profiles, dynamic rulebook.md rules, and README business context
using BAAI/bge-m3 via FlagEmbedding (BGEM3FlagModel) with FP16 acceleration and
stores unit-normalized dense vectors in faiss.IndexFlatIP for fast Cosine Similarity.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import faiss
import numpy as np

DEFAULT_MODEL_PATH = "/app/models/bge-m3"
DEFAULT_MAX_LENGTH = 4096
DEFAULT_BATCH_SIZE = 32
DEFAULT_TOP_K = 10
DEFAULT_EMBEDDING_DIM = 1024


@runtime_checkable
class EmbedderProtocol(Protocol):
    """Protocol defining interface for dense embedding models."""

    def encode(
        self,
        sentences: list[str] | str,
        batch_size: int | None = None,
        max_length: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any] | np.ndarray:
        """Encode text inputs into dense embeddings."""
        ...


class MockEmbedder:
    """Deterministic, offline pseudo-embedder for tests and low-resource environments.

    Generates reproducible unit L2-normalized float32 vectors based on string token hashing.
    Requires zero GPU memory and executes instantaneously without external network requests.
    """

    def __init__(self, dimension: int = DEFAULT_EMBEDDING_DIM) -> None:
        """Initialize MockEmbedder with a target embedding dimension."""
        self.dimension = dimension

    def get_sentence_embedding_dimension(self) -> int:
        """Return sentence embedding dimension."""
        return self.dimension

    def encode(
        self,
        sentences: list[str] | str,
        batch_size: int | None = None,
        max_length: int | None = None,
        **kwargs: Any,
    ) -> dict[str, np.ndarray]:
        """Generate deterministic unit-normalized float32 pseudo-vectors."""
        if isinstance(sentences, str):
            inputs = [sentences]
        else:
            inputs = list(sentences)

        if not inputs:
            vecs = np.zeros((0, self.dimension), dtype=np.float32)
            return {"dense_vecs": vecs}

        vectors = []
        for text in inputs:
            # Deterministic pseudo-random seed from MD5 hash of input text
            h = hashlib.md5(text.encode("utf-8")).digest()
            seed = int.from_bytes(h[:4], byteorder="little")
            rng = np.random.RandomState(seed)

            # Generate random vector and normalize to unit L2 norm
            vec = rng.randn(self.dimension).astype(np.float32)
            norm = np.linalg.norm(vec)
            if norm > 1e-8:
                vec = vec / norm
            else:
                vec[0] = 1.0
            vectors.append(vec)

        dense_vecs = np.vstack(vectors).astype(np.float32)
        return {"dense_vecs": dense_vecs}


class SemanticVectorIndexer:
    """Offline FAISS semantic vector indexer for Vietnamese SMB databases and rulebooks.

    Indexes column descriptions (Maximal Profiles), split-wide rulebook clauses,
    and business README context using FlagEmbedding (BGEM3FlagModel) and FAISS IndexFlatIP.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        embedder: Any | None = None,
        use_fp16: bool | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_length: int = DEFAULT_MAX_LENGTH,
        normalize_embeddings: bool = True,
    ) -> None:
        """Initialize SemanticVectorIndexer with embedder and FAISS IndexFlatIP.

        Args:
            model_path: Path to BGE-M3 model weights or HuggingFace model repo.
            embedder: Optional custom or pre-initialized embedder instance.
            use_fp16: Whether to enable FP16 inference for FlagEmbedding (default: True if CUDA).
            batch_size: Batch size for encoding texts.
            max_length: Maximum token length for BGEM3FlagModel (configured to 4096).
            normalize_embeddings: Whether to ensure L2 normalization for Cosine Similarity.
        """
        self.model_path = model_path
        self.batch_size = batch_size
        self.max_length = max_length
        self.normalize_embeddings = normalize_embeddings

        # 1. Initialize or assign embedder
        if embedder is not None:
            self.embedder = embedder
        elif model_path in ("mock", "mock-embedder"):
            self.embedder = MockEmbedder(dimension=DEFAULT_EMBEDDING_DIM)
        else:
            self.embedder = self._load_bgem3_model(model_path, use_fp16)

        # 2. Determine vector dimension
        self.dimension = self._detect_dimension()

        # 3. Initialize FAISS Inner Product index (equivalent to Cosine Similarity when normalized)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.doc_map: list[dict[str, Any]] = []

    def _load_bgem3_model(
        self, model_path: str, use_fp16: bool | None
    ) -> Any:
        """Safely import and initialize FlagEmbedding BGEM3FlagModel with GPU/CPU fallback."""
        try:
            import torch
            from FlagEmbedding import BGEM3FlagModel
        except ImportError as err:
            raise ImportError(
                "FlagEmbedding is required for SemanticVectorIndexer. "
                "Ensure flagembedding and torch are installed."
            ) from err

        fp16_enabled = use_fp16 if use_fp16 is not None else torch.cuda.is_available()

        # Check local folder existence or environment variable override
        effective_path = os.environ.get("EMBEDDING_MODEL_PATH", model_path)
        if not Path(effective_path).exists() and not Path(model_path).exists():
            # If path doesn't exist locally, try standard repo ID fallback
            effective_path = model_path if "/" in model_path and not model_path.startswith("/") else "BAAI/bge-m3"

        return BGEM3FlagModel(
            effective_path,
            use_fp16=fp16_enabled,
            normalize_embeddings=self.normalize_embeddings,
        )

    def _detect_dimension(self) -> int:
        """Detect sentence embedding dimension from embedder."""
        if hasattr(self.embedder, "get_sentence_embedding_dimension"):
            return int(self.embedder.get_sentence_embedding_dimension())
        if hasattr(self.embedder, "dimension"):
            return int(self.embedder.dimension)

        # Probe dimension with dummy test encoding
        try:
            probe = self._encode_texts(["probe"])
            return int(probe.shape[1])
        except Exception:
            return DEFAULT_EMBEDDING_DIM

    def _encode_texts(self, texts: list[str]) -> np.ndarray:
        """Encode a list of texts into L2-normalized float32 numpy array."""
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)

        out = self.embedder.encode(
            texts,
            batch_size=self.batch_size,
            max_length=self.max_length,
            return_dense=True,
        )

        if isinstance(out, dict) and "dense_vecs" in out:
            vecs = out["dense_vecs"]
        elif isinstance(out, np.ndarray):
            vecs = out
        else:
            vecs = np.asarray(out, dtype=np.float32)

        vecs = np.asarray(vecs, dtype=np.float32)

        # Ensure 2D shape (N, dim)
        if vecs.ndim == 1:
            vecs = np.expand_dims(vecs, axis=0)

        # Ensure unit L2 normalization
        if self.normalize_embeddings:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            vecs = vecs / norms

        return np.ascontiguousarray(vecs, dtype=np.float32)

    def _extract_columns_dict(
        self, processed_metadata: dict[str, Any]
    ) -> dict[str, Any]:
        """Flatten various metadata profile input structures into col_key -> meta."""
        if "metadata_profile" in processed_metadata and isinstance(
            processed_metadata["metadata_profile"], dict
        ):
            return processed_metadata["metadata_profile"]

        if "raw_profile" in processed_metadata and isinstance(
            processed_metadata["raw_profile"], dict
        ):
            raw = processed_metadata["raw_profile"]
            flat: dict[str, Any] = {}
            for tbl, t_info in raw.get("tables", {}).items():
                for col, c_info in t_info.get("columns", {}).items():
                    flat[f"{tbl}.{col}"] = c_info
            return flat

        if "tables" in processed_metadata and isinstance(
            processed_metadata["tables"], dict
        ):
            flat: dict[str, Any] = {}
            for tbl, t_info in processed_metadata["tables"].items():
                for col, c_info in t_info.get("columns", {}).items():
                    flat[f"{tbl}.{col}"] = c_info
            return flat

        return processed_metadata

    def build_index(
        self,
        processed_metadata: dict[str, Any],
        rulebook_text: str = "",
        readme_text: str = "",
    ) -> int:
        """Build FAISS vector index over schema profiles, rulebook rules, and README context.

        Args:
            processed_metadata: Dictionary mapping column keys (e.g. 'table.col') to metadata.
            rulebook_text: Text content of split rulebook.md.
            readme_text: Text content of business README.md.

        Returns:
            Total count of indexed items added to FAISS.
        """
        # Reset existing index and doc_map
        self.index = faiss.IndexFlatIP(self.dimension)
        self.doc_map = []

        texts: list[str] = []

        # 1. Embed database columns from Maximal Profile (or fallback)
        target_columns = self._extract_columns_dict(processed_metadata)
        for col_key, meta in target_columns.items():
            if not isinstance(meta, dict):
                continue

            # Prioritize maximal_profile, fallback to minimal_profile or basic stats
            desc = (
                meta.get("maximal_profile")
                or meta.get("minimal_profile")
                or (
                    f"Format: {meta.get('data_format', 'STRING')}. "
                    f"Distinct count: {meta.get('distinct_count', 0)}."
                )
            )

            text_to_embed = f"Column: {col_key}. Description: {desc}"
            texts.append(text_to_embed)

            # Split table and column identifiers if formatted as 'table.col'
            parts = col_key.split(".", 1)
            table_name = parts[0] if len(parts) == 2 else meta.get("table_name", "")
            col_name = parts[1] if len(parts) == 2 else meta.get("col_name", col_key)

            self.doc_map.append({
                "type": "column",
                "key": col_key,
                "content": text_to_embed,
                "table_name": table_name,
                "col_name": col_name,
                "data_format": meta.get("data_format", "STRING"),
            })

        # 2. Embed business rules from rulebook.md
        if rulebook_text and rulebook_text.strip():
            rules = [r.strip() for r in rulebook_text.split("\n\n") if len(r.strip()) > 10]
            for idx, rule in enumerate(rules):
                texts.append(rule)
                self.doc_map.append({
                    "type": "rulebook",
                    "key": f"rule_{idx}",
                    "content": rule,
                    "rule_index": idx,
                })

        # 3. Embed business context from README.md if provided
        if readme_text and readme_text.strip():
            sections = [
                s.strip()
                for s in readme_text.split("\n\n")
                if len(s.strip()) > 20 and not s.strip().startswith("```")
            ]
            for idx, sec in enumerate(sections):
                texts.append(sec)
                self.doc_map.append({
                    "type": "readme",
                    "key": f"readme_{idx}",
                    "content": sec,
                    "section_index": idx,
                })

        # Encode and add to FAISS index
        if texts:
            embeddings = self._encode_texts(texts)
            self.index.add(embeddings)

        return len(texts)

    def search_similar_schema(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        doc_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search top-k most semantically similar schema items or rules for a given query.

        Args:
            query: Natural language query string (e.g. 'doanh thu các mặt hàng bán chạy').
            top_k: Number of top results to return.
            doc_type: Optional filter for document type ('column', 'rulebook', 'readme').

        Returns:
            List of result dictionaries containing key, type, content, and cosine score.
        """
        clean_query = query.strip() if query else ""
        if self.index.ntotal == 0 or not clean_query:
            return []

        # Retrieve a wider candidate pool if filtering by doc_type
        search_k = min(
            top_k * 4 if doc_type else top_k,
            int(self.index.ntotal),
        )

        query_vec = self._encode_texts([clean_query])
        distances, indices = self.index.search(query_vec, search_k)

        # Correctly unwrap 2D arrays from FAISS search (indices[0], distances[0])
        row_indices = indices[0]
        row_distances = distances[0]

        results: list[dict[str, Any]] = []
        for i, idx in enumerate(row_indices):
            idx_int = int(idx)
            # Guard against FAISS -1 padding index for missing candidate slots
            if 0 <= idx_int < len(self.doc_map):
                doc = self.doc_map[idx_int]
                if doc_type is None or doc.get("type") == doc_type:
                    # Shallow copy to prevent in-place mutation of doc_map across queries
                    item = dict(doc)
                    item["score"] = float(row_distances[i])
                    results.append(item)
                    if len(results) >= top_k:
                        break

        return results

    def search_columns(
        self, query: str, top_k: int = 5
    ) -> list[dict[str, Any]]:
        """Convenience method to retrieve top semantically matching database columns."""
        return self.search_similar_schema(query, top_k=top_k, doc_type="column")

    def search_rules(
        self, query: str, top_k: int = 3
    ) -> list[dict[str, Any]]:
        """Convenience method to retrieve top semantically matching rulebook clauses."""
        return self.search_similar_schema(query, top_k=top_k, doc_type="rulebook")

    def save(self, target_dir: str | Path) -> None:
        """Persist FAISS index and doc_map metadata to disk for fast caching.

        Args:
            target_dir: Target directory path where index.faiss and doc_map.json will be saved.
        """
        out_dir = Path(target_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        index_file = out_dir / "index.faiss"
        doc_map_file = out_dir / "doc_map.json"
        meta_file = out_dir / "vector_meta.json"

        # Write FAISS binary index
        faiss.write_index(self.index, str(index_file))

        # Write doc_map JSON
        with open(doc_map_file, "w", encoding="utf-8") as f:
            json.dump(self.doc_map, f, ensure_ascii=False, indent=2)

        # Write metadata configuration
        metadata = {
            "dimension": self.dimension,
            "model_path": self.model_path,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "total_docs": len(self.doc_map),
        }
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(
        cls,
        target_dir: str | Path,
        model_path: str = DEFAULT_MODEL_PATH,
        embedder: Any | None = None,
    ) -> SemanticVectorIndexer:
        """Load a persisted FAISS index and doc_map metadata from disk.

        Args:
            target_dir: Directory containing index.faiss and doc_map.json.
            model_path: Model path override if embedder is not passed.
            embedder: Optional pre-initialized embedder.

        Returns:
            Restored SemanticVectorIndexer instance ready for querying.
        """
        in_dir = Path(target_dir)
        index_file = in_dir / "index.faiss"
        doc_map_file = in_dir / "doc_map.json"
        meta_file = in_dir / "vector_meta.json"

        if not index_file.is_file():
            raise FileNotFoundError(f"FAISS index file not found: {index_file}")
        if not doc_map_file.is_file():
            raise FileNotFoundError(f"Doc map file not found: {doc_map_file}")

        max_length = DEFAULT_MAX_LENGTH
        if meta_file.is_file():
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
                max_length = meta.get("max_length", DEFAULT_MAX_LENGTH)

        indexer = cls(
            model_path=model_path,
            embedder=embedder,
            max_length=max_length,
        )

        # Read FAISS binary index
        indexer.index = faiss.read_index(str(index_file))

        # Read doc_map JSON
        with open(doc_map_file, "r", encoding="utf-8") as f:
            indexer.doc_map = json.load(f)

        indexer.dimension = indexer.index.d
        return indexer
