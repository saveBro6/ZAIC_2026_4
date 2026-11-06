"""Module 4: Offline FAISS semantic vector indexer for schema and rule matching.

Embeds Maximal Profiles, dynamic rulebook.md rules, and README business context
using BAAI/bge-m3 via vLLM (runner="pooling") with FP16 acceleration and
stores unit-normalized dense vectors in faiss.IndexFlatIP for fast Cosine Similarity.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import faiss
import numpy as np

DEFAULT_MODEL_PATH = "/app/models/bge-m3"
DEFAULT_MAX_LENGTH = 4096
DEFAULT_BATCH_SIZE = 32
DEFAULT_TOP_K = 10
DEFAULT_EMBEDDING_DIM = 1024
DEFAULT_MAX_CHUNK_CHARS = 1000
DEFAULT_MIN_CHUNK_CHARS = 150
DEFAULT_VLLM_GPU_MEMORY_UTILIZATION = 0.20
MIN_TABLE_ROWS_FOR_SPLIT = 3

# Module-level in-memory cache to ensure BGE-M3 is loaded only once per process lifecycle
_EMBEDDER_CACHE: dict[tuple[str, float, int, bool], Any] = {}


def clear_embedder_cache() -> None:
    """Clear all cached embedder instances (useful for test isolation)."""
    _EMBEDDER_CACHE.clear()


def get_embedder_cache_size() -> int:
    """Return the number of cached embedder instances."""
    return len(_EMBEDDER_CACHE)


def recursive_split_paragraph(
    text: str,
    max_chars: int = DEFAULT_MAX_CHUNK_CHARS,
    delimiters: list[str] | None = None,
) -> list[str]:
    """Recursively split text without cutting words in half using priority delimiters.

    Delimiters are attempted in decreasing priority: ["\n", ".", ",", " "].
    Guarantees no word is split across chunks and trailing sentence punctuation is preserved.
    """
    if delimiters is None:
        delimiters = ["\n", ".", ",", " "]

    clean_text = text.strip()
    if len(clean_text) <= max_chars:
        return [clean_text] if clean_text else []

    # Find highest priority delimiter present in clean_text
    current_delim = None
    remaining_delims: list[str] = []
    for idx, d in enumerate(delimiters):
        if d in clean_text:
            current_delim = d
            remaining_delims = delimiters[idx + 1 :]
            break

    if current_delim is None:
        # Fallback if no delimiter can split text
        return [clean_text[i : i + max_chars] for i in range(0, len(clean_text), max_chars)]

    raw_parts = clean_text.split(current_delim)
    merged_chunks: list[str] = []
    current_chunk = ""

    for part in raw_parts:
        part_clean = part.strip()
        if not part_clean:
            continue

        item_to_add = part_clean + (current_delim if current_delim in (".", ",") else "")
        if len(item_to_add) > max_chars:
            # Recursively split oversized fragment with lower priority delimiters
            sub_parts = recursive_split_paragraph(item_to_add, max_chars, remaining_delims)
            if current_chunk:
                merged_chunks.append(current_chunk.strip())
                current_chunk = ""
            merged_chunks.extend(sub_parts)
            continue

        joiner = " " if current_delim in (".", ",", " ") else "\n"
        candidate = f"{current_chunk}{joiner}{item_to_add}" if current_chunk else item_to_add
        if len(candidate) <= max_chars:
            current_chunk = candidate
        else:
            if current_chunk:
                merged_chunks.append(current_chunk.strip())
            current_chunk = item_to_add

    if current_chunk.strip():
        merged_chunks.append(current_chunk.strip())

    return merged_chunks


def parse_markdown_hierarchical(
    markdown_text: str,
    max_chars: int = DEFAULT_MAX_CHUNK_CHARS,
    min_chunk_chars: int = DEFAULT_MIN_CHUNK_CHARS,
    min_table_rows_for_split: int = MIN_TABLE_ROWS_FOR_SPLIT,
) -> list[str]:
    """Parse Markdown document into contextualized chunks using hierarchical heading tracking,
    table row-level decomposition, and buffer coalescing.

    Args:
        markdown_text: Raw Markdown content (e.g. from rulebook.md or README.md).
        max_chars: Maximum character length per paragraph chunk before recursive splitting (1000).
        min_chunk_chars: Minimum character length for coalescing short paragraphs (150).
        min_table_rows_for_split: Threshold of total table rows (header + data) to trigger row-level splitting (3).

    Returns:
        List of contextualized chunk strings.
    """
    if not markdown_text or not markdown_text.strip():
        return []

    lines = markdown_text.splitlines()
    n = len(lines)
    i = 0
    current_headings: list[tuple[int, str]] = []
    chunks: list[str] = []
    current_buffer = ""

    def _flush_buffer() -> None:
        nonlocal current_buffer
        if current_buffer.strip():
            chunks.append(current_buffer.strip())
            current_buffer = ""

    while i < n:
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        # 1. Heading detection
        h_match = re.match(r"^(#{1,6})\s+(.+)$", line)
        if h_match:
            level = len(h_match.group(1))
            title = h_match.group(2).strip()

            # Major section boundary (H1 or H2): flush any pending buffer
            if level <= 2:
                _flush_buffer()

            while current_headings and current_headings[-1][0] >= level:
                current_headings.pop()
            current_headings.append((level, title))
            i += 1
            continue

        # Exclude H1 (level == 1) document titles to prevent semantic dilution across all chunks
        relevant_headings = [t for lvl, t in current_headings if lvl > 1]
        heading_ctx = " > ".join(relevant_headings)

        # 2. Markdown Table Detection
        if (
            line.startswith("|")
            and line.endswith("|")
            and i + 1 < n
            and "---" in lines[i + 1]
        ):
            header_line = line
            headers = [c.strip() for c in header_line.strip("|").split("|")]
            sep_line = lines[i + 1]
            i += 2  # skip header and separator row

            data_lines: list[str] = []
            while i < n and lines[i].strip().startswith("|") and lines[i].strip().endswith("|"):
                data_lines.append(lines[i].strip())
                i += 1

            total_rows = 1 + len(data_lines)
            if total_rows > min_table_rows_for_split:
                # Flush pending buffer before processing large table
                _flush_buffer()
                for row_line in data_lines:
                    cells = [c.strip() for c in row_line.strip("|").split("|")]
                    if any(cells):
                        if len(cells) >= 2 and len(headers) >= 2:
                            body = f"{headers[0]}: {cells[0]} -> {headers[1]}: {cells[1]}"
                            if len(cells) > 2 and len(headers) > 2:
                                extra = "; ".join(
                                    f"{headers[j]}: {cells[j]}"
                                    for j in range(2, min(len(headers), len(cells)))
                                    if cells[j]
                                )
                                if extra:
                                    body += f"; {extra}"
                        else:
                            body = "; ".join(
                                f"{headers[j] if j < len(headers) else 'Cột'}: {cells[j]}"
                                for j in range(len(cells))
                                if cells[j]
                            )

                        chunk = f"[Mục: {heading_ctx}] {body}" if heading_ctx else body
                        chunks.append(chunk)
                continue
            else:
                # Small table (<= 3 rows): treat as single block
                table_block = header_line + "\n" + sep_line + "\n" + "\n".join(data_lines)
                para_to_process = table_block
        else:
            # 3. Regular paragraph / list block
            para_lines = [line]
            i += 1
            while (
                i < n
                and lines[i].strip()
                and not re.match(r"^(#{1,6})\s+(.+)$", lines[i].strip())
                and not (
                    lines[i].strip().startswith("|")
                    and "---" in (lines[i + 1 : i + 2] or [""])[0]
                )
            ):
                para_lines.append(lines[i].strip())
                i += 1

            para_to_process = "\n".join(para_lines).strip()

        # Filter out standalone code block markers
        if para_to_process.startswith("```") and para_to_process.endswith("```"):
            continue

        # Process paragraph with buffer coalescing and recursive splitting
        if len(para_to_process) > max_chars:
            _flush_buffer()
            sub_chunks = recursive_split_paragraph(para_to_process, max_chars=max_chars)
            for sc in sub_chunks:
                if len(sc) > 10:
                    chunk = f"[Mục: {heading_ctx}] {sc}" if heading_ctx else sc
                    chunks.append(chunk)
        else:
            item_with_ctx = (
                f"[Mục: {heading_ctx}] {para_to_process}"
                if heading_ctx
                else para_to_process
            )
            if current_buffer:
                current_buffer = f"{current_buffer}\n\n{item_with_ctx}"
            else:
                current_buffer = item_with_ctx

            if len(current_buffer) >= min_chunk_chars:
                chunks.append(current_buffer)
                current_buffer = ""

    # Flush any remaining buffer at end of document
    _flush_buffer()
    return chunks


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
    and business README context using BAAI/bge-m3 via vLLM (runner="pooling") and FAISS IndexFlatIP.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        embedder: Any | None = None,
        use_fp16: bool | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_length: int = DEFAULT_MAX_LENGTH,
        normalize_embeddings: bool = True,
        gpu_memory_utilization: float | None = None,
        enforce_eager: bool = True,
    ) -> None:
        """Initialize SemanticVectorIndexer with embedder and FAISS IndexFlatIP.

        Args:
            model_path: Path to BGE-M3 model weights or HuggingFace model repo.
            embedder: Optional custom or pre-initialized embedder instance.
            use_fp16: Whether to enable FP16 inference (default: True if CUDA).
            batch_size: Batch size for encoding texts.
            max_length: Maximum token length for BGE-M3 (configured to 4096).
            normalize_embeddings: Whether to ensure L2 normalization for Cosine Similarity.
            gpu_memory_utilization: GPU memory fraction for vLLM pooling runner (default: 0.20).
            enforce_eager: Whether to enforce eager mode for vLLM to skip compilation warmup.
        """
        self.model_path = model_path
        self.batch_size = batch_size
        self.max_length = max_length
        self.normalize_embeddings = normalize_embeddings
        self.gpu_memory_utilization = (
            gpu_memory_utilization
            if gpu_memory_utilization is not None
            else float(
                os.environ.get(
                    "VLLM_EMBEDDING_GPU_MEMORY_UTILIZATION",
                    DEFAULT_VLLM_GPU_MEMORY_UTILIZATION,
                )
            )
        )
        self.enforce_eager = enforce_eager

        # 1. Initialize or assign embedder with lazy-loading support
        self.use_fp16 = use_fp16
        self._embedder = embedder
        if embedder is not None:
            self.dimension = self._detect_dimension()
        elif model_path in ("mock", "mock-embedder"):
            self._embedder = MockEmbedder(dimension=DEFAULT_EMBEDDING_DIM)
            self.dimension = DEFAULT_EMBEDDING_DIM
        else:
            self.dimension = DEFAULT_EMBEDDING_DIM

        # 2. Initialize FAISS Inner Product index (equivalent to Cosine Similarity when normalized)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.doc_map: list[dict[str, Any]] = []

    @property
    def embedder(self) -> Any:
        """Lazily load the underlying embedding engine on first access."""
        if self._embedder is None:
            if self.model_path in ("mock", "mock-embedder"):
                self._embedder = MockEmbedder(dimension=self.dimension or DEFAULT_EMBEDDING_DIM)
            else:
                self._embedder = self._load_bgem3_model(self.model_path, self.use_fp16)
        return self._embedder

    @embedder.setter
    def embedder(self, val: Any) -> None:
        """Explicitly set or replace the embedder instance."""
        self._embedder = val
        if val is not None:
            self.dimension = self._detect_dimension()

    def _load_bgem3_model(
        self, model_path: str, use_fp16: bool | None
    ) -> Any:
        """Safely import and initialize vLLM pooling runner for BGE-M3."""
        # Check local folder existence or environment variable override
        effective_path = os.environ.get("EMBEDDING_MODEL_PATH", model_path)
        if not Path(effective_path).exists() and not Path(model_path).exists():
            # If path doesn't exist locally, try standard repo ID fallback
            effective_path = model_path if "/" in model_path and not model_path.startswith("/") else "BAAI/bge-m3"

        cache_key = (
            effective_path,
            float(self.gpu_memory_utilization),
            int(self.max_length),
            bool(self.enforce_eager),
        )
        if cache_key in _EMBEDDER_CACHE:
            return _EMBEDDER_CACHE[cache_key]

        try:
            from vllm import LLM
        except ImportError as err:
            raise ImportError(
                "vLLM is required for SemanticVectorIndexer. "
                "Ensure vllm is installed."
            ) from err

        instance = LLM(
            model=effective_path,
            runner="pooling",
            max_model_len=self.max_length,
            gpu_memory_utilization=self.gpu_memory_utilization,
            enforce_eager=self.enforce_eager,
            trust_remote_code=True,
        )
        _EMBEDDER_CACHE[cache_key] = instance
        return instance

    def _detect_dimension(self) -> int:
        """Detect sentence embedding dimension from embedder."""
        target = self._embedder if self._embedder is not None else self.embedder
        if hasattr(target, "get_sentence_embedding_dimension"):
            return int(target.get_sentence_embedding_dimension())
        if hasattr(target, "dimension"):
            return int(target.dimension)

        # Probe dimension with dummy test encoding
        try:
            probe = self._encode_texts(["probe"])
            return int(probe.shape[1])
        except (AttributeError, ValueError, TypeError, RuntimeError):
            return DEFAULT_EMBEDDING_DIM

    def _encode_texts(self, texts: list[str]) -> np.ndarray:
        """Encode a list of texts into L2-normalized float32 numpy array."""
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)

        if hasattr(self.embedder, "embed"):
            # vLLM native pooling API: returns list[PoolingRequestOutput] or list[EmbeddingRequestOutput]
            outputs = self.embedder.embed(texts)
            vectors: list[list[float] | np.ndarray] = []
            for item in outputs:
                if hasattr(item, "outputs") and hasattr(item.outputs, "embedding"):
                    vectors.append(item.outputs.embedding)
                elif hasattr(item, "outputs") and hasattr(item.outputs, "data"):
                    vectors.append(item.outputs.data)
                elif hasattr(item, "embedding"):
                    vectors.append(item.embedding)
                else:
                    raise ValueError(f"Unrecognized vLLM embedder output format: {type(item)}")
            vecs = np.asarray(vectors, dtype=np.float32)
        elif hasattr(self.embedder, "encode"):
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
        else:
            raise TypeError(
                f"Embedder of type {type(self.embedder)} does not support 'embed' or 'encode'."
            )

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

        # 2. Embed business rules from rulebook.md using hierarchical table-row & buffer chunking
        if rulebook_text and rulebook_text.strip():
            rules = parse_markdown_hierarchical(
                rulebook_text,
                max_chars=DEFAULT_MAX_CHUNK_CHARS,
                min_chunk_chars=DEFAULT_MIN_CHUNK_CHARS,
                min_table_rows_for_split=MIN_TABLE_ROWS_FOR_SPLIT,
            )
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
            sections = parse_markdown_hierarchical(
                readme_text,
                max_chars=DEFAULT_MAX_CHUNK_CHARS,
                min_chunk_chars=DEFAULT_MIN_CHUNK_CHARS,
                min_table_rows_for_split=MIN_TABLE_ROWS_FOR_SPLIT,
            )
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

        # If filtering by doc_type, search all candidate documents to ensure candidate pool contains target type
        search_k = int(self.index.ntotal) if doc_type else min(top_k, int(self.index.ntotal))

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

    def merge(self, other: SemanticVectorIndexer) -> int:
        """Merge another SemanticVectorIndexer's indexed documents and vectors into this index.

        Args:
            other: Another SemanticVectorIndexer instance to merge.

        Returns:
            Number of new items added.
        """
        if other.index.ntotal == 0 or not other.doc_map:
            return 0

        existing_keys = {d.get("key") for d in self.doc_map if d.get("key")}
        docs_to_add = [d for d in other.doc_map if d.get("key") not in existing_keys]
        if not docs_to_add:
            return 0

        texts = [d["content"] for d in docs_to_add]
        embeddings = self._encode_texts(texts)
        self.index.add(embeddings)
        self.doc_map.extend(docs_to_add)
        return len(docs_to_add)

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
