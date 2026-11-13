"""Module 2: Database profile summarization using vLLM minimal profiles and deterministic composition.

Generates Minimal Profiles via LLM (Short Summaries for Schema Linking) and
Maximal Profiles via deterministic composition from raw statistics and join metadata.
"""

import os
import time
from typing import Any, Protocol, runtime_checkable

# Disable FlashInfer top-p/top-k sampler by default to prevent JIT compilation failures on NVCC < 12.2
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

from zaic_2026_4.common.trace_logger import (
    LLMCallRecord,
    extract_llm_engine_call_traces,
    make_llm_call_record,
    sanitize_sampling_params,
)
from zaic_2026_4.constants import (
    DEFAULT_LLM_MAX_LENGTH,
    DEFAULT_LOCAL_GPU_UTIL,
    DEFAULT_LOCAL_MODEL,
    DEFAULT_PROD_GPU_UTIL,
    DEFAULT_PROD_MODEL,
    DEFAULT_PROFILE_BATCH_SIZE,
    DEFAULT_PROFILE_MAX_TOKENS,
    DEFAULT_PROFILE_RETRY_ATTEMPTS,
    DEFAULT_QUANTIZATION,
    DEFAULT_REASONING_END_STR,
    DEFAULT_REASONING_START_STR,
    MINIMAL_PROFILE_PROMPT,
)
from zaic_2026_4.profiling.maximal_composer import (
    build_join_index,
    build_table_context,
    compose_maximal_profile,
)
from zaic_2026_4.profiling.raw_profiler import ColumnProfile, DatabaseProfile


@runtime_checkable
class LLMEngineProtocol(Protocol):
    """Protocol for offline inference engines supporting batched generation."""

    def generate(
        self,
        prompts: list[str] | str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        stop: list[str] | None = None,
        seed: int | None = None,
        thinking_token_budget: int | None = None,
        **kwargs: Any,
    ) -> list[str] | str:
        """Generate text completions for a single prompt or batch of prompts."""
        ...


class VLLMEngine:
    """Wrapper around vLLM offline engine with continuous batching support."""

    def __init__(
        self,
        model: str = DEFAULT_PROD_MODEL,
        gpu_memory_utilization: float = DEFAULT_PROD_GPU_UTIL,
        max_model_len: int = DEFAULT_LLM_MAX_LENGTH,
        tensor_parallel_size: int = 1,
        trust_remote_code: bool = True,
        dtype: str = "auto",
        quantization: str | None = DEFAULT_QUANTIZATION,
        reasoning_config: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize vLLM engine instance."""
        from vllm import LLM

        self.model_name = model
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.tensor_parallel_size = tensor_parallel_size
        self.quantization = quantization

        if reasoning_config is None:
            try:
                from vllm.config import ReasoningConfig

                reasoning_config = ReasoningConfig(
                    reasoning_start_str=DEFAULT_REASONING_START_STR,
                    reasoning_end_str=DEFAULT_REASONING_END_STR,
                )
            except (ImportError, AttributeError, Exception):  # noqa: BLE001
                reasoning_config = None

        self.reasoning_config = reasoning_config

        llm_kwargs: dict[str, Any] = {
            "model": model,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_model_len": max_model_len,
            "tensor_parallel_size": tensor_parallel_size,
            "trust_remote_code": trust_remote_code,
            "dtype": dtype,
            "quantization": quantization,
            **kwargs,
        }
        if reasoning_config is not None:
            llm_kwargs["reasoning_config"] = reasoning_config

        self.llm = LLM(**llm_kwargs)
        self.last_call_traces: list[dict[str, Any]] = []

    def generate(
        self,
        prompts: list[str] | str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        stop: list[str] | None = None,
        seed: int | None = None,
        thinking_token_budget: int | None = None,
        **kwargs: Any,
    ) -> list[str] | str:
        """Generate completions using vLLM continuous batching."""
        from vllm import SamplingParams

        is_single = isinstance(prompts, str)
        if is_single and not prompts:
            self.last_call_traces = []
            return ""
        prompt_list = [prompts] if is_single else prompts

        if not prompt_list:
            self.last_call_traces = []
            return []

        default_stop = ["<|im_end|>", "<|endoftext|>", "</output>", "\n</output>"]
        stop_tokens = list(set(default_stop + (stop or [])))

        sampling_kwargs: dict[str, Any] = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stop": stop_tokens,
            **kwargs,
        }
        if seed is not None:
            sampling_kwargs["seed"] = seed
        if thinking_token_budget is not None:
            sampling_kwargs["thinking_token_budget"] = thinking_token_budget

        sampling_params = SamplingParams(**sampling_kwargs)
        clean_sampling = sanitize_sampling_params(sampling_kwargs)

        t0 = time.perf_counter()
        outputs = self.llm.generate(prompt_list, sampling_params)
        batch_latency_ms = (time.perf_counter() - t0) * 1000.0
        per_item_latency_ms = batch_latency_ms / max(1, len(outputs))

        results: list[str] = []
        call_traces: list[dict[str, Any]] = []
        for idx, output in enumerate(outputs):
            comp = output.outputs[0]
            text = comp.text.strip()
            results.append(text)
            prompt_token_ids = getattr(output, "prompt_token_ids", None)
            comp_token_ids = getattr(comp, "token_ids", None)
            call_traces.append({
                "prompt": prompt_list[idx] if idx < len(prompt_list) else "",
                "raw_output": text,
                "prompt_tokens": len(prompt_token_ids) if prompt_token_ids is not None else None,
                "completion_tokens": len(comp_token_ids) if comp_token_ids is not None else None,
                "finish_reason": getattr(comp, "finish_reason", None),
                "stop_reason": getattr(comp, "stop_reason", None),
                "latency_ms": round(per_item_latency_ms, 3),
                "batch_latency_ms": round(batch_latency_ms, 3),
                "batch_size": len(outputs),
                "sampling_params": clean_sampling,
            })
        self.last_call_traces = call_traces

        return results[0] if is_single else results


class MockLLMEngine:
    """Heuristic offline mock engine generating realistic profiles without GPU or network access."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize MockLLMEngine."""
        self.kwargs = kwargs
        self.last_call_traces: list[dict[str, Any]] = []

    def generate(
        self,
        prompts: list[str] | str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        stop: list[str] | None = None,
        seed: int | None = None,
        thinking_token_budget: int | None = None,
        **kwargs: Any,
    ) -> list[str] | str:
        """Generate mock Vietnamese metadata summaries from prompt hints."""
        is_single = isinstance(prompts, str)
        if is_single and not prompts:
            self.last_call_traces = []
            return ""
        prompt_list = [prompts] if is_single else prompts

        if not prompt_list:
            self.last_call_traces = []
            return []

        t0 = time.perf_counter()
        sampling_kwargs: dict[str, Any] = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stop": stop,
            "seed": seed,
            "thinking_token_budget": thinking_token_budget,
            **kwargs,
        }
        clean_sampling = sanitize_sampling_params(sampling_kwargs)

        results: list[str] = []
        for prompt in prompt_list:
            if not prompt:
                results.append("")
                continue

            # Support SQL generation prompts for Mock testing
            if "<target_column>" not in prompt and (
                "SQL engineer" in prompt
                or "<database_schema>" in prompt
                or "Text-to-SQL" in prompt
                or "<temporal_anchor>" in prompt
                or kwargs.get("structured_outputs") is not None
            ):
                import json
                import re

                if "Fresno" in prompt:
                    sql = (
                        "SELECT T2.Zip FROM frpm AS T1 "
                        "INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode "
                        "WHERE T1.`Charter School (Y/N)` = 1 AND T2.District = 'Fresno County Office of Education'"
                        if "reprompt_correction_guidance" in prompt
                        else "SELECT T2.Zip FROM frpm AS T1 "
                        "INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode "
                        "WHERE T1.`Charter School (Y/N)` = 1 AND T1.`County Name` = 'Fresno County Office of Education'"
                    )
                else:
                    tables = re.findall(r'<table name="([^"]+)">', prompt)
                    cols = re.findall(r'<column name="([^"]+)"', prompt)
                    if tables and cols:
                        main_tbl = tables[0]
                        col_refs = [f'T1."{c}"' for c in cols[:3]]
                        sql = f'SELECT {", ".join(col_refs)} FROM "{main_tbl}" AS T1'
                        if len(cols) > 3:
                            sql += f' WHERE T1."{cols[3]}" IS NOT NULL'
                    else:
                        sql = "SELECT 1"

                # If structured outputs is requested or prompt expects JSON {"sql": ..., "type": ...}
                if kwargs.get("structured_outputs") is not None or '{"sql":' in prompt or "Text-to-SQL" in prompt:
                    out_dict = {
                        "reasoning": "<think>Lập kế hoạch truy vấn SQL</think>",
                        "sql": sql,
                        "type": "number",
                    }
                    results.append(json.dumps(out_dict, ensure_ascii=False))
                else:
                    results.append(sql)
                continue

            # Scope extraction to the target column section (after <few_shot_examples>)
            target_text = (
                prompt.split("</few_shot_examples>")[-1]
                if "</few_shot_examples>" in prompt
                else prompt
            )

            # Extract table and column name from target task
            table_name = "Table"
            col_name = "Column"
            if 'Table: "' in target_text:
                try:
                    table_name = target_text.split('Table: "')[1].split('"')[0]
                except IndexError:
                    pass
            elif 'Bảng: "' in target_text:
                try:
                    table_name = target_text.split('Bảng: "')[1].split('"')[0]
                except IndexError:
                    pass

            if 'Column: "' in target_text:
                try:
                    col_name = target_text.split('Column: "')[1].split('"')[0]
                except IndexError:
                    pass
            elif 'Cột: "' in target_text:
                try:
                    col_name = target_text.split('Cột: "')[1].split('"')[0]
                except IndexError:
                    pass

            # Extract data format
            format_str = "STRING"
            if "Định dạng: " in target_text:
                format_str = target_text.split("Định dạng: ")[1].split("\n")[0]
            elif "Format & Hints: " in target_text:
                format_str = (
                    target_text.split("Format & Hints: ")[1]
                    .split("\n")[0]
                    .split(" | ")[0]
                )
            elif "Format & Pattern: " in target_text:
                format_str = target_text.split("Format & Pattern: ")[1].split("\n")[0]

            col_lower = col_name.lower()
            if any(k in col_lower for k in ("ngay", "ngày", "date", "time")):
                summary = f'Cột "{col_name}" lưu trữ ngày tháng giao dịch trong bảng {table_name}; định dạng thời gian cần chuẩn hóa ISO YYYY-MM-DD khi truy vấn.'
            elif any(k in col_lower for k in ("tien", "tiền", "gia", "giá", "cost", "price", "amount")):
                summary = f'Cột "{col_name}" chứa giá trị tiền tệ trong bảng {table_name}; định dạng số tiền VNĐ cần làm sạch dấu phân cách trước khi tính toán tổng/bình quân.'
            elif any(k in col_lower for k in ("ma", "mã", "id", "code", "sku")):
                summary = f'Cột "{col_name}" là mã định danh nghiệp vụ trong bảng {table_name}; định dạng chuỗi mã số/ký tự chữ và số.'
            elif any(k in col_lower for k in ("trang_thai", "trạng thái", "status", "ket_qua", "kết quả", "tien_do")):
                summary = f'Cột "{col_name}" theo dõi trạng thái xử lý giao dịch trong bảng {table_name}; cờ phân loại cần lọc theo quy ước nghiệp vụ.'
            else:
                summary = f'Cột "{col_name}" lưu trữ thuộc tính thông tin trong bảng {table_name}; kiểu dữ liệu {format_str}.'

            results.append(summary)

        batch_latency_ms = (time.perf_counter() - t0) * 1000.0
        per_item_latency_ms = batch_latency_ms / max(1, len(results))
        self.last_call_traces = [
            {
                "prompt": prompt_list[idx],
                "raw_output": results[idx],
                "prompt_tokens": None,
                "completion_tokens": None,
                "finish_reason": "stop",
                "stop_reason": None,
                "latency_ms": round(per_item_latency_ms, 3),
                "batch_latency_ms": round(batch_latency_ms, 3),
                "batch_size": len(results),
                "sampling_params": clean_sampling,
            }
            for idx in range(len(results))
        ]

        return results[0] if is_single else results


def create_engine(
    local: bool = False,
    model: str | None = None,
    gpu_memory_utilization: float | None = None,
    mock: bool = False,
    quantize: str | None = None,
    quantization: str | None = None,
    **kwargs: Any,
) -> LLMEngineProtocol:
    """Factory creating the appropriate LLM engine based on target environment.

    - Default (competition container with --gpus all):
        Model: Qwen/Qwen2.5-7B-Instruct, gpu_memory_utilization: 0.70
    - Local mode (--local):
        Model: Qwen/Qwen2.5-1.5B-Instruct, gpu_memory_utilization: 0.60
    - Mock mode (mock=True):
        Deterministic rule-based MockLLMEngine (instant execution without GPU/network).
    - Quantization (--quantize):
        Optional quantization format (e.g. 'fp8', 'awq', 'gptq'). Default: None.
    """
    if mock:
        return MockLLMEngine(**kwargs)

    resolved_model = model or (DEFAULT_LOCAL_MODEL if local else DEFAULT_PROD_MODEL)
    resolved_gpu_util = (
        gpu_memory_utilization
        if gpu_memory_utilization is not None
        else (DEFAULT_LOCAL_GPU_UTIL if local else DEFAULT_PROD_GPU_UTIL)
    )
    resolved_quantization = quantize if quantize is not None else (quantization or DEFAULT_QUANTIZATION)

    return VLLMEngine(
        model=resolved_model,
        gpu_memory_utilization=resolved_gpu_util,
        quantization=resolved_quantization,
        **kwargs,
    )


class ProfileLLMSummarizer:
    """Generates Minimal and Maximal metadata profiles using batched LLM inference."""

    def __init__(
        self,
        llm_engine: LLMEngineProtocol | Any,
        batch_size: int = DEFAULT_PROFILE_BATCH_SIZE,
    ) -> None:
        """Initialize ProfileLLMSummarizer with an LLM engine."""
        self.llm = llm_engine
        self.batch_size = batch_size
        self.last_profiling_trace: list[dict[str, Any]] = []
        self.last_llm_call_records: list[LLMCallRecord] = []

    def build_minimal_prompt(
        self,
        table_name: str,
        col_name: str,
        col_stats: ColumnProfile | dict[str, Any],
        sibling_cols: list[str],
        total_rows: int = 0,
        business_readme: str = "",
    ) -> str:
        """Construct the Minimal Profile XML prompt with full business_readme."""
        data_format = col_stats.get("data_format", "STRING") or "STRING"
        samples = (col_stats.get("sample_values") or [])[:10]

        return MINIMAL_PROFILE_PROMPT.format(
            table_name=table_name,
            col_name=col_name,
            sibling_cols=sibling_cols,
            data_format=data_format,
            samples=samples,
            business_readme=business_readme.strip(),
        )

    def generate_profiles(
        self,
        db_profile: DatabaseProfile | dict[str, Any],
        business_readme: str = "",
    ) -> dict[str, Any]:
        """Generate Minimal profiles via LLM and Maximal profiles via deterministic composition.

        Workflow:
        1. Batch inference for concise Minimal Profiles (semantic meaning & abbreviation decoding).
        2. Deterministic composition of Maximal Profiles using raw statistics, domain patterns,
           unified VIEW notices, and filtered join candidates.
        """
        readme_clean = business_readme.strip()

        # Phase 1: Collect minimal prompt metadata for all columns
        work_items: list[dict[str, Any]] = []

        for table_name, table_info in db_profile.get("tables", {}).items():
            total_rows = table_info.get("row_count", 0)
            columns = table_info.get("columns", {})
            all_cols = list(columns.keys())

            for col_name, col_stats in columns.items():
                sibling_cols = [c for c in all_cols if c != col_name]
                min_prompt = self.build_minimal_prompt(
                    table_name=table_name,
                    col_name=col_name,
                    col_stats=col_stats,
                    sibling_cols=sibling_cols,
                    total_rows=total_rows,
                    business_readme=readme_clean,
                )
                work_items.append({
                    "col_key": f"{table_name}.{col_name}",
                    "table_name": table_name,
                    "col_name": col_name,
                    "sibling_cols": sibling_cols,
                    "total_rows": total_rows,
                    "col_stats": col_stats,
                    "min_prompt": min_prompt,
                })

        if not work_items:
            self.last_profiling_trace = []
            self.last_llm_call_records = []
            return {}

        # Phase 2: Batch inference for minimal prompts.
        # The "\n\n" stop sequence used to abort every one-sentence answer at its first
        # blank line, and 160 tokens left no headroom, which is why a share of the profiles
        # came back empty. Both are corrected here, with one retry and a deterministic
        # fallback for anything the model still leaves blank.
        all_min_prompts = [item["min_prompt"] for item in work_items]
        min_summaries: list[str] = []
        chunk_engine_traces: list[dict[str, Any]] = []
        profile_sources: list[str] = ["llm"] * len(all_min_prompts)
        stop_sequences = ["</output>", "<|im_end|>", "<|endoftext|>"]

        for i in range(0, len(all_min_prompts), self.batch_size):
            chunk = all_min_prompts[i : i + self.batch_size]
            chunk_outs = self.llm.generate(
                chunk,
                max_tokens=DEFAULT_PROFILE_MAX_TOKENS,
                stop=stop_sequences,
            )
            if isinstance(chunk_outs, str):
                chunk_outs = [chunk_outs]
            min_summaries.extend(chunk_outs)
            traces = extract_llm_engine_call_traces(self.llm)
            for j in range(len(chunk)):
                chunk_engine_traces.append(traces[j] if j < len(traces) else {})

        for attempt in range(DEFAULT_PROFILE_RETRY_ATTEMPTS):
            empty_indices = [i for i, text in enumerate(min_summaries) if not str(text).strip()]
            if not empty_indices:
                break
            retry_prompts = [all_min_prompts[i] for i in empty_indices]
            retry_outs = self.llm.generate(
                retry_prompts,
                max_tokens=DEFAULT_PROFILE_MAX_TOKENS,
                stop=stop_sequences,
            )
            if isinstance(retry_outs, str):
                retry_outs = [retry_outs]
            retry_traces = extract_llm_engine_call_traces(self.llm)
            for slot, work_idx in enumerate(empty_indices):
                if slot >= len(retry_outs):
                    continue
                retry_text = str(retry_outs[slot]).strip()
                if not retry_text:
                    continue
                min_summaries[work_idx] = retry_text
                profile_sources[work_idx] = f"llm_retry_{attempt + 1}"
                if slot < len(retry_traces):
                    chunk_engine_traces[work_idx] = retry_traces[slot]

        for idx, text in enumerate(min_summaries):
            if str(text).strip():
                continue
            item = work_items[idx]
            data_format = str(item["col_stats"].get("data_format", "STRING") or "STRING")
            min_summaries[idx] = (
                f'Cột "{item["col_name"]}" thuộc bảng "{item["table_name"]}" '
                f"lưu trữ dữ liệu kiểu {data_format}."
            )
            profile_sources[idx] = "fallback"

        # Phase 3: Deterministic Maximal Profile Composition & Structural Indexing
        table_ctx_map = build_table_context(db_profile)
        join_index = build_join_index(db_profile)

        # Phase 4: Construct the final processed_metadata dictionary
        processed_metadata: dict[str, Any] = {}
        profiling_trace: list[dict[str, Any]] = []
        llm_call_records: list[LLMCallRecord] = []
        for idx, item in enumerate(work_items):
            col_key = item["col_key"]
            col_stats = item["col_stats"]
            raw_out = min_summaries[idx] if idx < len(min_summaries) else ""
            short_summary = raw_out.strip()
            eng_trace = chunk_engine_traces[idx] if idx < len(chunk_engine_traces) else {}
            profiling_trace.append({
                "col_key": col_key,
                "table_name": item["table_name"],
                "col_name": item["col_name"],
                "prompt": item["min_prompt"],
                "raw_output": raw_out,
                "minimal_profile": short_summary,
                "profile_source": profile_sources[idx] if idx < len(profile_sources) else "llm",
                "prompt_tokens": eng_trace.get("prompt_tokens"),
                "completion_tokens": eng_trace.get("completion_tokens"),
                "finish_reason": eng_trace.get("finish_reason"),
                "stop_reason": eng_trace.get("stop_reason"),
                "latency_ms": eng_trace.get("latency_ms"),
                "sampling_params": eng_trace.get("sampling_params") or {},
            })
            llm_call_records.append(
                make_llm_call_record(
                    stage="minimal_profile",
                    prompt=item["min_prompt"],
                    raw_output=raw_out,
                    engine_trace=eng_trace,
                    metadata={
                        "col_key": col_key,
                        "table_name": item["table_name"],
                        "col_name": item["col_name"],
                    },
                )
            )
            tbl_name = item["table_name"]
            c_name = item["col_name"]
            tbl_ctx = table_ctx_map.get(tbl_name)
            col_joins = join_index.get(col_key, [])

            maximal_profile = compose_maximal_profile(
                minimal_profile=short_summary,
                table_name=tbl_name,
                col_name=c_name,
                col_stats=col_stats,
                table_ctx=tbl_ctx,
                joins=col_joins,
            )

            distinct_cnt = col_stats.get("distinct_count", 0) or 0
            null_cnt = col_stats.get("null_count", 0) or 0
            total_r = item["total_rows"]
            is_unique = bool(distinct_cnt > 0 and distinct_cnt == (total_r - null_cnt))

            processed_metadata[col_key] = {
                "minimal_profile": short_summary,
                "maximal_profile": maximal_profile,
                "sample_values": col_stats.get("sample_values") or [],
                # Enriched attributes preserved for downstream vector and schema operations
                "table_name": tbl_name,
                "col_name": c_name,
                "data_format": col_stats.get("data_format", "STRING") or "STRING",
                "date_subtype": col_stats.get("date_subtype"),
                "currency_subtype": col_stats.get("currency_subtype"),
                "null_count": null_cnt,
                "distinct_count": distinct_cnt,
                "top_10_samples": col_stats.get("top_10_samples") or {},
                "llm_hint": col_stats.get("llm_hint"),
                "min_val": col_stats.get("min_val"),
                "max_val": col_stats.get("max_val"),
                "is_unique": is_unique,
                "join_candidates": [
                    {
                        "target": j.target_col,
                        "resemblance": j.resemblance,
                        "is_target_unique": j.is_target_unique,
                    }
                    for j in col_joins
                ],
            }

        self.last_profiling_trace = profiling_trace
        self.last_llm_call_records = llm_call_records
        return processed_metadata

    def summarize_calendar(
        self,
        calendar_raw_profile: DatabaseProfile | dict[str, Any],
        business_readme: str = "",
    ) -> dict[str, Any]:
        """Generate Minimal and Maximal profiles for split-level calendar.csv.

        Args:
            calendar_raw_profile: Raw DatabaseProfile of calendar.csv.
            business_readme: Optional domain description or rulebook text for calendar.

        Returns:
            Dictionary of processed column metadata.
        """
        return self.generate_profiles(
            calendar_raw_profile,
            business_readme=business_readme,
        )
