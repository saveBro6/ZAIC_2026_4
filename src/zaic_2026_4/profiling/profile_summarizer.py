"""Module 2: LLM-powered database profile summarization using vLLM and Qwen2.5.

Generates Minimal Profiles (Short Summaries for Schema Linking) and
Maximal Profiles (Long Summaries for Code Synthesis & FAISS Vector Indexing).
"""

import os
from typing import Any, Protocol, runtime_checkable

# Disable FlashInfer top-p/top-k sampler by default to prevent JIT compilation failures on NVCC < 12.2
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

from zaic_2026_4.constants import (
    DEFAULT_LLM_MAX_LENGTH,
    DEFAULT_LOCAL_GPU_UTIL,
    DEFAULT_LOCAL_MODEL,
    DEFAULT_PROD_GPU_UTIL,
    DEFAULT_PROD_MODEL,
    DEFAULT_QUANTIZATION,
    MAXIMAL_PROFILE_PROMPT,
    MINIMAL_PROFILE_PROMPT,
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
        **kwargs: Any,
    ) -> None:
        """Initialize vLLM engine instance."""
        from vllm import LLM

        self.model_name = model
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.tensor_parallel_size = tensor_parallel_size
        self.quantization = quantization

        self.llm = LLM(
            model=model,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=trust_remote_code,
            dtype=dtype,
            quantization=quantization,
            **kwargs,
        )

    def generate(
        self,
        prompts: list[str] | str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        stop: list[str] | None = None,
        seed: int | None = None,
        **kwargs: Any,
    ) -> list[str] | str:
        """Generate completions using vLLM continuous batching."""
        from vllm import SamplingParams

        is_single = isinstance(prompts, str)
        if is_single and not prompts:
            return ""
        prompt_list = [prompts] if is_single else prompts

        if not prompt_list:
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

        sampling_params = SamplingParams(**sampling_kwargs)

        outputs = self.llm.generate(prompt_list, sampling_params)
        results = [output.outputs[0].text.strip() for output in outputs]

        return results[0] if is_single else results


class MockLLMEngine:
    """Heuristic offline mock engine generating realistic profiles without GPU or network access."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize MockLLMEngine."""
        self.kwargs = kwargs

    def generate(
        self,
        prompts: list[str] | str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        stop: list[str] | None = None,
        seed: int | None = None,
        **kwargs: Any,
    ) -> list[str] | str:
        """Generate mock Vietnamese metadata summaries from prompt hints."""
        is_single = isinstance(prompts, str)
        if is_single and not prompts:
            return ""
        prompt_list = [prompts] if is_single else prompts

        if not prompt_list:
            return []

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
                    out_dict = {"sql": sql, "type": "number"}
                    results.append(json.dumps(out_dict, ensure_ascii=False))
                else:
                    results.append(sql)
                continue

            # Check whether it is a minimal or maximal prompt
            is_minimal = "SHORT SUMMARY" in prompt or "TÓM TẮT RÚT GỌN" in prompt

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
            if is_minimal:
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
            else:
                minimal_profile_in_prompt = ""
                if 'Tóm tắt rút gọn: "' in target_text:
                    try:
                        raw_line = target_text.split('Tóm tắt rút gọn: "')[1].split("\n")[0].strip()
                        raw_line = raw_line.removesuffix('"')
                        minimal_profile_in_prompt = raw_line
                    except IndexError:
                        pass
                elif 'Minimal Profile: "' in target_text:
                    try:
                        raw_line = target_text.split('Minimal Profile: "')[1].split("\n")[0].strip()
                        raw_line = raw_line.removesuffix('"')
                        minimal_profile_in_prompt = raw_line
                    except IndexError:
                        pass

                lead_in = (
                    f"{minimal_profile_in_prompt} "
                    if minimal_profile_in_prompt
                    else f'Cột "{col_name}" thuộc bảng {table_name}, giữ vai trò quản lý thông tin trong luồng nghiệp vụ SMB. '
                )
                summary = (
                    f"{lead_in}"
                    f"Kiểu dữ liệu và mẫu nhận diện: {format_str}. Dữ liệu được quản lý đồng nhất, phục vụ trực tiếp cho "
                    f"các câu truy vấn phân tích, kết hợp quy chuẩn tính toán và trích xuất theo quy ước nghiệp vụ."
                )
            results.append(summary)

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
        batch_size: int = 64,
    ) -> None:
        """Initialize ProfileLLMSummarizer with an LLM engine."""
        self.llm = llm_engine
        self.batch_size = batch_size

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
        samples = (col_stats.get("sample_values") or [])[:5]

        return MINIMAL_PROFILE_PROMPT.format(
            table_name=table_name,
            col_name=col_name,
            sibling_cols=sibling_cols,
            data_format=data_format,
            samples=samples,
            business_readme=business_readme.strip(),
        )

    def build_maximal_prompt(
        self,
        table_name: str,
        col_name: str,
        col_stats: ColumnProfile | dict[str, Any],
        total_rows: int,
        business_readme: str = "",
        minimal_profile: str = "",
        sibling_cols: list[str] | None = None,
    ) -> str:
        """Construct the Maximal Profile XML prompt with full business_readme."""
        null_count = col_stats.get("null_count", 0) or 0
        distinct_count = col_stats.get("distinct_count", 0) or 0
        min_val = col_stats.get("min_val", "N/A")
        if min_val is None:
            min_val = "N/A"
        max_val = col_stats.get("max_val", "N/A")
        if max_val is None:
            max_val = "N/A"
        data_format = col_stats.get("data_format", "STRING") or "STRING"

        subtype = (
            col_stats.get("date_subtype")
            or col_stats.get("currency_subtype")
            or "Standard"
        )

        return MAXIMAL_PROFILE_PROMPT.format(
            table_name=table_name,
            col_name=col_name,
            minimal_profile=minimal_profile.strip(),
            sibling_cols=sibling_cols or [],
            null_count=null_count,
            total_rows=total_rows,
            distinct_count=distinct_count,
            min_val=min_val,
            max_val=max_val,
            data_format=data_format,
            subtype_info=subtype,
            business_readme=business_readme.strip(),
        )

    def generate_profiles(
        self,
        db_profile: DatabaseProfile | dict[str, Any],
        business_readme: str = "",
    ) -> dict[str, Any]:
        """Generate Minimal and Maximal profiles for all columns across all tables.

        Follows a Two-Stage generation workflow:
        Stage 1: Generate Minimal Profiles (concise business meaning & abbreviation decoding).
        Stage 2: Inject Minimal Profiles into Maximal Prompts to generate in-depth summaries.
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
            return {}

        # Phase 2: Batch inference for minimal prompts (Stage 1)
        all_min_prompts = [item["min_prompt"] for item in work_items]
        min_summaries: list[str] = []
        for i in range(0, len(all_min_prompts), self.batch_size):
            chunk = all_min_prompts[i : i + self.batch_size]
            chunk_outs = self.llm.generate(
                chunk,
                max_tokens=160,
                stop=["\n\n", "</output>", "<|im_end|>", "<|endoftext|>"],
            )
            if isinstance(chunk_outs, str):
                chunk_outs = [chunk_outs]
            min_summaries.extend(chunk_outs)

        # Phase 3: Build and batch inference for maximal prompts (Stage 2: Two-stage inheritance)
        all_max_prompts: list[str] = []
        for idx, item in enumerate(work_items):
            min_summary = min_summaries[idx].strip()
            max_prompt = self.build_maximal_prompt(
                table_name=item["table_name"],
                col_name=item["col_name"],
                col_stats=item["col_stats"],
                total_rows=item["total_rows"],
                business_readme=readme_clean,
                minimal_profile=min_summary,
                sibling_cols=item["sibling_cols"],
            )
            all_max_prompts.append(max_prompt)

        max_summaries: list[str] = []
        for i in range(0, len(all_max_prompts), self.batch_size):
            chunk = all_max_prompts[i : i + self.batch_size]
            chunk_outs = self.llm.generate(
                chunk,
                max_tokens=450,
                stop=["\n\n\n", "</output>", "<|im_end|>", "<|endoftext|>"],
            )
            if isinstance(chunk_outs, str):
                chunk_outs = [chunk_outs]
            max_summaries.extend(chunk_outs)

        # Phase 4: Construct the final processed_metadata dictionary
        processed_metadata: dict[str, Any] = {}
        for idx, item in enumerate(work_items):
            col_key = item["col_key"]
            col_stats = item["col_stats"]
            short_summary = min_summaries[idx].strip()
            long_summary = max_summaries[idx].strip()

            top_10_list = list((col_stats.get("top_10_samples") or {}).keys())[:10]
            if top_10_list:
                top_str = f"Các giá trị phổ biến: {top_10_list}"
                maximal_profile = f"{long_summary} {top_str}" if long_summary else top_str
            else:
                maximal_profile = long_summary

            processed_metadata[col_key] = {
                "minimal_profile": short_summary,
                "maximal_profile": maximal_profile,
                "raw_samples": col_stats.get("sample_values") or [],
                # Enriched attributes preserved for downstream vector and schema operations
                "table_name": item["table_name"],
                "col_name": item["col_name"],
                "data_format": col_stats.get("data_format", "STRING") or "STRING",
                "date_subtype": col_stats.get("date_subtype"),
                "currency_subtype": col_stats.get("currency_subtype"),
                "null_count": col_stats.get("null_count", 0) or 0,
                "distinct_count": col_stats.get("distinct_count", 0) or 0,
                "top_10_samples": col_stats.get("top_10_samples") or {},
                "llm_hint": col_stats.get("llm_hint"),
            }

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
