"""Module 2: LLM-powered database profile summarization using vLLM and Qwen2.5.

Generates Minimal Profiles (Short Summaries for Schema Linking) and
Maximal Profiles (Long Summaries for Code Synthesis & FAISS Vector Indexing).
"""

import os
from typing import Any, Protocol, runtime_checkable

# Disable FlashInfer top-p/top-k sampler by default to prevent JIT compilation failures on NVCC < 12.2
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

from zaic_2026_4.profiling.raw_profiler import ColumnProfile, DatabaseProfile

DEFAULT_PROD_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_LOCAL_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_PROD_GPU_UTIL = 0.95
DEFAULT_LOCAL_GPU_UTIL = 0.60


MINIMAL_PROFILE_PROMPT = """<system>
You are an expert data analyst for Vietnamese SMB (Small and Medium Business) databases.
Your objective is to generate a concise, 1-sentence SHORT SUMMARY in Vietnamese describing the exact business meaning and basic data format of the target column.
</system>

<instructions>
1. Business Meaning: Describe what business entity, attribute, or transaction property this column represents. Decode Vietnamese SMB abbreviations if present (e.g., SP -> Sản phẩm, KH -> Khách hàng, ĐVT -> Đơn vị tính, HĐ -> Hóa đơn).
2. Data Format: Specify the underlying physical and logical format (e.g., Alphanumeric code, ISO date, VND currency with thousand separators, categorical status flag, integer count).
3. Brevity Constraint: Output EXACTLY ONE concise Vietnamese sentence. Do not add boilerplate, preambles, or conversational prefixes.
</instructions>

<business_readme>
{business_readme}
</business_readme>

<few_shot_examples>
  <example id="1">
    <target_column>
      Table: "DonHang"
      Column: "ma_kh"
      Sibling Columns: ["id", "ngay_tao", "tong_tien", "trang_thai"]
      Nulls: 0/15000 (0.00%)
      Distinct Count: 3200
      Format & Hints: STRING | Code identifier
      Sample Values: ["KH0001", "KH0002", "KH0158", "VANG_LAI"]
    </target_column>
    <output>
      Mã định danh khách hàng (KH: Khách Hàng, VANG_LAI: khách mua vãng lai không lưu hồ sơ); định dạng chuỗi mã ký tự chữ và số.
    </output>
  </example>

  <example id="2">
    <target_column>
      Table: "ChiTietDonHang"
      Column: "thanh_tien"
      Sibling Columns: ["ma_sp", "so_luong", "don_gia", "chiet_khau"]
      Nulls: 0/45000 (0.00%)
      Distinct Count: 1250
      Format & Hints: CURRENCY | Formatted VND currency. Clean thousand separators before aggregations.
      Sample Values: ["12,000", "45,000", "150,000", "0"]
    </target_column>
    <output>
      Tổng số tiền thực thu của dòng sản phẩm sau khi trừ chiết khấu dòng; định dạng số tiền VNĐ có dấu phân cách hàng nghìn cần làm sạch trước khi tính toán.
    </output>
  </example>
</few_shot_examples>

<target_column>
  Table: "{table_name}"
  Column: "{col_name}"
  Sibling Columns: {sibling_cols}
  Nulls: {null_count}/{total_rows} ({null_ratio:.2%})
  Distinct Count: {distinct_count}
  Format & Hints: {data_format} | {llm_hint}
  Sample Values: {samples}
</target_column>

<output>
"""


MAXIMAL_PROFILE_PROMPT = """<system>
You are an expert data architect and SQL engineer for Vietnamese SMB databases.
Your objective is to generate an in-depth, structured LONG SUMMARY in Vietnamese for the target column to guide downstream Text-to-SQL generation and semantic vector retrieval.
</system>

<instructions>
Include the following 4 structural aspects in Vietnamese:
1. Business Meaning: Operational role in the business workflow, entity relationship, and relation to sibling columns.
2. Data Cleanliness & Distribution: Completeness (null count and ratio), cardinality (distinct count), value boundaries ([min_val to max_val]).
3. Dominant Patterns: Frequent categories from Top-10 samples and percentage dominance.
4. Technical SQL / Query Guidance: Specific data cleaning, casting instructions (e.g. handling formatted VND strings, parsing DD/MM/YYYY dates to ISO YYYY-MM-DD), and business status filtering.
</instructions>

<business_readme>
{business_readme}
</business_readme>

<few_shot_examples>
  <example id="1">
    <target_column>
      Table: "SoBanHang"
      Column: "Trạng thái xử lý"
      Nulls: 526/43605 (1.21%)
      Distinct Count: 4
      Value Range: ["Không giao được" to "đang gửi"]
      Format & Pattern: STRING (Categorical status)
      Top 10 Samples: {{"Đã hoàn thành": 41248, "Không giao được": 1163, "trả hàng": 660, "đang gửi": 8}}
      Code / SQL Hint: Categorical status string. Exclude cancelled/returned orders for revenue calculation per business rules.
    </target_column>
    <output>
      Cột lưu trữ trạng thái xử lý đơn hàng trong sổ bán hàng, đóng vai trò cờ phân loại giao dịch (status flag). Dữ liệu có 4 giá trị phân loại với 1.21% giá trị NULL; giá trị chiếm đa số tuyệt đối là 'Đã hoàn thành' (94.6%), tiếp theo là 'Không giao được' và 'trả hàng'. Khi truy vấn tính toán doanh thu hoặc sản phẩm bán chạy theo quy ước nghiệp vụ, BẮT BUỘC phải lọc loại bỏ các trạng thái huỷ/hoàn ('trả hàng', 'Không giao được') bằng điều kiện: WHERE "Trạng thái xử lý" = 'Đã hoàn thành'.
    </output>
  </example>

  <example id="2">
    <target_column>
      Table: "HoaDon"
      Column: "ngay_lap"
      Nulls: 0/50000 (0.00%)
      Distinct Count: 730
      Value Range: ["01/01/2024" to "31/12/2025"]
      Format & Pattern: DATE (SLASH_DATE_DD_MM_YYYY)
      Top 10 Samples: {{"15/01/2025": 140, "20/01/2025": 135, "05/02/2025": 128}}
      Code / SQL Hint: Date in DD/MM/YYYY format. Convert to ISO YYYY-MM-DD for SQL temporal filtering.
    </target_column>
    <output>
      Cột ghi nhận mốc thời gian lập hóa đơn bán lẻ của doanh nghiệp. Cột không có giá trị NULL (100% đầy đủ) với 730 ngày giao dịch liên tục từ 01/01/2024 đến 31/12/2025. Định dạng ngày tháng lưu trữ là chuỗi DD/MM/YYYY (Việt Nam), do đó khi thực hiện truy vấn so sánh thời gian, lọc theo mốc as_of_date hoặc liên kết với calendar.csv, cần chuẩn hóa sang chuẩn ISO YYYY-MM-DD hoặc sử dụng hàm phân tích ngày: SUBSTR(ngay_lap, 7, 4) || '-' || SUBSTR(ngay_lap, 4, 2) || '-' || SUBSTR(ngay_lap, 1, 2).
    </output>
  </example>
</few_shot_examples>

<target_column>
  Table: "{table_name}"
  Column: "{col_name}"
  Nulls: {null_count}/{total_rows} ({null_ratio:.2%})
  Distinct Count: {distinct_count}
  Value Range: [{min_val} to {max_val}]
  Format & Pattern: {data_format} ({subtype_info})
  Top 10 Samples: {top_10}
  Code / SQL Hint: {llm_hint}
</target_column>

<output>
"""


@runtime_checkable
class LLMEngineProtocol(Protocol):
    """Protocol for offline inference engines supporting batched generation."""

    def generate(
        self,
        prompts: list[str] | str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        stop: list[str] | None = None,
    ) -> list[str] | str:
        """Generate text completions for a single prompt or batch of prompts."""
        ...


class VLLMEngine:
    """Wrapper around vLLM offline engine with continuous batching support."""

    def __init__(
        self,
        model: str = DEFAULT_PROD_MODEL,
        gpu_memory_utilization: float = DEFAULT_PROD_GPU_UTIL,
        max_model_len: int = 4096,
        tensor_parallel_size: int = 1,
        trust_remote_code: bool = True,
        dtype: str = "auto",
        **kwargs: Any,
    ) -> None:
        """Initialize vLLM engine instance."""
        from vllm import LLM

        self.model_name = model
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.tensor_parallel_size = tensor_parallel_size

        self.llm = LLM(
            model=model,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=trust_remote_code,
            dtype=dtype,
            **kwargs,
        )

    def generate(
        self,
        prompts: list[str] | str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        stop: list[str] | None = None,
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

        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop_tokens,
        )

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

            # Check whether it is a minimal or maximal prompt
            is_minimal = "SHORT SUMMARY" in prompt

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
            if 'Column: "' in target_text:
                try:
                    col_name = target_text.split('Column: "')[1].split('"')[0]
                except IndexError:
                    pass

            # Extract data format
            format_str = "STRING"
            if "Format & Hints: " in target_text:
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
                    summary = f"Cột ngày tháng giao dịch ({col_name}) trong bảng {table_name}; định dạng thời gian cần chuẩn hóa ISO YYYY-MM-DD khi truy vấn."
                elif any(k in col_lower for k in ("tien", "tiền", "gia", "giá", "cost", "price", "amount")):
                    summary = f"Cột giá trị tiền tệ ({col_name}) trong bảng {table_name}; định dạng số tiền VNĐ cần làm sạch dấu phân cách trước khi tính toán tổng/bình quân."
                elif any(k in col_lower for k in ("ma", "mã", "id", "code", "sku")):
                    summary = f"Mã định danh nghiệp vụ ({col_name}) trong bảng {table_name}; định dạng chuỗi mã số/ký tự chữ và số."
                elif any(k in col_lower for k in ("trang_thai", "trạng thái", "status", "ket_qua", "kết quả")):
                    summary = f"Trạng thái xử lý giao dịch ({col_name}) trong bảng {table_name}; cờ phân loại cần lọc theo quy ước nghiệp vụ."
                else:
                    summary = f"Thuộc tính thông tin {col_name} trong bảng {table_name}; kiểu dữ liệu {format_str}."
            else:
                summary = (
                    f"Cột {col_name} thuộc bảng {table_name}, giữ vai trò quản lý thông tin trong luồng nghiệp vụ SMB. "
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
    **kwargs: Any,
) -> LLMEngineProtocol:
    """Factory creating the appropriate LLM engine based on target environment.

    - Default (competition container with --gpus all):
        Model: Qwen/Qwen2.5-7B-Instruct, gpu_memory_utilization: 0.95
    - Local mode (--local):
        Model: Qwen/Qwen2.5-1.5B-Instruct, gpu_memory_utilization: 0.60
    - Mock mode (mock=True):
        Deterministic rule-based MockLLMEngine (instant execution without GPU/network).
    """
    if mock:
        return MockLLMEngine(**kwargs)

    resolved_model = model or (DEFAULT_LOCAL_MODEL if local else DEFAULT_PROD_MODEL)
    resolved_gpu_util = (
        gpu_memory_utilization
        if gpu_memory_utilization is not None
        else (DEFAULT_LOCAL_GPU_UTIL if local else DEFAULT_PROD_GPU_UTIL)
    )

    return VLLMEngine(
        model=resolved_model,
        gpu_memory_utilization=resolved_gpu_util,
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
        total_rows: int,
        business_readme: str = "",
    ) -> str:
        """Construct the Minimal Profile XML prompt with full business_readme."""
        null_count = col_stats.get("null_count", 0) or 0
        null_ratio = (null_count / total_rows) if total_rows > 0 else 0.0
        distinct_count = col_stats.get("distinct_count", 0) or 0
        data_format = col_stats.get("data_format", "STRING") or "STRING"
        llm_hint = col_stats.get("llm_hint") or "Standard tabular column"

        samples = (col_stats.get("sample_values") or [])[:5]

        return MINIMAL_PROFILE_PROMPT.format(
            table_name=table_name,
            col_name=col_name,
            sibling_cols=sibling_cols,
            null_count=null_count,
            total_rows=total_rows,
            null_ratio=null_ratio,
            distinct_count=distinct_count,
            data_format=data_format,
            llm_hint=llm_hint,
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
    ) -> str:
        """Construct the Maximal Profile XML prompt with full business_readme."""
        null_count = col_stats.get("null_count", 0) or 0
        null_ratio = (null_count / total_rows) if total_rows > 0 else 0.0
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
        llm_hint = col_stats.get("llm_hint") or "Standard tabular column"
        top_10 = col_stats.get("top_10_samples") or {}
        top_10_samples = dict(list(top_10.items())[:6])

        return MAXIMAL_PROFILE_PROMPT.format(
            table_name=table_name,
            col_name=col_name,
            null_count=null_count,
            total_rows=total_rows,
            null_ratio=null_ratio,
            distinct_count=distinct_count,
            min_val=min_val,
            max_val=max_val,
            data_format=data_format,
            subtype_info=subtype,
            top_10=top_10_samples,
            llm_hint=llm_hint,
            business_readme=business_readme.strip(),
        )

    def generate_profiles(
        self,
        db_profile: DatabaseProfile | dict[str, Any],
        business_readme: str = "",
    ) -> dict[str, Any]:
        """Generate Minimal and Maximal profiles for all columns across all tables.

        Accepts business_readme as the sole domain documentation context in strict accordance
        with code_template.txt. Leverages batched inference to send all column prompts to vLLM concurrently.
        """
        readme_clean = business_readme.strip()

        # Phase 1: Collect prompt metadata for all columns
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
                max_prompt = self.build_maximal_prompt(
                    table_name=table_name,
                    col_name=col_name,
                    col_stats=col_stats,
                    total_rows=total_rows,
                    business_readme=readme_clean,
                )
                work_items.append({
                    "col_key": f"{table_name}.{col_name}",
                    "table_name": table_name,
                    "col_name": col_name,
                    "col_stats": col_stats,
                    "min_prompt": min_prompt,
                    "max_prompt": max_prompt,
                })

        if not work_items:
            return {}

        # Phase 2: Batch inference for minimal and maximal prompts
        all_min_prompts = [item["min_prompt"] for item in work_items]
        all_max_prompts = [item["max_prompt"] for item in work_items]

        # Generate minimal profiles (1 sentence, constrained max_tokens)
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

        # Generate maximal profiles (detailed summary)
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

        # Phase 3: Construct the final processed_metadata dictionary
        processed_metadata: dict[str, Any] = {}
        for idx, item in enumerate(work_items):
            col_key = item["col_key"]
            col_stats = item["col_stats"]
            short_summary = min_summaries[idx].strip()
            long_summary = max_summaries[idx].strip()

            processed_metadata[col_key] = {
                "minimal_profile": short_summary,
                "maximal_profile": long_summary,
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
