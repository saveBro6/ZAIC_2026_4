# P0 + P1 — Hợp nhất prompt sửa lỗi với prompt sinh SQL

Artifact sống cho đợt sửa A6 (prompt sửa lỗi yếu hơn prompt gốc). Cập nhật trực tiếp file này khi tiến độ thay đổi: tick checklist, điền bảng metrics, append changelog.

## Trạng thái

| Hạng mục | Trạng thái |
|---|---|
| P0 — `sanitize_cdata` + `RepairContext` + nội dung `<repair_task>` | done |
| P1 — prefix dùng chung + override schema/rulebook + wrapper back-compat | done |
| Code Review Gate | **PASSED** (274 insertions / 79 deletions, không scope creep) |
| Lint (ruff) | passed (7 file, 0 error) |
| Đo M1–M5 | done |

## 1. Mục tiêu đo được

| # | Tiêu chí | Cách đo | Trước | Sau |
|---|---|---|---|---|
| M1 | 100% block `<previous_output>` là JSON hợp lệ | `json.loads` trong unit test | 0% (dùng `repr()`, nháy đơn) | 100% (`json.dumps` trong CDATA, `json.loads` khôi phục đúng byte) |
| M2 | Prompt sửa lỗi có **byte-identical prefix** với prompt sinh SQL (tới `</user_question>`) | `repair_prompt.startswith(prefix)` | không (thứ tự section khác) | `startswith=True`; 32.161 / 33.082 ký tự = **97,2%** prompt sửa lỗi được cache |
| M3 | Không còn prompt tự mâu thuẫn: output thật của model và SQL đang giữ hiển thị tách biệt | unit test với stub trả rewrite bị từ chối | sai (bản bị từ chối bị vứt) | đúng: prompt lần 3 có `<accepted_sql>SELECT nope FROM t;</accepted_sql>` + `<last_raw_output>` chứa rewrite bị từ chối |
| M4 | Prompt sửa lỗi có đủ contract type, rule SQL, `reasoning`, few-shot, rulebook | unit test snapshot | thiếu toàn bộ | đủ: `<repair_diagnostics>`, `<repair_contract>`, `few_shot_examples`, `split_rulebook`, `reasoning`, `NULLIF`/`IS NOT NULL` |
| M5 | Không hồi quy: test suite xanh, số lượt gọi LLM/candidate không tăng | `uv run pytest -q` | 41 passed, 1 failed (sandbox EROFS) | **201 passed**, 2 failed pre-existing (xem ghi chú); `format_rulebook_context` gọi **1 lần**/candidate (trước: 1 lần/lượt sinh + 1 lần/lượt sửa); số lượt gọi LLM không đổi |

Ghi chú M5 — 2 fail pre-existing, không nằm trong diff: `tests/test_profile_summarizer.py::test_default_constants` (`constants.py` đã là `Qwen/Qwen3.5-9B`, test cũ assert Qwen2.5-7B) và `tests/test_submission_pipeline.py::test_submission_pipeline_permission_fallback` (sandbox trả `OSError [Errno 30]` thay vì `PermissionError`).

Thêm: repair growth **911 ký tự** (fixture 40 cột + rulebook public_test) — dưới ngưỡng 2000 của test; CDATA cân bằng, không có `<|`, không có `&lt;=` (SQL `<=` giữ nguyên).

## 2. Ngoài phạm vi

`rules.py`/`ValidationIssue`, state machine chống dao động (stagnation/best-retention), ngưỡng abstain, sửa 3 check false-positive của validator, evidence injection (rows/literals), batch + early-exit, typed views ở ingest.

## 3. Ràng buộc đã kiểm chứng

- vLLM **0.33.0**; `enable_prefix_caching` **mặc định bật** (`CacheConfig.enable_prefix_caching = True`; `EngineArgs` để `None` rồi resolve qua `ModelConfig.is_prefix_caching_supported`, trả `True` cho decoder-only). Repo không truyền cờ này. Hệ quả: repair prompt dùng chung prefix **không tốn thêm prefill** ⇒ bỏ knob `repair_full_context`, luôn full parity.
- Prefix cache khớp theo token block (64) và đòi prefix **byte-identical** ⇒ `rulebook_xml` và `schema_xml` phải render **một lần** trong `generate_candidate` rồi truyền xuống cả hai lượt; gọi lại `vector_indexer.search_rules` ở lượt sửa có thể trả chunk khác ⇒ vừa miss cache vừa lệch nội dung luật.
- `sanitize_text` escape `<`, `>`, `&` ⇒ **không dùng cho SQL** (`<=`, `<>`, `&` sẽ thành entity và model dễ echo lại). SQL phải đi qua `sanitize_cdata`.
- Thẻ lỗi đổi tên thành `<repair_diagnostics>`; `MockLLMEngine`/mock engine trong test route theo chuỗi này nên phải cập nhật đồng thời.

## 4. Thiết kế

### 4.1 `sanitize_cdata` (common/xml_sanitizer.py)

```python
def sanitize_cdata(val: Any, max_len: int = 4000) -> str:
    """Sanitize text for embedding inside a CDATA section.

    Strips C0 control characters, defangs ChatML markers, neutralises the CDATA
    terminator so the section cannot be closed early, and truncates long payloads.
    XML entities are deliberately NOT escaped: raw SQL must stay byte-identical.
    """
```

Quy tắc: `clean_c0_and_newlines` → `defang_chatml` → `]]>` thành `]] >` → cap `max_len`.

### 4.2 `RepairContext` (submission/prompt_template.py)

```python
@dataclass(frozen=True)
class RepairContext:
    """Repair-specific inputs appended after the shared generation prefix."""

    diagnostics: list[str]
    previous_sql: str = ""
    previous_type: str = "number"
    accepted_sql: str = ""
    last_output: str = ""
    max_chars: int = 4000
```

### 4.3 Prefix dùng chung

- `_build_prompt_prefix(question, as_of_date, schema_xml, rulebook_xml, grounded_entities, include_few_shot) -> str` — render đúng chuỗi hiện tại, kết thúc ở `</user_question>`.
- `build_sql_generation_prompt(...)`: thêm `repair`, `schema_xml_override`, `rulebook_xml_override`.
  - không repair: `f"{prefix}\n\n<output>"` (byte-identical với bản hiện tại)
  - có repair: `f"{prefix}\n\n{repair_task}\n\n<output>"`
- `_format_repair_task(repair) -> str` render section dưới đây.

### 4.4 Nội dung `<repair_task>`

```xml
<repair_task>
<accepted_sql><![CDATA[SELECT …]]></accepted_sql>
<previous_output><![CDATA[{"sql": "SELECT …", "type": "number"}]]></previous_output>
<last_raw_output><![CDATA[…]]></last_raw_output>
<repair_diagnostics>
- …
</repair_diagnostics>
<repair_contract>
Chỉ sửa tối thiểu để giải quyết đúng các mục trong <repair_diagnostics>.
Giữ nguyên logic thời gian, bộ lọc, JOIN và thứ tự sắp xếp đang đúng trong <accepted_sql>.
Chỉ dùng bảng/cột có trong <selected_schema>; không thêm cột không tồn tại.
Trả lại đúng định dạng JSON {"reasoning": "…", "sql": "…", "type": "…"} như lượt sinh SQL.
Nếu <accepted_sql> đã đúng, trả lại nguyên văn câu lệnh đó.
Nếu dữ liệu phát hành không đủ để trả lời, trả {"sql": "", "type": "unanswerable"}.
</repair_contract>
</repair_task>
```

- `<previous_output>` luôn là `json.dumps({"sql": previous_sql, "type": previous_type}, ensure_ascii=False)` bọc CDATA ⇒ M1 luôn đạt.
- `<last_raw_output>` chỉ phát khi `last_output` khác rỗng và khác JSON ở trên.
- Nội dung CDATA đi qua `sanitize_cdata(..., max_len=repair.max_chars)`.

### 4.5 Wrapper back-compat

`build_sql_correction_prompt` giữ nguyên chữ ký cũ, thêm keyword tùy chọn `last_output`, `accepted_sql`, `rulebook_text`, `vector_indexer`, `rulebook_mode`, `include_few_shot`; nội bộ dựng `RepairContext` rồi gọi `build_sql_generation_prompt(..., schema_xml_override=schema_xml, repair=repair)`; docstring ghi `Deprecated`.

## 5. Thay đổi theo file

- [x] `docs/p0_p1_repair_prompt_plan.md` — artifact này
- [x] `src/zaic_2026_4/common/xml_sanitizer.py` — `sanitize_cdata` + `__all__`
- [x] `src/zaic_2026_4/submission/prompt_template.py` — `RepairContext`, `_build_prompt_prefix`, `_format_repair_task`, 3 tham số mới, wrapper mới, `__all__`
- [x] `src/zaic_2026_4/submission/candidate_generator.py` — render rulebook một lần, lưu `last_raw_output`, 2 call site dựng `RepairContext`, prefix guard
- [x] `src/zaic_2026_4/submission/__init__.py` — export `RepairContext`
- [x] `tests/test_xml_sanitizer.py` — 3 test `sanitize_cdata`
- [x] `tests/test_repair_prompt.py` (mới) — 7 test M1–M4 + back-compat
- [x] `tests/test_candidate_generator.py` — mock routing sang `<repair_diagnostics>`, sửa assert prompt cũ

## 6. Test plan

`tests/test_repair_prompt.py`:

1. `test_previous_output_is_valid_json`
2. `test_prefix_identical_to_generation_prompt`
3. `test_repair_contract_and_parity`
4. `test_cdata_cannot_break_structure`
5. `test_correction_wrapper_backward_compatible`
6. `test_rejected_attempt_is_shown`
7. `test_repair_prompt_growth_bounded`

`tests/test_xml_sanitizer.py`: `test_sanitize_cdata_strips_c0`, `test_sanitize_cdata_neutralises_terminator`, `test_sanitize_cdata_caps_length`.

## 7. Trình tự & gate

1. Artifact (done)
2. `sanitize_cdata` + test
3. `RepairContext` + prefix refactor + wrapper + export
4. Call site `candidate_generator`
5. Test mới + cập nhật test cũ
6. Code Review Gate (`code-reviewer`, phải `[PASSED]`)
7. `uv run ruff check` (không chạy lại pytest)
8. Đo M1–M5, cập nhật artifact
9. Walkthrough + commit message

Commit message dự kiến: `fix(submission): unify repair prompt with generation prefix and emit valid JSON previous output`

## 8. Decisions log

| # | Ngày | Quyết định |
|---|---|---|
| D1 | 2026-11-12 | ~~Giữ tên thẻ `<validation_errors>`~~ — **thay bởi D6** |
| D2 | 2026-11-12 | Bỏ knob `repair_full_context`; luôn full parity vì prefix caching mặc định bật |
| D3 | 2026-11-12 | Luôn phát cả `<accepted_sql>` và `<previous_output>` để shape prompt tất định |
| D4 | 2026-11-12 | SQL trong prompt đi qua `sanitize_cdata`, không qua `sanitize_text` |
| D5 | 2026-11-12 | Không đổi byte prompt sinh SQL khi không truyền override/repair |
| D6 | 2026-11-12 | Đổi `<validation_errors>` → `<repair_diagnostics>`; cập nhật mock routing + assert trong `tests/test_candidate_generator.py` |

## 9. Rủi ro & giảm thiểu

| Rủi ro | Giảm thiểu |
|---|---|
| Model echo `&lt;=` do escape | CDATA + test 4 |
| Prefix lệch ⇒ miss cache, lệch nội dung luật | render rulebook một lần + override + runtime `startswith` warning |
| Repair prompt dài thêm ~1k ký tự | test 7 chặn phình; prefix cache bù prefill |
| Model quên "sửa tối thiểu" | `<repair_contract>` + test 6 |
| Hồi quy hành vi | M5: pytest xanh, số lượt gọi LLM/candidate không tăng |

## 10. Theo dõi sau P0/P1 (không chặn, cho P2/P3)

| # | Việc | Nguồn |
|---|---|---|
| F1 | Chạy `sanitize_cdata` + cap độ dài cho chính `diagnostics` (hiện chỉ SQL được sanitize; thông điệp lỗi SQLite có thể chứa định danh do model sinh) | review finding B |
| F2 | `sanitize_cdata` gộp mọi khoảng trắng nên SQL nhiều dòng thành 1 dòng; nếu cần giữ layout, thêm biến thể collapse mềm | review finding C |
| F3 | Ở vòng static, `<accepted_sql>` có thể chứa bản sửa vừa fail validation (semantics cũ của vòng lặp); state machine P3 sẽ loại bỏ | review finding D |
| F4 | Rename `accepted_sql`/`previous_sql` trùng nhau về mặt khái niệm khi vòng static — xem lại khi làm P3 | review finding D |

## 11. Changelog

| Ngày | Thay đổi |
|---|---|
| 2026-11-12 | Tạo artifact từ plan đã duyệt; ghi nhận D6 (rename thẻ) |
| 2026-11-12 | Hoàn tất P0+P1: gate `[PASSED]`, ruff sạch (fix FURB167 `re.S` → `re.DOTALL` trong test), metrics M1–M5 đã điền |
